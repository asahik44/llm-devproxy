"""
Anthropic provider wrapper.
Supports both regular and streaming responses.
"""

from typing import Any, Optional
from ..core.engine import Engine


class AnthropicWrapper:
    """
    anthropic.Anthropic() のドロップイン代替。

    Usage:
        # Before
        client = anthropic.Anthropic(api_key="...")

        # After
        client = proxy.wrap_anthropic(anthropic.Anthropic(api_key="..."))
    """

    def __init__(self, client: Any, engine: Engine, session_id: Optional[str] = None):
        self._client = client
        self._engine = engine
        self._session_id = session_id
        self.messages = _Messages(client, engine, session_id)


class _Messages:
    def __init__(self, client: Any, engine: Engine, session_id: Optional[str]):
        self._client = client
        self._engine = engine
        self._session_id = session_id

    def create(self, **kwargs) -> Any:
        model = kwargs.get("model", "claude-sonnet-4-5")
        is_stream = kwargs.get("stream", False)

        if is_stream:
            return self._create_stream(model, kwargs)
        else:
            return self._create_normal(model, kwargs)

    def _create_normal(self, model: str, kwargs: dict) -> Any:
        """通常レスポンス"""
        def real_api():
            response = self._client.messages.create(**kwargs)

            reasoning_tokens = 0
            thinking_blocks = [
                b for b in response.content
                if hasattr(b, "type") and b.type == "thinking"
            ]
            if thinking_blocks:
                for tb in thinking_blocks:
                    if hasattr(tb, "thinking"):
                        reasoning_tokens += len(tb.thinking) // 3

            response_body = {
                "content": [
                    {"type": b.type, "text": getattr(b, "text", getattr(b, "thinking", ""))}
                    for b in response.content
                    if hasattr(b, "text") or hasattr(b, "thinking")
                ],
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                },
            }
            return (
                response_body,
                response.usage.input_tokens,
                response.usage.output_tokens,
                reasoning_tokens,
            )

        response_body, record = self._engine.call(
            provider="anthropic",
            model=model,
            request_body=kwargs,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        return _AnthropicResponse(response_body)

    def _create_stream(self, model: str, kwargs: dict) -> "_AnthropicStreamWrapper":
        """Streaming レスポンス"""
        # キャッシュチェック
        cached = self._engine.cache.get("anthropic", model, kwargs)
        if cached:
            session = self._engine.get_or_create_session()
            self._engine._step_counter += 1
            record = self._engine.cache.build_cached_record(
                cached, session.id, self._engine._step_counter
            )
            self._engine.storage.save_request(record)
            return _AnthropicCachedStream(cached.response_body)

        stream = self._client.messages.create(**kwargs)
        return _AnthropicStreamWrapper(
            stream=stream,
            engine=self._engine,
            model=model,
            request_body=kwargs,
            session_id=self._session_id,
        )


class _AnthropicStreamWrapper:
    """Anthropic streaming レスポンスのラッパー。"""

    def __init__(self, stream, engine: Engine, model: str,
                 request_body: dict, session_id: Optional[str]):
        self._stream = stream
        self._engine = engine
        self._model = model
        self._request_body = request_body
        self._session_id = session_id
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def __iter__(self):
        return self._iterate()

    def _iterate(self):
        for event in self._stream:
            # イベントからテキストを蓄積
            event_type = getattr(event, "type", "")

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta:
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        self._content_parts.append(text)
                    elif delta_type == "thinking_delta":
                        thinking = getattr(delta, "thinking", "")
                        self._thinking_parts.append(thinking)

            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    self._output_tokens = getattr(usage, "output_tokens", 0)

            elif event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg:
                    usage = getattr(msg, "usage", None)
                    if usage:
                        self._input_tokens = getattr(usage, "input_tokens", 0)

            yield event

        # ストリーム完了 → 記録
        self._record()

    def _record(self):
        full_content = "".join(self._content_parts)
        full_thinking = "".join(self._thinking_parts)
        reasoning_tokens = len(full_thinking) // 3 if full_thinking else 0

        if self._output_tokens == 0 and full_content:
            self._output_tokens = len(full_content) // 4

        content_blocks = []
        if full_thinking:
            content_blocks.append({"type": "thinking", "text": full_thinking})
        if full_content:
            content_blocks.append({"type": "text", "text": full_content})

        response_body = {
            "content": content_blocks,
            "model": self._model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "streamed": True,
        }

        def already_done():
            return (response_body, self._input_tokens, self._output_tokens, reasoning_tokens)

        self._engine.call(
            provider="anthropic",
            model=self._model,
            request_body=self._request_body,
            real_api_fn=already_done,
            session_id=self._session_id,
        )


class _AnthropicCachedStream:
    """キャッシュヒット時の疑似ストリーム。"""
    def __init__(self, response_body: dict):
        self._body = response_body

    def __iter__(self):
        for block in self._body.get("content", []):
            yield _FakeAnthropicEvent(block.get("text", ""))


class _FakeAnthropicEvent:
    def __init__(self, text: str):
        self.type = "content_block_delta"
        self.delta = _FakeAnthropicDelta(text)

class _FakeAnthropicDelta:
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


class _AnthropicResponse:
    """anthropic.Message風のレスポンスオブジェクト"""
    def __init__(self, body: dict):
        self._body = body
        self.content = [_ContentBlock(b) for b in body.get("content", [])]
        self.model = body.get("model", "")
        self.stop_reason = body.get("stop_reason", "end_turn")
        usage = body.get("usage", {})
        self.usage = _AnthropicUsage(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("reasoning_tokens", 0),
        )

    @property
    def text(self) -> str:
        """最初のtextブロックのテキストを返す便利プロパティ"""
        for block in self.content:
            if block.type == "text":
                return block.text
        return ""


class _ContentBlock:
    def __init__(self, data: dict):
        self.type = data.get("type", "text")
        self.text = data.get("text", "")


class _AnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
