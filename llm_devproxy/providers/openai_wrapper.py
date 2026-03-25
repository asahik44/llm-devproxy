"""
OpenAI provider wrapper.
Intercepts openai.OpenAI() calls transparently.
Supports both regular and streaming responses.
"""

from typing import Any, Optional
from ..core.engine import Engine
from ..core.models import ProxyConfig, RequestRecord
from ..core.cost_guard import estimate_cost
from datetime import datetime, timezone


class OpenAIWrapper:
    """
    openai.OpenAI() のドロップイン代替。
    既存コードのclientを差し替えるだけで動く。

    Usage:
        # Before
        client = openai.OpenAI(api_key="sk-...")

        # After
        from llm_devproxy import DevProxy
        proxy = DevProxy()
        client = proxy.wrap_openai(openai.OpenAI(api_key="sk-..."))
    """

    def __init__(self, client: Any, engine: Engine, session_id: Optional[str] = None):
        self._client = client
        self._engine = engine
        self._session_id = session_id
        self.chat = _ChatCompletions(client, engine, session_id)


class _ChatCompletions:
    def __init__(self, client: Any, engine: Engine, session_id: Optional[str]):
        self._client = client
        self._engine = engine
        self._session_id = session_id
        self.completions = self

    def create(self, **kwargs) -> Any:
        model = kwargs.get("model", "gpt-4o")
        is_stream = kwargs.get("stream", False)

        if is_stream:
            return self._create_stream(model, kwargs)
        else:
            return self._create_normal(model, kwargs)

    def _create_normal(self, model: str, kwargs: dict) -> Any:
        """通常（non-streaming）レスポンス"""
        def real_api():
            response = self._client.chat.completions.create(**kwargs)

            reasoning_tokens = 0
            completion_details = getattr(response.usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

            response_body = {
                "choices": [
                    {
                        "message": {
                            "role": c.message.role,
                            "content": c.message.content,
                        },
                        "finish_reason": c.finish_reason,
                    }
                    for c in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                },
                "model": response.model,
            }
            return (
                response_body,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                reasoning_tokens,
            )

        response_body, record = self._engine.call(
            provider="openai",
            model=model,
            request_body=kwargs,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        return _OpenAIResponse(response_body)

    def _create_stream(self, model: str, kwargs: dict) -> "_OpenAIStreamWrapper":
        """Streaming レスポンス。チャンクをそのまま返しつつ、完了後に記録。"""
        # キャッシュチェック
        cached = self._engine.cache.get("openai", model, kwargs)
        if cached:
            session = self._engine.get_or_create_session()
            self._engine._step_counter += 1
            record = self._engine.cache.build_cached_record(
                cached, session.id, self._engine._step_counter
            )
            self._engine.storage.save_request(record)
            # キャッシュヒット時は疑似ストリーム
            return _OpenAICachedStream(cached.response_body)

        # 実ストリームを開始
        stream = self._client.chat.completions.create(**kwargs)
        return _OpenAIStreamWrapper(
            stream=stream,
            engine=self._engine,
            model=model,
            request_body=kwargs,
            session_id=self._session_id,
        )


class _OpenAIStreamWrapper:
    """
    OpenAI streaming レスポンスのラッパー。
    チャンクをそのままyieldしつつ、ストリーム完了後にEngineに記録。
    """

    def __init__(self, stream, engine: Engine, model: str,
                 request_body: dict, session_id: Optional[str]):
        self._stream = stream
        self._engine = engine
        self._model = model
        self._request_body = request_body
        self._session_id = session_id
        self._content_parts: list[str] = []
        self._finish_reason: str = "stop"
        self._usage_data: dict = {}

    def __iter__(self):
        return self._iterate()

    def _iterate(self):
        for chunk in self._stream:
            # コンテンツ蓄積
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    self._content_parts.append(delta.content)
                if chunk.choices[0].finish_reason:
                    self._finish_reason = chunk.choices[0].finish_reason

            # usage（OpenAI stream_options={"include_usage": True} の場合）
            if hasattr(chunk, "usage") and chunk.usage:
                self._usage_data = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                }
                details = getattr(chunk.usage, "completion_tokens_details", None)
                if details:
                    self._usage_data["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0

            yield chunk

        # ストリーム完了 → 記録
        self._record()

    def _record(self):
        full_content = "".join(self._content_parts)

        # usage が stream 内で取れなかった場合は文字数から概算
        prompt_tokens = self._usage_data.get("prompt_tokens", 0)
        completion_tokens = self._usage_data.get("completion_tokens", 0)
        reasoning_tokens = self._usage_data.get("reasoning_tokens", 0)

        if completion_tokens == 0 and full_content:
            completion_tokens = len(full_content) // 4  # 概算

        response_body = {
            "choices": [{
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": self._finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "model": self._model,
            "streamed": True,
        }

        def already_done():
            return (response_body, prompt_tokens, completion_tokens, reasoning_tokens)

        self._engine.call(
            provider="openai",
            model=self._model,
            request_body=self._request_body,
            real_api_fn=already_done,
            session_id=self._session_id,
        )


class _OpenAICachedStream:
    """キャッシュヒット時に疑似ストリームを返すラッパー。"""

    def __init__(self, response_body: dict):
        self._body = response_body

    def __iter__(self):
        content = ""
        choices = self._body.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        # 1チャンクで全コンテンツを返す
        yield _FakeChunk(content)


class _FakeChunk:
    """キャッシュ用の疑似チャンク。"""
    def __init__(self, content: str):
        self.choices = [_FakeChunkChoice(content)]
        self.usage = None

class _FakeChunkChoice:
    def __init__(self, content: str):
        self.delta = _FakeDelta(content)
        self.finish_reason = "stop"

class _FakeDelta:
    def __init__(self, content: str):
        self.content = content
        self.role = None


class _OpenAIResponse:
    """openai.ChatCompletion風のレスポンスオブジェクト"""
    def __init__(self, body: dict):
        self._body = body
        self.choices = [_Choice(c) for c in body.get("choices", [])]
        usage = body.get("usage", {})
        self.usage = _Usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("reasoning_tokens", 0),
        )
        self.model = body.get("model", "")


class _Choice:
    def __init__(self, data: dict):
        self.message = _Message(data.get("message", {}))
        self.finish_reason = data.get("finish_reason", "stop")


class _Message:
    def __init__(self, data: dict):
        self.role = data.get("role", "assistant")
        self.content = data.get("content", "")


class _Usage:
    def __init__(self, prompt: int, completion: int, reasoning: int = 0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.reasoning_tokens = reasoning
        self.total_tokens = prompt + completion
