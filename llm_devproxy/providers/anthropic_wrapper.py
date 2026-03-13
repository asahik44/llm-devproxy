"""
Anthropic provider wrapper.
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

        def real_api():
            response = self._client.messages.create(**kwargs)
            response_body = {
                "content": [
                    {"type": b.type, "text": b.text}
                    for b in response.content
                    if hasattr(b, "text")
                ],
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
            return (
                response_body,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

        response_body, record = self._engine.call(
            provider="anthropic",
            model=model,
            request_body=kwargs,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        return _AnthropicResponse(response_body)


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
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
