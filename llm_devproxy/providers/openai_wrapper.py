"""
OpenAI provider wrapper.
Intercepts openai.OpenAI() calls transparently.
"""

from typing import Any, Optional
from ..core.engine import Engine
from ..core.models import ProxyConfig


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

        def real_api():
            response = self._client.chat.completions.create(**kwargs)
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
                },
                "model": response.model,
            }
            return (
                response_body,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        response_body, record = self._engine.call(
            provider="openai",
            model=model,
            request_body=kwargs,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        # 元のopenaiオブジェクト形式に近いオブジェクトを返す
        return _OpenAIResponse(response_body)


class _OpenAIResponse:
    """openai.ChatCompletion風のレスポンスオブジェクト"""
    def __init__(self, body: dict):
        self._body = body
        self.choices = [_Choice(c) for c in body.get("choices", [])]
        usage = body.get("usage", {})
        self.usage = _Usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
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
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
