"""
Google Gemini provider wrapper.
Supports google-generativeai SDK.
"""

from typing import Any, Optional
from ..core.engine import Engine


class GeminiWrapper:
    """
    google.generativeai.GenerativeModel のドロップイン代替。

    Usage:
        import google.generativeai as genai

        # Before
        model = genai.GenerativeModel("gemini-1.5-flash")

        # After
        from llm_devproxy import DevProxy
        proxy = DevProxy()
        model = proxy.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))
    """

    def __init__(self, model: Any, engine: Engine, session_id: Optional[str] = None):
        self._model = model
        self._engine = engine
        self._session_id = session_id
        # モデル名を取得（SDKのattribute名に対応）
        self._model_name = getattr(model, "model_name", "gemini-1.5-flash")

    def generate_content(self, contents: Any, **kwargs) -> Any:
        """
        model.generate_content() のラッパー。
        contentsは文字列またはリスト形式に対応。
        """
        # request_bodyを統一フォーマットに変換
        if isinstance(contents, str):
            messages = [{"role": "user", "content": contents}]
        elif isinstance(contents, list):
            messages = self._normalize_contents(contents)
        else:
            messages = [{"role": "user", "content": str(contents)}]

        request_body = {
            "model": self._model_name,
            "messages": messages,
            **kwargs,
        }

        def real_api():
            response = self._model.generate_content(contents, **kwargs)
            # Gemini レスポンスを統一フォーマットに変換
            text = response.text if hasattr(response, "text") else ""
            response_body = {
                "choices": [{"message": {"role": "model", "content": text}}],
                "model": self._model_name,
                "usage": self._extract_usage(response),
            }
            usage = response_body["usage"]
            return response_body, usage["prompt_tokens"], usage["completion_tokens"]

        response_body, record = self._engine.call(
            provider="gemini",
            model=self._model_name,
            request_body=request_body,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        return _GeminiResponse(response_body)

    def _normalize_contents(self, contents: list) -> list:
        """Gemini形式のcontentsをmessages形式に変換"""
        messages = []
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                parts = item.get("parts", [])
                content = parts[0] if parts and isinstance(parts[0], str) else str(parts)
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages

    def _extract_usage(self, response: Any) -> dict:
        """Geminiレスポンスからtoken使用量を取得"""
        try:
            meta = response.usage_metadata
            return {
                "prompt_tokens": meta.prompt_token_count,
                "completion_tokens": meta.candidates_token_count,
            }
        except Exception:
            # usage_metadataがない場合はテキスト長で概算
            text = getattr(response, "text", "")
            estimated = len(text) // 4
            return {"prompt_tokens": 0, "completion_tokens": estimated}


class _GeminiResponse:
    """google.generativeai.GenerateContentResponse 風オブジェクト"""
    def __init__(self, body: dict):
        self._body = body
        choices = body.get("choices", [{}])
        self.text = choices[0].get("message", {}).get("content", "") if choices else ""
        self.model = body.get("model", "")
        usage = body.get("usage", {})
        self.usage_metadata = _GeminiUsage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    @property
    def parts(self):
        return [_Part(self.text)]

    @property
    def candidates(self):
        return [_Candidate(self.text)]


class _Part:
    def __init__(self, text: str):
        self.text = text


class _Candidate:
    def __init__(self, text: str):
        self.content = _CandidateContent(text)


class _CandidateContent:
    def __init__(self, text: str):
        self.parts = [_Part(text)]


class _GeminiUsage:
    def __init__(self, prompt_tokens: int, candidates_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.total_token_count = prompt_tokens + candidates_tokens
