"""
Google Gemini provider wrapper.
Supports google-generativeai SDK with streaming.
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
        self._model_name = getattr(model, "model_name", "gemini-1.5-flash")

    def generate_content(self, contents: Any, stream: bool = False, **kwargs) -> Any:
        """
        model.generate_content() のラッパー。
        contentsは文字列またはリスト形式に対応。
        """
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

        if stream:
            return self._generate_stream(contents, request_body, kwargs)
        else:
            return self._generate_normal(contents, request_body, kwargs)

    def _generate_normal(self, contents: Any, request_body: dict, kwargs: dict) -> Any:
        def real_api():
            response = self._model.generate_content(contents, **kwargs)
            text = response.text if hasattr(response, "text") else ""
            usage = self._extract_usage(response)
            response_body = {
                "choices": [{"message": {"role": "model", "content": text}}],
                "model": self._model_name,
                "usage": usage,
            }
            return (
                response_body,
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["reasoning_tokens"],
            )

        response_body, record = self._engine.call(
            provider="gemini",
            model=self._model_name,
            request_body=request_body,
            real_api_fn=real_api,
            session_id=self._session_id,
        )

        return _GeminiResponse(response_body)

    def _generate_stream(self, contents: Any, request_body: dict, kwargs: dict):
        """Streaming レスポンス。"""
        cached = self._engine.cache.get("gemini", self._model_name, request_body)
        if cached:
            session = self._engine.get_or_create_session()
            self._engine._step_counter += 1
            record = self._engine.cache.build_cached_record(
                cached, session.id, self._engine._step_counter
            )
            self._engine.storage.save_request(record)
            return _GeminiCachedStream(cached.response_body)

        stream = self._model.generate_content(contents, stream=True, **kwargs)
        return _GeminiStreamWrapper(
            stream=stream,
            engine=self._engine,
            model_name=self._model_name,
            request_body=request_body,
            session_id=self._session_id,
        )

    def _normalize_contents(self, contents: list) -> list:
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
        """Geminiレスポンスからtoken使用量を取得（thinking対応）"""
        try:
            meta = response.usage_metadata
            reasoning_tokens = getattr(meta, "thoughts_token_count", 0) or 0
            return {
                "prompt_tokens": meta.prompt_token_count,
                "completion_tokens": meta.candidates_token_count,
                "reasoning_tokens": reasoning_tokens,
            }
        except Exception:
            text = getattr(response, "text", "")
            estimated = len(text) // 4
            return {"prompt_tokens": 0, "completion_tokens": estimated, "reasoning_tokens": 0}


class _GeminiStreamWrapper:
    """Gemini streaming ラッパー。"""

    def __init__(self, stream, engine: Engine, model_name: str,
                 request_body: dict, session_id: Optional[str]):
        self._stream = stream
        self._engine = engine
        self._model_name = model_name
        self._request_body = request_body
        self._session_id = session_id
        self._content_parts: list[str] = []
        self._usage: dict = {}

    def __iter__(self):
        return self._iterate()

    def _iterate(self):
        for chunk in self._stream:
            text = ""
            if hasattr(chunk, "text"):
                text = chunk.text
                self._content_parts.append(text)

            # usage_metadata from final chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                meta = chunk.usage_metadata
                self._usage = {
                    "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
                    "reasoning_tokens": getattr(meta, "thoughts_token_count", 0) or 0,
                }

            yield chunk

        self._record()

    def _record(self):
        full_content = "".join(self._content_parts)
        prompt_tokens = self._usage.get("prompt_tokens", 0)
        completion_tokens = self._usage.get("completion_tokens", 0)
        reasoning_tokens = self._usage.get("reasoning_tokens", 0)

        if completion_tokens == 0 and full_content:
            completion_tokens = len(full_content) // 4

        response_body = {
            "choices": [{"message": {"role": "model", "content": full_content}}],
            "model": self._model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "streamed": True,
        }

        def already_done():
            return (response_body, prompt_tokens, completion_tokens, reasoning_tokens)

        self._engine.call(
            provider="gemini",
            model=self._model_name,
            request_body=self._request_body,
            real_api_fn=already_done,
            session_id=self._session_id,
        )


class _GeminiCachedStream:
    """キャッシュヒット時の疑似ストリーム。"""
    def __init__(self, response_body: dict):
        choices = response_body.get("choices", [{}])
        self._text = choices[0].get("message", {}).get("content", "") if choices else ""

    def __iter__(self):
        yield _FakeGeminiChunk(self._text)


class _FakeGeminiChunk:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None
        self.parts = [_Part(text)]
        self.candidates = [_Candidate(text)]


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
