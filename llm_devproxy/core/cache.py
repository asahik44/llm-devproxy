"""
Cache Manager - returns stored responses for identical requests.
Eliminates redundant API calls in CI/CD and repeated dev runs.
"""

import hashlib
import json
from typing import Optional

from .models import RequestRecord
from .storage import Storage


def _make_hash(provider: str, model: str, request_body: dict) -> str:
    """
    プロバイダー + モデル + リクエスト内容からキャッシュキーを生成。
    temperatureなど再現性に影響するパラメータも含める。
    """
    key_data = {
        "provider": provider,
        "model": model,
        "messages": request_body.get("messages", []),
        "temperature": request_body.get("temperature", 1.0),
        "max_tokens": request_body.get("max_tokens"),
        "system": request_body.get("system", ""),
    }
    serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


class CacheManager:
    def __init__(self, storage: Storage, enabled: bool = True):
        self.storage = storage
        self.enabled = enabled

    def get(
        self, provider: str, model: str, request_body: dict
    ) -> Optional[RequestRecord]:
        """キャッシュヒットがあれば返す。なければNone。"""
        if not self.enabled:
            return None

        prompt_hash = _make_hash(provider, model, request_body)
        cached = self.storage.find_cached(prompt_hash, model)
        return cached

    def make_hash(
        self, provider: str, model: str, request_body: dict
    ) -> str:
        return _make_hash(provider, model, request_body)

    def build_cached_record(
        self, original: RequestRecord, session_id: str, step_id: int
    ) -> RequestRecord:
        """キャッシュヒット時の新しいRecordを生成（コスト0で記録）"""
        import uuid
        from datetime import datetime, timezone
        return RequestRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            step_id=step_id,
            parent_id=original.id,
            branch_name="main",
            timestamp=datetime.now(timezone.utc),
            provider=original.provider,
            model=original.model,
            prompt_hash=original.prompt_hash,
            request_body=original.request_body,
            response_body=original.response_body,
            input_tokens=original.input_tokens,
            output_tokens=original.output_tokens,
            cost_usd=0.0,          # キャッシュからなのでコスト0
            is_cached=True,
        )
