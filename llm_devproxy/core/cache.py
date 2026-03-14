"""
Cache Manager - returns stored responses for identical or similar requests.
Supports exact match (hash) and semantic similarity (embeddings).
Eliminates redundant API calls in CI/CD and repeated dev runs.
"""

import hashlib
import json
import logging
from typing import Optional

from .models import ProxyConfig, RequestRecord
from .storage import Storage

logger = logging.getLogger(__name__)


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
    def __init__(self, storage: Storage, config: Optional[ProxyConfig] = None,
                 enabled: bool = True):
        self.storage = storage
        self.enabled = enabled
        self.config = config or ProxyConfig()
        self._semantic: Optional["SemanticCacheManager"] = None

        # セマンティックキャッシュの遅延初期化
        if self.config.semantic_cache:
            self._init_semantic()

    def _init_semantic(self):
        """SemanticCacheManagerを初期化（importも遅延）。"""
        try:
            from .semantic_cache import SemanticCacheManager
            self._semantic = SemanticCacheManager(self.config, self.storage)
            logger.info(
                "Semantic cache enabled (backend=%s, threshold=%.2f)",
                self.config.semantic_backend,
                self.config.similarity_threshold,
            )
        except ImportError as e:
            logger.warning("Semantic cache disabled: %s", e)
            self._semantic = None

    @property
    def semantic(self) -> Optional["SemanticCacheManager"]:
        return self._semantic

    def get(
        self, provider: str, model: str, request_body: dict
    ) -> Optional[RequestRecord]:
        """
        キャッシュを検索。cache_modeに応じて:
        - "exact": 完全一致のみ
        - "semantic": セマンティックのみ
        - "both": 完全一致 → fallback でセマンティック
        """
        if not self.enabled:
            return None

        mode = self.config.cache_mode

        # Step 1: 完全一致チェック（exactまたはbothモード）
        if mode in ("exact", "both"):
            prompt_hash = _make_hash(provider, model, request_body)
            cached = self.storage.find_cached(prompt_hash, model)
            if cached:
                logger.debug("Exact cache hit: %s", cached.id)
                return cached

        # Step 2: セマンティック検索（semanticまたはbothモード）
        if mode in ("semantic", "both") and self._semantic:
            result = self._semantic.find_similar(provider, model, request_body)
            if result:
                record, similarity = result
                logger.info(
                    "Semantic cache hit (sim=%.3f): %s",
                    similarity, record.id,
                )
                return record

        return None

    def store_semantic(
        self, record: RequestRecord, request_body: dict
    ) -> None:
        """レスポンス記録後にセマンティックembeddingを保存。"""
        if self._semantic and not record.is_cached:
            try:
                self._semantic.store_embedding(record, request_body)
            except Exception as e:
                logger.warning("Failed to store semantic embedding: %s", e)

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
