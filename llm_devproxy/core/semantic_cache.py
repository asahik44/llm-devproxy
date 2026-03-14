"""
Semantic Cache - find similar (not identical) past requests.
Supports both local sentence-transformers and OpenAI embeddings API.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .models import ProxyConfig, RequestRecord
from .storage import Storage

logger = logging.getLogger(__name__)


# ============================================================
# Embedding Backends
# ============================================================

class EmbeddingBackend(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbeddingBackend(EmbeddingBackend):
    """sentence-transformers (local, free, heavier)."""

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # プリセット: ユーザーはフルモデル名を覚えなくてOK
    MODEL_PRESETS = {
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 50+言語, ~470MB
        "english": "all-MiniLM-L6-v2",                            # 英語特化, ~80MB, 高速
    }

    def __init__(self, model_name: str = ""):
        resolved = self.MODEL_PRESETS.get(model_name, model_name)
        self._model_name = resolved or self.DEFAULT_MODEL
        self._model = None  # lazy load

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local semantic cache.\n"
                    "Install it with: pip install llm-devproxy[semantic-local]"
                )
            logger.info("Loading local embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI Embeddings API (lightweight, costs money)."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = ""):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI semantic cache.\n"
                    "Install it with: pip install llm-devproxy[openai]"
                )
            self._client = OpenAI()
        return self._client

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return resp.data[0].embedding

    def dimension(self) -> int:
        return self.DIMENSIONS.get(self._model_name, 1536)


# ============================================================
# Prompt Normalizer
# ============================================================

def normalize_prompt(request_body: dict) -> str:
    """
    リクエストからセマンティック比較用テキストを抽出。
    3プロバイダーのフォーマットに対応:
      - OpenAI:    messages[].content (str or list of content parts)
      - Anthropic: system (str or list) + messages[].content (str or list of blocks)
      - Gemini:    contents[].parts[].text + systemInstruction.parts[].text
    temperature等のパラメータは無視する（意味が同じなら同じ）。
    """
    parts: list[str] = []

    # ── system prompt ──
    # Anthropic: "system" (str or list of blocks)
    system = request_body.get("system", "")
    if isinstance(system, list):
        # Anthropic style: [{"type": "text", "text": "..."}]
        system = " ".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if system:
        parts.append(f"[system] {system}")

    # Gemini: "systemInstruction.parts[].text"
    sys_instruction = request_body.get("systemInstruction", {})
    if isinstance(sys_instruction, dict):
        sys_parts = sys_instruction.get("parts", [])
        sys_text = " ".join(
            p.get("text", "") for p in sys_parts if isinstance(p, dict)
        )
        if sys_text.strip():
            parts.append(f"[system] {sys_text}")

    # ── messages (OpenAI / Anthropic) ──
    messages = request_body.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if content is None:
            # tool_call等でcontentがNoneの場合はスキップ
            continue

        if isinstance(content, list):
            # Anthropic: [{"type": "text", "text": "..."}]
            # OpenAI vision: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts)

        if isinstance(content, str) and content.strip():
            parts.append(f"[{role}] {content}")

    # ── contents (Gemini) ──
    contents = request_body.get("contents", [])
    for entry in contents:
        role = entry.get("role", "user")
        entry_parts = entry.get("parts", [])
        text_pieces = []
        for p in entry_parts:
            if isinstance(p, dict) and "text" in p:
                text_pieces.append(p["text"])
            elif isinstance(p, str):
                text_pieces.append(p)
        if text_pieces:
            parts.append(f"[{role}] {' '.join(text_pieces)}")

    return "\n".join(parts)


# ============================================================
# Cosine Similarity (numpy, no extra deps)
# ============================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ============================================================
# Semantic Cache Manager
# ============================================================

class SemanticCacheManager:
    """
    セマンティックキャッシュの統合マネージャー。
    既存の Storage と連携して、embeddingの保存・検索を行う。
    """

    def __init__(self, config: ProxyConfig, storage: Storage):
        self.config = config
        self.storage = storage
        self.backend = self._create_backend()

        # storage側にembeddingsテーブルがなければ作成
        self._ensure_table()

    def _create_backend(self) -> EmbeddingBackend:
        if self.config.semantic_backend == "local":
            return LocalEmbeddingBackend(self.config.semantic_model)
        elif self.config.semantic_backend == "openai":
            return OpenAIEmbeddingBackend(self.config.semantic_model)
        else:
            raise ValueError(
                f"Unknown semantic_backend: {self.config.semantic_backend}. "
                f"Use 'local' or 'openai'."
            )

    def _ensure_table(self):
        """embeddings用テーブルを作成（なければ）。"""
        self.storage.execute_sql("""
            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                record_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.storage.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_semantic_provider_model
            ON semantic_embeddings (provider, model)
        """)

    # ----- 保存 -----

    def store_embedding(
        self, record: RequestRecord, request_body: dict
    ) -> None:
        """リクエストの embedding を生成して保存。"""
        prompt_text = normalize_prompt(request_body)
        if not prompt_text.strip():
            return

        embedding = self.backend.embed(prompt_text)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        self.storage.execute_sql(
            """
            INSERT OR REPLACE INTO semantic_embeddings
            (record_id, provider, model, prompt_text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record.id, record.provider, record.model,
             prompt_text, embedding_blob),
        )

    # ----- 検索 -----

    def find_similar(
        self,
        provider: str,
        model: str,
        request_body: dict,
    ) -> Optional[tuple[RequestRecord, float]]:
        """
        セマンティックに類似するキャッシュを検索。
        閾値以上で最も類似度が高いものを返す。
        Returns: (record, similarity) or None
        """
        prompt_text = normalize_prompt(request_body)
        if not prompt_text.strip():
            return None

        query_embedding = self.backend.embed(prompt_text)

        # 同じprovider+modelの候補を取得
        rows = self.storage.fetch_all(
            """
            SELECT record_id, embedding
            FROM semantic_embeddings
            WHERE provider = ? AND model = ?
            """,
            (provider, model),
        )

        if not rows:
            return None

        # 全候補とコサイン類似度を計算
        best_id: Optional[str] = None
        best_sim: float = 0.0

        for row in rows:
            record_id = row[0]
            stored_blob = row[1]
            stored_vec = np.frombuffer(stored_blob, dtype=np.float32).tolist()
            sim = cosine_similarity(query_embedding, stored_vec)

            if sim > best_sim:
                best_sim = sim
                best_id = record_id

        # 閾値チェック
        if best_sim < self.config.similarity_threshold:
            logger.debug(
                "Semantic cache miss (best=%.3f < threshold=%.3f)",
                best_sim, self.config.similarity_threshold,
            )
            return None

        # RequestRecordを取得
        record = self.storage.find_by_id(best_id)
        if record is None:
            return None

        logger.info(
            "Semantic cache hit! similarity=%.3f record=%s",
            best_sim, best_id,
        )
        return (record, best_sim)