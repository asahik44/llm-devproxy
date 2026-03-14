"""
Tests for Semantic Cache - covers normalize_prompt, cosine_similarity,
and SemanticCacheManager integration across OpenAI, Anthropic, and Gemini formats.
"""

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import numpy as np

from llm_devproxy.core.models import ProxyConfig, RequestRecord
from llm_devproxy.core.storage import Storage
from llm_devproxy.core.cache import CacheManager
from llm_devproxy.core.semantic_cache import (
    SemanticCacheManager,
    normalize_prompt,
    cosine_similarity,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_db():
    """一時DBパスを返す。テスト終了後に自動削除。"""
    tmp_dir = tempfile.mkdtemp()
    return str(Path(tmp_dir) / "test.db")


@pytest.fixture
def storage(tmp_db):
    return Storage(tmp_db)


@pytest.fixture
def semantic_config(tmp_db):
    return ProxyConfig(
        db_path=tmp_db,
        cache_enabled=True,
        cache_mode="both",
        semantic_cache=True,
        semantic_backend="local",
        similarity_threshold=0.85,
    )


@pytest.fixture
def cache_manager(storage, semantic_config):
    return CacheManager(storage, semantic_config, enabled=True)


def _make_record(
    storage: Storage,
    provider: str,
    model: str,
    request_body: dict,
    response_text: str = "dummy response",
    prompt_hash: str = "",
) -> RequestRecord:
    """テスト用のダミーRecordを作成してDBに保存。"""
    record = RequestRecord(
        id=str(uuid.uuid4()),
        session_id="test-session",
        step_id=1,
        provider=provider,
        model=model,
        prompt_hash=prompt_hash or str(uuid.uuid4()),
        request_body=request_body,
        response_body={"choices": [{"message": {"content": response_text}}]},
        input_tokens=50,
        output_tokens=100,
        cost_usd=0.001,
        is_cached=False,
    )
    storage.save_request(record)
    return record


# ============================================================
# Test: normalize_prompt - OpenAI format
# ============================================================

class TestNormalizePromptOpenAI:
    """OpenAI形式のリクエストを正しく正規化できるか。"""

    def test_simple_message(self):
        body = {"messages": [{"role": "user", "content": "hello world"}]}
        result = normalize_prompt(body)
        assert "[user] hello world" in result

    def test_system_message_in_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "hello"},
            ]
        }
        result = normalize_prompt(body)
        assert "[system] You are a helpful assistant" in result
        assert "[user] hello" in result

    def test_multi_turn(self):
        body = {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a language."},
                {"role": "user", "content": "Show me an example."},
            ]
        }
        result = normalize_prompt(body)
        assert "[user] What is Python?" in result
        assert "[assistant] Python is a language." in result
        assert "[user] Show me an example." in result

    def test_vision_content_blocks(self):
        """OpenAI vision: contentがlist型（textとimage_urlの混在）。"""
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            }]
        }
        result = normalize_prompt(body)
        assert "[user] What is in this image?" in result

    def test_tool_call_none_content(self):
        """tool_callでcontentがNoneの場合にクラッシュしない。"""
        body = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None},  # tool_call
                {"role": "user", "content": "Thanks"},
            ]
        }
        result = normalize_prompt(body)
        assert "[user] What's the weather?" in result
        assert "[user] Thanks" in result
        assert "None" not in result


# ============================================================
# Test: normalize_prompt - Anthropic format
# ============================================================

class TestNormalizePromptAnthropic:
    """Anthropic形式のリクエストを正しく正規化できるか。"""

    def test_system_as_string(self):
        body = {
            "system": "You are a coding assistant",
            "messages": [{"role": "user", "content": "Write fizzbuzz"}],
        }
        result = normalize_prompt(body)
        assert "[system] You are a coding assistant" in result
        assert "[user] Write fizzbuzz" in result

    def test_system_as_blocks(self):
        """Anthropic: systemがblock list形式。"""
        body = {
            "system": [
                {"type": "text", "text": "You are an expert."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Explain quantum computing"}],
        }
        result = normalize_prompt(body)
        assert "[system] You are an expert. Be concise." in result

    def test_content_blocks(self):
        """Anthropic: messagesのcontentがblock list形式。"""
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this data:"},
                    {"type": "text", "text": "Revenue is $1M"},
                ],
            }]
        }
        result = normalize_prompt(body)
        assert "[user] Analyze this data: Revenue is $1M" in result


# ============================================================
# Test: normalize_prompt - Gemini format
# ============================================================

class TestNormalizePromptGemini:
    """Gemini形式のリクエストを正しく正規化できるか。"""

    def test_simple_contents(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "What is machine learning?"}]},
            ]
        }
        result = normalize_prompt(body)
        assert "[user] What is machine learning?" in result

    def test_multi_turn_contents(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi there!"}]},
                {"role": "user", "parts": [{"text": "Tell me a joke"}]},
            ]
        }
        result = normalize_prompt(body)
        assert "[user] Hello" in result
        assert "[model] Hi there!" in result
        assert "[user] Tell me a joke" in result

    def test_system_instruction(self):
        body = {
            "systemInstruction": {
                "parts": [{"text": "You are a math tutor"}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": "Solve x^2 = 4"}]},
            ],
        }
        result = normalize_prompt(body)
        assert "[system] You are a math tutor" in result
        assert "[user] Solve x^2 = 4" in result

    def test_multi_part(self):
        """Gemini: 1メッセージに複数partsがある場合。"""
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Look at this image."},
                    {"text": "What do you see?"},
                ],
            }]
        }
        result = normalize_prompt(body)
        assert "[user] Look at this image. What do you see?" in result

    def test_empty_contents(self):
        body = {"contents": []}
        result = normalize_prompt(body)
        assert result == ""


# ============================================================
# Test: normalize_prompt - Edge cases
# ============================================================

class TestNormalizePromptEdgeCases:

    def test_empty_body(self):
        assert normalize_prompt({}) == ""

    def test_empty_messages(self):
        assert normalize_prompt({"messages": []}) == ""

    def test_mixed_format_ignored(self):
        """messagesとcontentsが両方あっても両方抽出される。"""
        body = {
            "messages": [{"role": "user", "content": "from messages"}],
            "contents": [{"role": "user", "parts": [{"text": "from contents"}]}],
        }
        result = normalize_prompt(body)
        assert "from messages" in result
        assert "from contents" in result


# ============================================================
# Test: Model Presets
# ============================================================

class TestModelPresets:
    """プリセット名でモデルが正しく解決されるか。"""

    def test_preset_multilingual(self):
        from llm_devproxy.core.semantic_cache import LocalEmbeddingBackend
        backend = LocalEmbeddingBackend("multilingual")
        assert backend._model_name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_preset_english(self):
        from llm_devproxy.core.semantic_cache import LocalEmbeddingBackend
        backend = LocalEmbeddingBackend("english")
        assert backend._model_name == "all-MiniLM-L6-v2"

    def test_preset_empty_uses_default(self):
        from llm_devproxy.core.semantic_cache import LocalEmbeddingBackend
        backend = LocalEmbeddingBackend("")
        assert backend._model_name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_custom_model_name_passthrough(self):
        from llm_devproxy.core.semantic_cache import LocalEmbeddingBackend
        backend = LocalEmbeddingBackend("all-mpnet-base-v2")
        assert backend._model_name == "all-mpnet-base-v2"


# ============================================================
# Test: cosine_similarity
# ============================================================

class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


# ============================================================
# Test: SemanticCacheManager integration
# ============================================================

class TestSemanticCacheIntegration:
    """SemanticCacheManagerの統合テスト（実際にembeddingを生成）。"""

    def test_store_and_find_similar_openai(self, storage, cache_manager):
        """OpenAI形式: 類似プロンプトでキャッシュヒットする。"""
        original = {
            "messages": [{"role": "user", "content": "How to sort a list in Python"}],
        }
        similar = {
            "messages": [{"role": "user", "content": "How do I sort a Python list?"}],
        }
        record = _make_record(storage, "openai", "gpt-4o-mini", original)
        cache_manager.store_semantic(record, original)

        result = cache_manager.semantic.find_similar("openai", "gpt-4o-mini", similar)
        assert result is not None
        found_record, similarity = result
        assert found_record.id == record.id
        assert similarity >= 0.85

    def test_store_and_find_similar_anthropic(self, storage, cache_manager):
        """Anthropic形式: system + content blocks。"""
        original = {
            "system": "You are a Python expert",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Explain list comprehension"}],
            }],
        }
        similar = {
            "system": "You are a Python expert",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "How do list comprehensions work?"}],
            }],
        }
        record = _make_record(storage, "anthropic", "claude-sonnet-4-20250514", original)
        cache_manager.store_semantic(record, original)

        result = cache_manager.semantic.find_similar(
            "anthropic", "claude-sonnet-4-20250514", similar
        )
        assert result is not None
        _, similarity = result
        assert similarity >= 0.80  # system promptが同じなので高めになるはず

    def test_store_and_find_similar_gemini(self, storage, cache_manager):
        """Gemini形式: contents/parts構造。"""
        original = {
            "contents": [
                {"role": "user", "parts": [{"text": "Recommend good ramen restaurants in Tokyo"}]},
            ],
        }
        similar = {
            "contents": [
                {"role": "user", "parts": [{"text": "Where can I find delicious ramen in Tokyo?"}]},
            ],
        }
        record = _make_record(storage, "gemini", "gemini-2.0-flash", original)
        cache_manager.store_semantic(record, original)

        result = cache_manager.semantic.find_similar(
            "gemini", "gemini-2.0-flash", similar
        )
        assert result is not None
        _, similarity = result
        assert similarity >= 0.80

    def test_different_content_no_hit(self, storage, cache_manager):
        """全く異なる内容はヒットしない。"""
        original = {
            "messages": [{"role": "user", "content": "Pythonでフィボナッチ数列を計算する"}],
        }
        different = {
            "messages": [{"role": "user", "content": "明日の東京の天気を教えて"}],
        }
        record = _make_record(storage, "openai", "gpt-4o-mini", original)
        cache_manager.store_semantic(record, original)

        result = cache_manager.semantic.find_similar(
            "openai", "gpt-4o-mini", different
        )
        assert result is None

    def test_cross_provider_isolation(self, storage, cache_manager):
        """プロバイダーが違えばヒットしない。"""
        body = {
            "messages": [{"role": "user", "content": "What is Python?"}],
        }
        record = _make_record(storage, "openai", "gpt-4o-mini", body)
        cache_manager.store_semantic(record, body)

        # 同じ内容でもprovider=anthropicで検索するとヒットしない
        result = cache_manager.semantic.find_similar(
            "anthropic", "gpt-4o-mini", body
        )
        assert result is None

    def test_cross_model_isolation(self, storage, cache_manager):
        """モデルが違えばヒットしない。"""
        body = {
            "messages": [{"role": "user", "content": "What is Python?"}],
        }
        record = _make_record(storage, "openai", "gpt-4o-mini", body)
        cache_manager.store_semantic(record, body)

        result = cache_manager.semantic.find_similar(
            "openai", "gpt-4o", body
        )
        assert result is None


# ============================================================
# Test: CacheManager.get() integration (exact → semantic fallback)
# ============================================================

class TestCacheManagerFallback:
    """cache.get() で exact → semantic のフォールバックが動くか。"""

    def test_exact_hit_takes_priority(self, storage, cache_manager):
        """完全一致がある場合はセマンティック検索に行かない。"""
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        }
        prompt_hash = cache_manager.make_hash("openai", "gpt-4o-mini", body)
        record = _make_record(
            storage, "openai", "gpt-4o-mini", body,
            prompt_hash=prompt_hash,
        )
        cache_manager.store_semantic(record, body)

        # 完全に同じリクエスト → exact hit
        hit = cache_manager.get("openai", "gpt-4o-mini", body)
        assert hit is not None
        assert hit.id == record.id

    def test_semantic_fallback_on_exact_miss(self, storage, cache_manager):
        """完全一致がない場合、セマンティックにフォールバックする。"""
        original = {
            "messages": [{"role": "user", "content": "Explain recursion in Python"}],
            "temperature": 0.7,
        }
        similar = {
            "messages": [{"role": "user", "content": "How does Python recursion work?"}],
            "temperature": 0.7,
        }
        prompt_hash = cache_manager.make_hash("openai", "gpt-4o-mini", original)
        record = _make_record(
            storage, "openai", "gpt-4o-mini", original,
            prompt_hash=prompt_hash,
        )
        cache_manager.store_semantic(record, original)

        # 類似だが完全一致しない → semantic fallback
        hit = cache_manager.get("openai", "gpt-4o-mini", similar)
        assert hit is not None
        assert hit.id == record.id

    def test_no_hit_at_all(self, storage, cache_manager):
        """完全一致もセマンティックもない場合はNone。"""
        body = {
            "messages": [{"role": "user", "content": "What's the capital of France?"}],
        }
        hit = cache_manager.get("openai", "gpt-4o-mini", body)
        assert hit is None


# ============================================================
# Test: Storage helper methods (used by semantic cache)
# ============================================================

class TestStorageHelpers:

    def test_execute_sql_and_fetch_all(self, storage):
        storage.execute_sql(
            "CREATE TABLE IF NOT EXISTS test_tbl (id TEXT, val TEXT)"
        )
        storage.execute_sql(
            "INSERT INTO test_tbl VALUES (?, ?)", ("a", "hello")
        )
        storage.execute_sql(
            "INSERT INTO test_tbl VALUES (?, ?)", ("b", "world")
        )
        rows = storage.fetch_all("SELECT * FROM test_tbl ORDER BY id")
        assert len(rows) == 2
        assert rows[0] == ("a", "hello")
        assert rows[1] == ("b", "world")

    def test_find_by_id(self, storage):
        record = _make_record(
            storage, "openai", "gpt-4o-mini",
            {"messages": [{"role": "user", "content": "test"}]},
        )
        found = storage.find_by_id(record.id)
        assert found is not None
        assert found.id == record.id
        assert found.provider == "openai"

    def test_find_by_id_not_found(self, storage):
        found = storage.find_by_id("nonexistent-id")
        assert found is None