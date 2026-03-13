"""
基本的な動作テスト
"""

import pytest
from llm_devproxy.core import Storage
from llm_devproxy.core.models import RequestRecord, Session


def test_storage_save_and_retrieve_session(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="test_session")
    storage.save_session(session)
    retrieved = storage.get_session(session.id)
    assert retrieved is not None
    assert retrieved.name == "test_session"


def test_storage_save_and_cache_request(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="s1")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="abc123",
        request_body={"messages": [{"role": "user", "content": "Hello"}]},
        response_body={"choices": [{"message": {"content": "Hi!"}}]},
        input_tokens=10, output_tokens=5, cost_usd=0.00005,
    )
    storage.save_request(record)
    cached = storage.find_cached("abc123", "gpt-4o")
    assert cached is not None
    assert cached.input_tokens == 10


def test_storage_search_japanese(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="s1")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="xyz",
        request_body={"messages": [{"role": "user", "content": "化合物を教えて"}]},
        response_body={},
    )
    storage.save_request(record)
    results = storage.search_requests("化合物")
    assert len(results) == 1


def test_cost_guard_allows_under_limit(tmp_path):
    from llm_devproxy.core.models import ProxyConfig
    from llm_devproxy.core.cost_guard import CostGuard
    storage = Storage(str(tmp_path / "test.db"))
    guard = CostGuard(ProxyConfig(daily_limit_usd=1.0), storage)
    allowed, reason = guard.check("session_1")
    assert allowed is True


def test_cost_guard_blocks_over_limit(tmp_path):
    from llm_devproxy.core.models import ProxyConfig
    from llm_devproxy.core.cost_guard import CostGuard
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="expensive")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="h1", request_body={}, response_body={},
        cost_usd=2.0,
    )
    storage.save_request(record)
    guard = CostGuard(ProxyConfig(daily_limit_usd=1.0), storage)
    allowed, reason = guard.check(session.id)
    assert allowed is False
    assert "Daily limit" in reason


def test_cache_hit(tmp_path):
    from llm_devproxy.core.cache import CacheManager, _make_hash
    storage = Storage(str(tmp_path / "test.db"))
    cache = CacheManager(storage, enabled=True)
    session = Session(name="s1")
    storage.save_session(session)
    provider, model = "openai", "gpt-4o"
    request_body = {"messages": [{"role": "user", "content": "test"}], "temperature": 0}
    prompt_hash = _make_hash(provider, model, request_body)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider=provider, model=model,
        prompt_hash=prompt_hash,
        request_body=request_body,
        response_body={"choices": [{"message": {"content": "cached!"}}]},
    )
    storage.save_request(record)
    hit = cache.get(provider, model, request_body)
    assert hit is not None
    assert hit.response_body["choices"][0]["message"]["content"] == "cached!"


def test_cache_miss(tmp_path):
    from llm_devproxy.core.cache import CacheManager
    storage = Storage(str(tmp_path / "test.db"))
    cache = CacheManager(storage, enabled=True)
    hit = cache.get("openai", "gpt-4o", {"messages": [{"role": "user", "content": "never seen"}]})
    assert hit is None


def test_estimate_cost():
    from llm_devproxy.core.cost_guard import estimate_cost
    cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.0025 + 0.010, rel=1e-3)
    cost_unknown = estimate_cost("unknown-model-xyz", 1000, 1000)
    assert cost_unknown > 0
