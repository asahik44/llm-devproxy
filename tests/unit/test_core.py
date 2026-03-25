"""
基本的な動作テスト
"""

import pytest
import json
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


def test_estimate_cost_with_reasoning():
    from llm_devproxy.core.cost_guard import estimate_cost, is_reasoning_model
    # o1 with reasoning tokens
    cost = estimate_cost("o1", input_tokens=1000, output_tokens=500, reasoning_tokens=2000)
    expected = 1000/1000*0.015 + 500/1000*0.060 + 2000/1000*0.060
    assert cost == pytest.approx(expected, rel=1e-3)
    assert is_reasoning_model("o1") is True
    assert is_reasoning_model("gpt-4o") is False


def test_storage_reasoning_tokens(tmp_path):
    """reasoning_tokens がDB保存→取得で維持されること"""
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="reasoning_test")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="o1",
        prompt_hash="reason1",
        request_body={"messages": [{"role": "user", "content": "Think step by step"}]},
        response_body={"choices": [{"message": {"content": "Let me think..."}}]},
        input_tokens=50, output_tokens=100, reasoning_tokens=500,
        cost_usd=0.05,
    )
    storage.save_request(record)
    found = storage.find_by_id(record.id)
    assert found is not None
    assert found.reasoning_tokens == 500


def test_engine_backward_compat_3tuple(tmp_path):
    """3-tuple を返す旧形式の real_api_fn が動くこと"""
    from llm_devproxy.core.engine import Engine
    from llm_devproxy.core.models import ProxyConfig
    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("compat_test")

    def old_api():
        return ({"choices": [{"message": {"content": "ok"}}]}, 100, 50)

    _, record = engine.call("openai", "gpt-4o", {"messages": []}, old_api)
    assert record.reasoning_tokens == 0
    assert record.output_tokens == 50


def test_engine_4tuple_reasoning(tmp_path):
    """4-tuple を返す新形式で reasoning_tokens が記録されること"""
    from llm_devproxy.core.engine import Engine
    from llm_devproxy.core.models import ProxyConfig
    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("reasoning_test")

    def new_api():
        return ({"choices": [{"message": {"content": "thought..."}}]}, 100, 50, 800)

    _, record = engine.call("openai", "o1", {"messages": [{"role":"user","content":"x"}]}, new_api)
    assert record.reasoning_tokens == 800
    assert record.cost_usd > 0


# ── Alert tests (v0.3.0) ──────────────────────────────────


def test_alert_storage_save_and_get(tmp_path):
    from llm_devproxy.core.models import AlertRecord
    storage = Storage(str(tmp_path / "test.db"))
    alert = AlertRecord(
        level="warning",
        category="cost_daily",
        message="Test alert",
        details={"cost": 0.5},
    )
    storage.save_alert(alert)
    alerts = storage.get_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0].message == "Test alert"
    assert alerts[0].acknowledged is False


def test_alert_acknowledge(tmp_path):
    from llm_devproxy.core.models import AlertRecord
    storage = Storage(str(tmp_path / "test.db"))
    alert = AlertRecord(level="warning", category="test", message="ack me")
    storage.save_alert(alert)
    assert storage.get_unacknowledged_count() == 1
    storage.acknowledge_alert(alert.id)
    assert storage.get_unacknowledged_count() == 0
    alerts = storage.get_alerts()
    assert alerts[0].acknowledged is True


def test_alert_acknowledge_all(tmp_path):
    from llm_devproxy.core.models import AlertRecord
    storage = Storage(str(tmp_path / "test.db"))
    for i in range(3):
        storage.save_alert(AlertRecord(level="warning", category="test", message=f"alert {i}"))
    assert storage.get_unacknowledged_count() == 3
    storage.acknowledge_all_alerts()
    assert storage.get_unacknowledged_count() == 0


def test_alert_manager_daily_cost(tmp_path):
    """デイリーコスト80%超でwarningが発火すること"""
    from llm_devproxy.core.alerts import AlertManager
    from llm_devproxy.core.models import ProxyConfig, AlertRecord
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=1.0,
        alert_daily_threshold=0.8,
    )
    storage = Storage(config.db_path)
    am = AlertManager(config, storage)

    # コスト0.9を記録（90% > 80%閾値）
    session = Session(name="alert_test")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="h1", request_body={}, response_body={},
        cost_usd=0.9, input_tokens=100, output_tokens=50,
    )
    storage.save_request(record)

    alerts = am.evaluate(record, session.id)
    daily_alerts = [a for a in alerts if a.category == "cost_daily"]
    assert len(daily_alerts) == 1
    assert daily_alerts[0].level == "warning"


def test_alert_manager_single_cost(tmp_path):
    """高額単一リクエストでwarningが発火すること"""
    from llm_devproxy.core.alerts import AlertManager
    from llm_devproxy.core.models import ProxyConfig
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=10.0,
        alert_single_cost_usd=0.05,
    )
    storage = Storage(config.db_path)
    am = AlertManager(config, storage)

    session = Session(name="expensive_test")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="o1",
        prompt_hash="h2", request_body={}, response_body={},
        cost_usd=0.15, input_tokens=1000, output_tokens=500,
    )
    storage.save_request(record)

    alerts = am.evaluate(record, session.id)
    single_alerts = [a for a in alerts if a.category == "cost_single"]
    assert len(single_alerts) == 1


def test_alert_manager_reasoning_ratio(tmp_path):
    """推論トークン比率が閾値超でinfoアラートが発火すること"""
    from llm_devproxy.core.alerts import AlertManager
    from llm_devproxy.core.models import ProxyConfig
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=10.0,
        alert_reasoning_ratio=0.7,
        alert_single_cost_usd=999,  # disable single cost alert
    )
    storage = Storage(config.db_path)
    am = AlertManager(config, storage)

    session = Session(name="reasoning_test")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="o1",
        prompt_hash="h3", request_body={}, response_body={},
        cost_usd=0.01, input_tokens=100, output_tokens=50,
        reasoning_tokens=500,  # 500/(500+50) = 91% > 70%
    )
    storage.save_request(record)

    alerts = am.evaluate(record, session.id)
    reason_alerts = [a for a in alerts if a.category == "reasoning_ratio"]
    assert len(reason_alerts) == 1
    assert reason_alerts[0].level == "info"


def test_alert_manager_no_false_positive(tmp_path):
    """閾値以下ではアラートが発火しないこと"""
    from llm_devproxy.core.alerts import AlertManager
    from llm_devproxy.core.models import ProxyConfig
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=10.0,
        alert_daily_threshold=0.8,
        alert_single_cost_usd=1.0,
        alert_reasoning_ratio=0.9,
    )
    storage = Storage(config.db_path)
    am = AlertManager(config, storage)

    session = Session(name="quiet_test")
    storage.save_session(session)
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="h4", request_body={}, response_body={},
        cost_usd=0.001, input_tokens=10, output_tokens=10,
    )
    storage.save_request(record)

    alerts = am.evaluate(record, session.id)
    assert len(alerts) == 0


# ── Export tests (v0.3.0) ──────────────────────────────────


def test_export_csv():
    from llm_devproxy.core.export import export_requests
    records = [
        RequestRecord(
            session_id="s1", step_id=1,
            provider="openai", model="gpt-4o",
            prompt_hash="h1",
            request_body={"messages": [{"role": "user", "content": "Hello"}]},
            response_body={"choices": [{"message": {"content": "Hi"}}]},
            input_tokens=10, output_tokens=5, reasoning_tokens=0,
            cost_usd=0.0001,
        ),
        RequestRecord(
            session_id="s1", step_id=2,
            provider="openai", model="o1",
            prompt_hash="h2",
            request_body={"messages": [{"role": "user", "content": "Think hard"}]},
            response_body={"choices": [{"message": {"content": "..."}}]},
            input_tokens=100, output_tokens=50, reasoning_tokens=500,
            cost_usd=0.05,
        ),
    ]
    csv_str = export_requests(records, format="csv")
    assert "gpt-4o" in csv_str
    assert "reasoning_tokens" in csv_str
    assert "500" in csv_str  # reasoning_tokens value
    lines = csv_str.strip().split("\n")
    assert len(lines) == 3  # header + 2 records


def test_export_json():
    from llm_devproxy.core.export import export_requests
    records = [
        RequestRecord(
            session_id="s1", step_id=1,
            provider="anthropic", model="claude-sonnet-4-5",
            prompt_hash="h1",
            request_body={"messages": [{"role": "user", "content": "日本語テスト"}]},
            response_body={"content": [{"type": "text", "text": "はい"}]},
            input_tokens=20, output_tokens=10, reasoning_tokens=100,
            cost_usd=0.001,
        ),
    ]
    json_str = export_requests(records, format="json")
    data = json.loads(json_str)
    assert len(data) == 1
    assert data[0]["reasoning_tokens"] == 100
    assert data[0]["provider"] == "anthropic"
    assert "日本語テスト" in data[0]["prompt_preview"]


def test_export_empty():
    from llm_devproxy.core.export import export_requests
    assert export_requests([], format="csv") == ""
    assert export_requests([], format="json") == "[]"


def test_export_include_body():
    from llm_devproxy.core.export import export_requests
    records = [
        RequestRecord(
            session_id="s1", step_id=1,
            provider="openai", model="gpt-4o",
            prompt_hash="h1",
            request_body={"messages": [{"role": "user", "content": "test"}]},
            response_body={"choices": [{"message": {"content": "ok"}}]},
        ),
    ]
    json_str = export_requests(records, format="json", include_body=True)
    data = json.loads(json_str)
    assert "request_body" in data[0]
    assert "response_body" in data[0]

    json_str_no_body = export_requests(records, format="json", include_body=False)
    data2 = json.loads(json_str_no_body)
    assert "request_body" not in data2[0]
