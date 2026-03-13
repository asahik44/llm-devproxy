"""
Engine・DevProxy・Gemini・Rewind のテスト
"""

import pytest
from llm_devproxy.core.models import ProxyConfig, RequestRecord, Session
from llm_devproxy.core.storage import Storage
from llm_devproxy.core.engine import Engine, CostLimitExceededError
from llm_devproxy.dev_proxy import DevProxy


# ── Engine tests ──────────────────────────────────────────────

def test_engine_call_records_all(tmp_path):
    """APIコールが自動で記録されることを確認"""
    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    session = engine.start_session("test_auto_record")

    call_count = 0

    def fake_api():
        nonlocal call_count
        call_count += 1
        return {"choices": [{"message": {"content": "hello"}}]}, 10, 20

    response, record = engine.call(
        provider="openai",
        model="gpt-4o-mini",
        request_body={"messages": [{"role": "user", "content": "hi"}]},
        real_api_fn=fake_api,
    )

    assert call_count == 1
    assert record.session_id == session.id
    assert record.step_id == 1
    assert record.input_tokens == 10
    assert record.output_tokens == 20
    assert record.is_cached is False

    # DBに保存されているか確認
    saved = engine.storage.get_requests_by_session(session.id)
    assert len(saved) == 1


def test_engine_cache_prevents_duplicate_call(tmp_path):
    """同じリクエストはキャッシュから返されて実APIが呼ばれないことを確認"""
    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=True)
    engine = Engine(config)
    engine.start_session("cache_test")

    call_count = 0

    def fake_api():
        nonlocal call_count
        call_count += 1
        return {"choices": [{"message": {"content": "cached response"}}]}, 10, 5

    request = {"messages": [{"role": "user", "content": "same question"}], "temperature": 0}

    # 1回目: 実APIを叩く
    engine.call("openai", "gpt-4o", request, fake_api)
    assert call_count == 1

    # 2回目: キャッシュから返す
    response, record = engine.call("openai", "gpt-4o", request, fake_api)
    assert call_count == 1  # 実APIは呼ばれない
    assert record.is_cached is True


def test_engine_cost_limit_returns_mock(tmp_path):
    """コスト上限超えでmockレスポンスが返ることを確認"""
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=0.0001,  # 超低いlimit
        on_exceed="mock",
        cache_enabled=False,
    )
    engine = Engine(config)
    session = engine.start_session("limit_test")

    # まず上限を超える記録を作る
    storage = engine.storage
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="xxx", request_body={}, response_body={},
        cost_usd=1.0,
    )
    storage.save_request(record)

    call_count = 0

    def fake_api():
        nonlocal call_count
        call_count += 1
        return {}, 100, 100

    response, rec = engine.call("openai", "gpt-4o", {}, fake_api)
    assert call_count == 0  # 実APIは呼ばれない
    assert "MOCK" in str(response) or "mock" in str(response).lower() or "limit" in rec.memo.lower()


def test_engine_cost_limit_block_raises(tmp_path):
    """on_exceed=blockのとき例外が発生することを確認"""
    config = ProxyConfig(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=0.0001,
        on_exceed="block",
        cache_enabled=False,
    )
    engine = Engine(config)
    session = engine.start_session("block_test")

    storage = engine.storage
    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="yyy", request_body={}, response_body={},
        cost_usd=1.0,
    )
    storage.save_request(record)

    with pytest.raises(CostLimitExceededError):
        engine.call("openai", "gpt-4o", {}, lambda: ({}, 10, 10))


def test_engine_rewind(tmp_path):
    """rewindで指定ステップに戻れることを確認"""
    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("rewind_test")

    def fake_api(content):
        return {"choices": [{"message": {"content": f"response to {content}"}}]}, 10, 10

    # 3ステップ実行
    for i in range(1, 4):
        engine.call(
            "openai", "gpt-4o",
            {"messages": [{"role": "user", "content": f"step {i} question"}]},
            lambda: fake_api(f"step {i}"),
        )

    # step 2に巻き戻す
    record = engine.rewind("rewind_test", step=2)
    assert record is not None
    assert record.step_id == 2
    # step counterが1に設定されている（次のcallがstep 2になる）
    assert engine._step_counter == 1


# ── DevProxy integration tests ────────────────────────────────

def test_devproxy_session_workflow(tmp_path):
    """DevProxyの基本的なセッションワークフロー"""
    proxy = DevProxy(
        db_path=str(tmp_path / "test.db"),
        daily_limit_usd=10.0,
        cache_enabled=False,
    )
    session = proxy.start_session("integration_test", description="テスト用セッション")
    assert session.name == "integration_test"

    sessions = proxy.storage.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].name == "integration_test"


def test_devproxy_search(tmp_path):
    """DevProxyのsearch機能"""
    proxy = DevProxy(db_path=str(tmp_path / "test.db"))
    session = proxy.start_session("search_test")

    storage = proxy.storage
    for i, content in enumerate(["Python入門", "機械学習の基礎", "Pythonデータ分析"]):
        record = RequestRecord(
            session_id=session.id,
            step_id=i + 1,
            provider="openai",
            model="gpt-4o",
            prompt_hash=f"hash_{i}",
            request_body={"messages": [{"role": "user", "content": content}]},
            response_body={},
        )
        storage.save_request(record)

    results = proxy.search("Python")
    assert len(results) == 2  # "Python入門" と "Pythonデータ分析"


def test_devproxy_history(tmp_path, capsys):
    """DevProxyのhistory表示"""
    proxy = DevProxy(db_path=str(tmp_path / "test.db"))
    proxy.start_session("session_a")
    proxy.start_session("session_b")

    sessions = proxy.history()
    assert len(sessions) == 2

    captured = capsys.readouterr()
    assert "session_a" in captured.out
    assert "session_b" in captured.out


def test_devproxy_tag_and_memo(tmp_path):
    """タグとメモの追加"""
    proxy = DevProxy(db_path=str(tmp_path / "test.db"))
    session = proxy.start_session("tag_test")

    record = RequestRecord(
        session_id=session.id,
        step_id=1,
        provider="openai",
        model="gpt-4o",
        prompt_hash="tag_hash",
        request_body={"messages": [{"role": "user", "content": "test"}]},
        response_body={},
    )
    proxy.storage.save_request(record)

    proxy.tag(record.id, "important")
    proxy.memo(record.id, "この回答は参考になった")

    saved = proxy.storage.find_cached("tag_hash", "gpt-4o")
    assert "important" in saved.tags
    assert "この回答は参考になった" == saved.memo


# ── Gemini wrapper tests ──────────────────────────────────────

def test_gemini_wrapper_basic(tmp_path):
    """GeminiWrapperの基本動作（モックモデルで）"""
    from llm_devproxy.providers.gemini_wrapper import GeminiWrapper

    class MockGeminiModel:
        model_name = "gemini-1.5-flash"

        def generate_content(self, contents, **kwargs):
            return MockGeminiResponse("Geminiの回答です")

    class MockGeminiResponse:
        def __init__(self, text):
            self.text = text

        class usage_metadata:
            prompt_token_count = 10
            candidates_token_count = 20
            total_token_count = 30

    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("gemini_test")

    wrapper = GeminiWrapper(MockGeminiModel(), engine)
    response = wrapper.generate_content("日本語で答えてください")

    assert response.text == "Geminiの回答です"


# ── Storage full-history tests ────────────────────────────────

def test_storage_all_history_preserved(tmp_path):
    """全履歴が保存・検索できることを確認（タイムトラベルコンセプト）"""
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="history_test")
    storage.save_session(session)

    # 10ステップ記録
    for i in range(1, 11):
        record = RequestRecord(
            session_id=session.id,
            step_id=i,
            provider="anthropic",
            model="claude-sonnet-4-5",
            prompt_hash=f"hash_{i}",
            request_body={"messages": [{"role": "user", "content": f"質問{i}"}]},
            response_body={"content": [{"type": "text", "text": f"回答{i}"}]},
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        storage.save_request(record)

    # 全ステップが取得できる
    all_records = storage.get_requests_by_session(session.id)
    assert len(all_records) == 10

    # 任意のステップが取得できる
    step5 = storage.get_request_at_step(session.id, 5)
    assert step5 is not None
    assert step5.step_id == 5

    # 日本語検索が効く
    results = storage.search_requests("質問7")
    assert len(results) == 1


def test_storage_branch_support(tmp_path):
    """ブランチ（rewind後の別試み）が区別して記録されることを確認"""
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="branch_test")
    storage.save_session(session)

    # mainブランチに3ステップ
    for i in range(1, 4):
        storage.save_request(RequestRecord(
            session_id=session.id, step_id=i, branch_name="main",
            provider="openai", model="gpt-4o",
            prompt_hash=f"main_{i}", request_body={}, response_body={},
        ))

    # rewind後の新ブランチに2ステップ
    for i in range(2, 4):
        storage.save_request(RequestRecord(
            session_id=session.id, step_id=i, branch_name="new_idea",
            provider="openai", model="gpt-4o",
            prompt_hash=f"new_{i}", request_body={}, response_body={},
        ))

    main_records = storage.get_requests_by_session(session.id, "main")
    new_records = storage.get_requests_by_session(session.id, "new_idea")

    assert len(main_records) == 3
    assert len(new_records) == 2


def test_storage_compress_old_records(tmp_path):
    """古いレコードのレスポンス本文が圧縮されることを確認"""
    from datetime import timedelta, timezone
    storage = Storage(str(tmp_path / "test.db"))
    session = Session(name="compress_test")
    storage.save_session(session)

    record = RequestRecord(
        session_id=session.id, step_id=1,
        provider="openai", model="gpt-4o",
        prompt_hash="compress_hash",
        request_body={"messages": [{"role": "user", "content": "old question"}]},
        response_body={"choices": [{"message": {"content": "detailed long response"}}]},
    )
    storage.save_request(record)

    # 40日前のデータとして圧縮
    storage.compress_old_records(older_than_days=-1)  # -1で全件圧縮

    compressed = storage.find_cached("compress_hash", "gpt-4o")
    assert compressed is not None
    assert compressed.response_body == {"compressed": True}
