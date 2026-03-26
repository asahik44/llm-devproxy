"""
Integration tests for LangChain / LlamaIndex callback handlers.
Tests internal helpers and import behavior (actual frameworks not required).
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch


# ── Helper function tests (no framework dependency) ────────

def test_detect_provider_openai():
    """OpenAIモデル名の検出"""
    # Import the function by patching langchain_core
    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")
    mock_cb.BaseCallbackHandler = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
    mock_out.LLMResult = type("LLMResult", (), {})
    mock_lc.callbacks = mock_cb
    mock_lc.outputs = mock_out

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        from llm_devproxy.integrations.langchain import _detect_provider
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("o3") == "openai"
        assert _detect_provider("o4-mini") == "openai"
        assert _detect_provider("o1-pro") == "openai"


def test_detect_provider_anthropic():
    """Anthropicモデル名の検出"""
    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")
    mock_cb.BaseCallbackHandler = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
    mock_out.LLMResult = type("LLMResult", (), {})

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        from llm_devproxy.integrations.langchain import _detect_provider
        assert _detect_provider("claude-sonnet-4-6") == "anthropic"
        assert _detect_provider("claude-haiku-4-5") == "anthropic"
        assert _detect_provider("claude-opus-4-6") == "anthropic"


def test_detect_provider_gemini():
    """Geminiモデル名の検出"""
    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")
    mock_cb.BaseCallbackHandler = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
    mock_out.LLMResult = type("LLMResult", (), {})

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        from llm_devproxy.integrations.langchain import _detect_provider
        assert _detect_provider("gemini-3.1-pro-preview") == "gemini"
        assert _detect_provider("gemini-2.5-flash") == "gemini"


def test_detect_provider_unknown():
    """不明モデル名"""
    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")
    mock_cb.BaseCallbackHandler = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
    mock_out.LLMResult = type("LLMResult", (), {})

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        from llm_devproxy.integrations.langchain import _detect_provider
        assert _detect_provider("some-custom-model") == "unknown"


def test_langchain_handler_creation(tmp_path):
    """LangChain handler が正しく初期化される（モックあり）"""
    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")

    class MockBaseHandler:
        def __init__(self):
            pass

    mock_cb.BaseCallbackHandler = MockBaseHandler
    mock_out.LLMResult = type("LLMResult", (), {})

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        # Need to reimport to pick up the mock
        if "llm_devproxy.integrations.langchain" in sys.modules:
            del sys.modules["llm_devproxy.integrations.langchain"]

        from llm_devproxy.integrations.langchain import DevProxyCallbackHandler

        db_path = str(tmp_path / "test.db")
        handler = DevProxyCallbackHandler(
            daily_limit_usd=1.0,
            session_name="test-langchain",
            db_path=db_path,
        )
        assert handler.session_id is not None
        assert handler._step_counter == 0
        assert handler.daily_cost == 0.0


def test_llamaindex_handler_creation(tmp_path):
    """LlamaIndex handler が正しく初期化される（モックあり）"""
    mock_li = types.ModuleType("llama_index")
    mock_li_core = types.ModuleType("llama_index.core")
    mock_li_cb = types.ModuleType("llama_index.core.callbacks")
    mock_li_base = types.ModuleType("llama_index.core.callbacks.base_handler")
    mock_li_schema = types.ModuleType("llama_index.core.callbacks.schema")

    class MockBaseHandler:
        def __init__(self, event_starts_to_ignore=None, event_ends_to_ignore=None):
            pass

    class MockCBEventType:
        LLM = "llm"

    class MockEventPayload:
        RESPONSE = "response"
        MESSAGES = "messages"
        PROMPT = "prompt"
        SERIALIZED = "serialized"

    mock_li_base.BaseCallbackHandler = MockBaseHandler
    mock_li_schema.CBEventType = MockCBEventType
    mock_li_schema.EventPayload = MockEventPayload

    with patch.dict(sys.modules, {
        "llama_index": mock_li,
        "llama_index.core": mock_li_core,
        "llama_index.core.callbacks": mock_li_cb,
        "llama_index.core.callbacks.base_handler": mock_li_base,
        "llama_index.core.callbacks.schema": mock_li_schema,
    }):
        if "llm_devproxy.integrations.llamaindex" in sys.modules:
            del sys.modules["llm_devproxy.integrations.llamaindex"]

        from llm_devproxy.integrations.llamaindex import DevProxyCallbackHandler

        db_path = str(tmp_path / "test.db")
        handler = DevProxyCallbackHandler(
            daily_limit_usd=1.0,
            session_name="test-llamaindex",
            db_path=db_path,
        )
        assert handler.session_id is not None
        assert handler._step_counter == 0


def test_langchain_llm_end_records_request(tmp_path):
    """on_chat_model_start → on_llm_end でリクエストが記録されること"""
    import uuid

    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")

    class MockBaseHandler:
        def __init__(self): pass

    class MockLLMResult:
        def __init__(self, generations=None, llm_output=None):
            self.generations = generations or []
            self.llm_output = llm_output or {}

    mock_cb.BaseCallbackHandler = MockBaseHandler
    mock_out.LLMResult = MockLLMResult

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        if "llm_devproxy.integrations.langchain" in sys.modules:
            del sys.modules["llm_devproxy.integrations.langchain"]

        from llm_devproxy.integrations.langchain import DevProxyCallbackHandler

        db_path = str(tmp_path / "test_flow.db")
        handler = DevProxyCallbackHandler(
            daily_limit_usd=10.0, session_name="flow-test",
            db_path=db_path, verbose=False,
        )

        run_id = uuid.uuid4()
        mock_msg = MagicMock()
        mock_msg.type = "human"
        mock_msg.content = "Write a function"

        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            messages=[[mock_msg]],
            run_id=run_id,
            invocation_params={"model_name": "gpt-4o"},
        )

        mock_gen = MagicMock()
        mock_gen.text = "def hello(): pass"
        mock_gen.generation_info = {}

        response = MockLLMResult(
            generations=[[mock_gen]],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )
        handler.on_llm_end(response=response, run_id=run_id)

        assert handler._step_counter == 1
        assert handler.total_cost > 0


def test_langchain_reasoning_tokens_detected(tmp_path):
    """推論トークンが正しく検出されること"""
    import uuid

    mock_lc = types.ModuleType("langchain_core")
    mock_cb = types.ModuleType("langchain_core.callbacks")
    mock_out = types.ModuleType("langchain_core.outputs")

    class MockBaseHandler:
        def __init__(self): pass

    class MockLLMResult:
        def __init__(self, generations=None, llm_output=None):
            self.generations = generations or []
            self.llm_output = llm_output or {}

    mock_cb.BaseCallbackHandler = MockBaseHandler
    mock_out.LLMResult = MockLLMResult

    with patch.dict(sys.modules, {
        "langchain_core": mock_lc,
        "langchain_core.callbacks": mock_cb,
        "langchain_core.outputs": mock_out,
    }):
        if "llm_devproxy.integrations.langchain" in sys.modules:
            del sys.modules["llm_devproxy.integrations.langchain"]

        from llm_devproxy.integrations.langchain import DevProxyCallbackHandler

        db_path = str(tmp_path / "test_reasoning.db")
        handler = DevProxyCallbackHandler(
            daily_limit_usd=10.0, session_name="reasoning-test",
            db_path=db_path, verbose=False,
        )

        run_id = uuid.uuid4()
        mock_msg = MagicMock()
        mock_msg.type = "human"
        mock_msg.content = "Think step by step"

        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "o3"}},
            messages=[[mock_msg]],
            run_id=run_id,
            invocation_params={"model_name": "o3"},
        )

        mock_gen = MagicMock()
        mock_gen.text = "Step 1..."
        mock_gen.generation_info = {
            "completion_tokens_details": {"reasoning_tokens": 500},
        }

        response = MockLLMResult(
            generations=[[mock_gen]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 200, "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 500},
                },
            },
        )
        handler.on_llm_end(response=response, run_id=run_id)

        assert handler.total_cost > 0


def test_llamaindex_llm_event_records(tmp_path):
    """LlamaIndex LLMイベントがリクエストとして記録されること"""
    mock_li = types.ModuleType("llama_index")
    mock_li_core = types.ModuleType("llama_index.core")
    mock_li_cb = types.ModuleType("llama_index.core.callbacks")
    mock_li_base = types.ModuleType("llama_index.core.callbacks.base_handler")
    mock_li_schema = types.ModuleType("llama_index.core.callbacks.schema")

    class MockBaseHandler:
        def __init__(self, event_starts_to_ignore=None, event_ends_to_ignore=None): pass

    class MockCBEventType:
        LLM = "llm"
        EMBEDDING = "embedding"

    class MockEventPayload:
        RESPONSE = "response"
        MESSAGES = "messages"
        PROMPT = "prompt"
        SERIALIZED = "serialized"

    mock_li_base.BaseCallbackHandler = MockBaseHandler
    mock_li_schema.CBEventType = MockCBEventType
    mock_li_schema.EventPayload = MockEventPayload

    with patch.dict(sys.modules, {
        "llama_index": mock_li,
        "llama_index.core": mock_li_core,
        "llama_index.core.callbacks": mock_li_cb,
        "llama_index.core.callbacks.base_handler": mock_li_base,
        "llama_index.core.callbacks.schema": mock_li_schema,
    }):
        if "llm_devproxy.integrations.llamaindex" in sys.modules:
            del sys.modules["llm_devproxy.integrations.llamaindex"]

        from llm_devproxy.integrations.llamaindex import DevProxyCallbackHandler

        db_path = str(tmp_path / "test_li_flow.db")
        handler = DevProxyCallbackHandler(
            daily_limit_usd=10.0, session_name="li-flow",
            db_path=db_path, verbose=False,
        )

        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.content = "Hello"

        handler.on_event_start(
            event_type=MockCBEventType.LLM,
            payload={
                MockEventPayload.MESSAGES: [mock_msg],
                MockEventPayload.SERIALIZED: {"model": "gpt-4o"},
            },
            event_id="evt-1",
        )

        mock_response = MagicMock()
        mock_raw = MagicMock()
        mock_raw.model = "gpt-4o"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 30
        mock_usage.input_tokens = 0
        mock_usage.output_tokens = 0
        mock_usage.completion_tokens_details = None
        mock_raw.usage = mock_usage
        mock_response.raw = mock_raw
        mock_response.__str__ = lambda self: "Response text"

        handler.on_event_end(
            event_type=MockCBEventType.LLM,
            payload={MockEventPayload.RESPONSE: mock_response},
            event_id="evt-1",
        )

        assert handler._step_counter == 1
        assert handler.total_cost > 0
