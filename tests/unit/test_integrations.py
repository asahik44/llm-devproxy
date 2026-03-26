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
