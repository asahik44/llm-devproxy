"""
Streaming provider tests (mock-based, no real API calls).
"""

import pytest
from llm_devproxy.core import Storage
from llm_devproxy.core.models import RequestRecord, Session, ProxyConfig
from llm_devproxy.core.engine import Engine


# ── Mock objects for OpenAI streaming ──────────────────────


class MockOpenAIDelta:
    def __init__(self, content=None, role=None):
        self.content = content
        self.role = role


class MockOpenAIStreamChoice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = MockOpenAIDelta(content=content)
        self.finish_reason = finish_reason


class MockOpenAIStreamChunk:
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.choices = [MockOpenAIStreamChoice(content=content, finish_reason=finish_reason)]
        self.usage = usage


class MockOpenAIUsage:
    def __init__(self, prompt=0, completion=0, reasoning=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.completion_tokens_details = MockCompletionDetails(reasoning) if reasoning else None


class MockCompletionDetails:
    def __init__(self, reasoning=0):
        self.reasoning_tokens = reasoning


def make_openai_stream_chunks(text_parts, prompt_tokens=50, completion_tokens=100, reasoning_tokens=0):
    """OpenAI streaming チャンク列を生成（stream_options.include_usage=True 相当）"""
    chunks = []
    for part in text_parts:
        chunks.append(MockOpenAIStreamChunk(content=part))
    # final chunk with usage
    usage = MockOpenAIUsage(prompt_tokens, completion_tokens, reasoning_tokens)
    chunks.append(MockOpenAIStreamChunk(finish_reason="stop", usage=usage))
    return chunks


# ── Mock objects for Anthropic streaming ───────────────────


class MockAnthropicEvent:
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockAnthropicDelta:
    def __init__(self, delta_type, text="", thinking=""):
        self.type = delta_type
        self.text = text
        self.thinking = thinking


class MockAnthropicUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicMessage:
    def __init__(self, input_tokens=0):
        self.usage = MockAnthropicUsage(input_tokens=input_tokens)


def make_anthropic_stream_events(text_parts, thinking_parts=None, input_tokens=50, output_tokens=100):
    """Anthropic streaming イベント列を生成"""
    events = []
    # message_start
    events.append(MockAnthropicEvent(
        "message_start",
        message=MockAnthropicMessage(input_tokens=input_tokens),
    ))
    # thinking blocks
    if thinking_parts:
        for part in thinking_parts:
            events.append(MockAnthropicEvent(
                "content_block_delta",
                delta=MockAnthropicDelta("thinking_delta", thinking=part),
            ))
    # text blocks
    for part in text_parts:
        events.append(MockAnthropicEvent(
            "content_block_delta",
            delta=MockAnthropicDelta("text_delta", text=part),
        ))
    # message_delta with usage
    events.append(MockAnthropicEvent(
        "message_delta",
        usage=MockAnthropicUsage(output_tokens=output_tokens),
    ))
    return events


# ── Tests ──────────────────────────────────────────────────


def test_openai_stream_wrapper_collects_content(tmp_path):
    """OpenAI stream wrapper がチャンクを蓄積して記録すること"""
    from llm_devproxy.providers.openai_wrapper import _OpenAIStreamWrapper

    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("stream_test")

    chunks = make_openai_stream_chunks(
        ["Hello", " world", "!"],
        prompt_tokens=50, completion_tokens=10, reasoning_tokens=0,
    )

    wrapper = _OpenAIStreamWrapper(
        stream=iter(chunks),
        engine=engine,
        model="gpt-4o",
        request_body={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        session_id=None,
    )

    collected = []
    for chunk in wrapper:
        if chunk.choices and chunk.choices[0].delta.content:
            collected.append(chunk.choices[0].delta.content)

    assert "".join(collected) == "Hello world!"

    # DB に記録されていること
    records, total = engine.storage.list_requests()
    assert total >= 1
    last = records[0]
    assert last.output_tokens == 10
    assert "Hello world!" in last.response_body.get("choices", [{}])[0].get("message", {}).get("content", "")


def test_openai_stream_wrapper_with_reasoning(tmp_path):
    """OpenAI stream wrapper が推論トークンも記録すること"""
    from llm_devproxy.providers.openai_wrapper import _OpenAIStreamWrapper

    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("reasoning_stream")

    chunks = make_openai_stream_chunks(
        ["Result"],
        prompt_tokens=100, completion_tokens=50, reasoning_tokens=800,
    )

    wrapper = _OpenAIStreamWrapper(
        stream=iter(chunks),
        engine=engine,
        model="o1",
        request_body={"messages": [{"role": "user", "content": "think"}], "stream": True},
        session_id=None,
    )

    for _ in wrapper:
        pass

    records, _ = engine.storage.list_requests()
    assert records[0].reasoning_tokens == 800
    assert records[0].cost_usd > 0


def test_anthropic_stream_wrapper_collects_content(tmp_path):
    """Anthropic stream wrapper がイベントを蓄積して記録すること"""
    from llm_devproxy.providers.anthropic_wrapper import _AnthropicStreamWrapper

    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("anthropic_stream")

    events = make_anthropic_stream_events(
        ["Hello", " from", " Claude"],
        input_tokens=30, output_tokens=15,
    )

    wrapper = _AnthropicStreamWrapper(
        stream=iter(events),
        engine=engine,
        model="claude-sonnet-4-5",
        request_body={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        session_id=None,
    )

    for _ in wrapper:
        pass

    records, _ = engine.storage.list_requests()
    assert len(records) >= 1
    last = records[0]
    assert last.input_tokens == 30
    # content should include text blocks
    content_blocks = last.response_body.get("content", [])
    texts = [b["text"] for b in content_blocks if b.get("type") == "text"]
    assert "Hello from Claude" in "".join(texts)


def test_anthropic_stream_wrapper_with_thinking(tmp_path):
    """Anthropic stream wrapper がthinkingブロックも記録すること"""
    from llm_devproxy.providers.anthropic_wrapper import _AnthropicStreamWrapper

    config = ProxyConfig(db_path=str(tmp_path / "test.db"), cache_enabled=False)
    engine = Engine(config)
    engine.start_session("thinking_stream")

    events = make_anthropic_stream_events(
        ["Answer is 42"],
        thinking_parts=["Let me think about this carefully..." * 10],
        input_tokens=50, output_tokens=20,
    )

    wrapper = _AnthropicStreamWrapper(
        stream=iter(events),
        engine=engine,
        model="claude-sonnet-4-5",
        request_body={"messages": [{"role": "user", "content": "think"}], "stream": True},
        session_id=None,
    )

    for _ in wrapper:
        pass

    records, _ = engine.storage.list_requests()
    last = records[0]
    assert last.reasoning_tokens > 0
    # thinking block should be in response body
    content_blocks = last.response_body.get("content", [])
    types = [b["type"] for b in content_blocks]
    assert "thinking" in types


def test_openai_cached_stream(tmp_path):
    """OpenAI キャッシュヒット時の疑似ストリームが動くこと"""
    from llm_devproxy.providers.openai_wrapper import _OpenAICachedStream

    body = {
        "choices": [{"message": {"role": "assistant", "content": "cached response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    stream = _OpenAICachedStream(body)
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "cached response"


def test_anthropic_cached_stream(tmp_path):
    """Anthropic キャッシュヒット時の疑似ストリームが動くこと"""
    from llm_devproxy.providers.anthropic_wrapper import _AnthropicCachedStream

    body = {
        "content": [{"type": "text", "text": "cached!"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }

    stream = _AnthropicCachedStream(body)
    events = list(stream)
    assert len(events) == 1
    assert events[0].delta.text == "cached!"
