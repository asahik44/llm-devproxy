"""
Prompt Diff tests.
"""

import pytest
from llm_devproxy.core.models import RequestRecord
from llm_devproxy.core.diff import compute_diff, extract_messages, messages_to_text


def _make_record(messages, model="gpt-4o", provider="openai",
                 cost=0.01, reasoning_tokens=0, step_id=1, branch="main",
                 response_text="OK"):
    if provider == "anthropic":
        response_body = {"content": [{"type": "text", "text": response_text}]}
    else:
        response_body = {"choices": [{"message": {"content": response_text}}]}
    return RequestRecord(
        session_id="s1", step_id=step_id, branch_name=branch,
        provider=provider, model=model,
        prompt_hash="h",
        request_body={"messages": messages},
        response_body=response_body,
        input_tokens=100, output_tokens=50,
        reasoning_tokens=reasoning_tokens, cost_usd=cost,
    )


def test_identical_prompts():
    """同一プロンプトのdiffは類似度1.0"""
    msgs = [{"role": "user", "content": "Hello world"}]
    a = _make_record(msgs)
    b = _make_record(msgs)
    diff = compute_diff(a, b)
    assert diff["similarity"] == 1.0
    assert all(line["type"] == "equal" for line in diff["lines"])


def test_completely_different():
    """完全に異なるプロンプトは類似度が低い"""
    a = _make_record([{"role": "user", "content": "Write Python code"}])
    b = _make_record([{"role": "user", "content": "Explain quantum physics in detail"}])
    diff = compute_diff(a, b)
    assert diff["similarity"] <= 0.5
    assert any(line["type"] in ("delete", "insert", "replace") for line in diff["lines"])


def test_partial_change():
    """一部変更のdiffが正しく検出される"""
    a = _make_record([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "このPythonコードのバグを見つけて、修正案を3つ提示して。"},
    ])
    b = _make_record([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "このPythonコードのバグをシンプルに修正して。"},
    ])
    diff = compute_diff(a, b)
    # system message is equal, user message differs
    assert 0.3 < diff["similarity"] < 0.95
    equal_count = sum(1 for l in diff["lines"] if l["type"] == "equal")
    change_count = sum(1 for l in diff["lines"] if l["type"] != "equal")
    assert equal_count > 0
    assert change_count > 0


def test_meta_includes_cost_and_reasoning():
    """メタ情報にコストと推論トークンが含まれる"""
    a = _make_record(
        [{"role": "user", "content": "Think hard"}],
        model="o3", cost=0.212, reasoning_tokens=4200,
    )
    b = _make_record(
        [{"role": "user", "content": "Think simply"}],
        model="o4-mini", cost=0.004, reasoning_tokens=350,
        step_id=2, branch="rewind_step1",
    )
    diff = compute_diff(a, b)
    assert diff["meta_a"]["model"] == "o3"
    assert diff["meta_a"]["cost"] == 0.212
    assert diff["meta_a"]["reasoning_tokens"] == 4200
    assert diff["meta_b"]["model"] == "o4-mini"
    assert diff["meta_b"]["branch"] == "rewind_step1"


def test_anthropic_content_blocks():
    """Anthropic形式のcontent blocksが正しく抽出される"""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Analyze this"}]}]
    a = _make_record(msgs, provider="anthropic", model="claude-sonnet-4-6")
    b = _make_record(
        [{"role": "user", "content": [{"type": "text", "text": "Summarize this"}]}],
        provider="anthropic", model="claude-sonnet-4-6",
    )
    diff = compute_diff(a, b)
    assert diff["similarity"] < 1.0
    assert diff["meta_a"]["provider"] == "anthropic"


def test_extract_messages_gemini_format():
    """Gemini形式のcontentsからメッセージ抽出"""
    record = RequestRecord(
        session_id="s1", step_id=1, provider="gemini", model="gemini-3.1-pro",
        prompt_hash="h",
        request_body={"contents": [{"role": "user", "parts": [{"text": "Hello Gemini"}]}]},
        response_body={},
    )
    messages = extract_messages(record)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello Gemini"


def test_response_preview_in_meta():
    """レスポンスプレビューがメタに含まれる"""
    a = _make_record(
        [{"role": "user", "content": "Hi"}],
        response_text="Hello! How can I help you today?",
    )
    b = _make_record(
        [{"role": "user", "content": "Hi"}],
        response_text="Hey there! What's up?",
    )
    diff = compute_diff(a, b)
    assert "Hello!" in diff["meta_a"]["response_preview"]
    assert "Hey there!" in diff["meta_b"]["response_preview"]
