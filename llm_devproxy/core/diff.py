"""
Prompt Diff — compare two RequestRecords side by side.
Uses difflib for text comparison.
"""

import difflib
from typing import Optional

from .models import RequestRecord


def extract_messages(record: RequestRecord) -> list[dict]:
    """RequestRecordからメッセージ配列を抽出。"""
    body = record.request_body

    # OpenAI / Anthropic format
    messages = body.get("messages", [])
    if messages:
        return messages

    # Gemini format (contents)
    contents = body.get("contents", [])
    if contents:
        result = []
        for entry in contents:
            role = entry.get("role", "user")
            parts = entry.get("parts", [])
            text = parts[0].get("text", str(parts[0])) if parts else ""
            result.append({"role": role, "content": text})
        return result

    return []


def messages_to_text(messages: list[dict]) -> str:
    """メッセージ配列を比較用テキストに変換。"""
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Anthropic content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text", str(block)))
                else:
                    texts.append(str(block))
            content = "\n".join(texts)
        lines.append(f"[{role}]")
        lines.append(str(content))
        lines.append("")  # blank line between messages
    return "\n".join(lines)


def compute_diff(record_a: RequestRecord, record_b: RequestRecord) -> dict:
    """
    2つのRequestRecordを比較し、diff情報を返す。

    Returns:
        {
            "lines": [
                {"type": "equal"|"insert"|"delete"|"replace", "a": str, "b": str},
                ...
            ],
            "meta_a": {...},
            "meta_b": {...},
            "similarity": float,  # 0.0〜1.0
        }
    """
    msg_a = extract_messages(record_a)
    msg_b = extract_messages(record_b)

    text_a = messages_to_text(msg_a)
    text_b = messages_to_text(msg_b)

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    # difflib for line-level comparison
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    similarity = matcher.ratio()

    diff_lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in lines_a[i1:i2]:
                diff_lines.append({"type": "equal", "a": line, "b": line})
        elif tag == "delete":
            for line in lines_a[i1:i2]:
                diff_lines.append({"type": "delete", "a": line, "b": ""})
        elif tag == "insert":
            for line in lines_b[j1:j2]:
                diff_lines.append({"type": "insert", "a": "", "b": line})
        elif tag == "replace":
            max_len = max(i2 - i1, j2 - j1)
            a_lines = lines_a[i1:i2] + [""] * (max_len - (i2 - i1))
            b_lines = lines_b[j1:j2] + [""] * (max_len - (j2 - j1))
            for a_line, b_line in zip(a_lines, b_lines):
                diff_lines.append({"type": "replace", "a": a_line, "b": b_line})

    # Response previews
    def response_preview(record: RequestRecord) -> str:
        body = record.response_body
        # OpenAI
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")[:200]
        # Anthropic
        content = body.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")[:200]
        return ""

    total_out_a = record_a.output_tokens + record_a.reasoning_tokens
    total_out_b = record_b.output_tokens + record_b.reasoning_tokens
    reasoning_pct_a = (record_a.reasoning_tokens / total_out_a * 100) if total_out_a else 0
    reasoning_pct_b = (record_b.reasoning_tokens / total_out_b * 100) if total_out_b else 0

    return {
        "lines": diff_lines,
        "meta_a": {
            "id": record_a.id,
            "model": record_a.model,
            "provider": record_a.provider,
            "step_id": record_a.step_id,
            "branch": record_a.branch_name,
            "cost": record_a.cost_usd,
            "input_tokens": record_a.input_tokens,
            "output_tokens": record_a.output_tokens,
            "reasoning_tokens": record_a.reasoning_tokens,
            "reasoning_pct": round(reasoning_pct_a, 0),
            "response_preview": response_preview(record_a),
        },
        "meta_b": {
            "id": record_b.id,
            "model": record_b.model,
            "provider": record_b.provider,
            "step_id": record_b.step_id,
            "branch": record_b.branch_name,
            "cost": record_b.cost_usd,
            "input_tokens": record_b.input_tokens,
            "output_tokens": record_b.output_tokens,
            "reasoning_tokens": record_b.reasoning_tokens,
            "reasoning_pct": round(reasoning_pct_b, 0),
            "response_preview": response_preview(record_b),
        },
        "similarity": round(similarity, 3),
    }
