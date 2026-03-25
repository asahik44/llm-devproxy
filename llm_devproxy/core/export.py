"""
Export - convert RequestRecords to CSV or JSON format.
"""

import csv
import io
import json
from typing import Sequence

from .models import RequestRecord


def export_requests(
    records: Sequence[RequestRecord],
    format: str = "csv",
    include_body: bool = False,
) -> str:
    """
    RequestRecordのリストをCSVまたはJSON文字列に変換。

    Args:
        records: エクスポートするレコードのリスト
        format: "csv" or "json"
        include_body: request_body / response_body を含めるか（巨大になりうる）

    Returns:
        フォーマット済み文字列
    """
    if format == "json":
        return _to_json(records, include_body)
    else:
        return _to_csv(records, include_body)


def _record_to_dict(record: RequestRecord, include_body: bool = False) -> dict:
    """RequestRecordをフラットなdictに変換。"""
    d = {
        "id": record.id,
        "timestamp": record.timestamp.isoformat(),
        "session_id": record.session_id,
        "step_id": record.step_id,
        "branch_name": record.branch_name,
        "provider": record.provider,
        "model": record.model,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "total_tokens": record.input_tokens + record.output_tokens + record.reasoning_tokens,
        "cost_usd": record.cost_usd,
        "is_cached": record.is_cached,
        "tags": ",".join(record.tags) if record.tags else "",
        "memo": record.memo,
    }

    # プロンプトプレビュー（最後のuser message）
    messages = record.request_body.get("messages", [])
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict)]
                content = " ".join(texts)
            prompt = str(content)[:500]
            break
    d["prompt_preview"] = prompt

    if include_body:
        d["request_body"] = json.dumps(record.request_body, ensure_ascii=False)
        d["response_body"] = json.dumps(record.response_body, ensure_ascii=False)

    return d


def _to_csv(records: Sequence[RequestRecord], include_body: bool = False) -> str:
    """CSV形式でエクスポート。"""
    if not records:
        return ""

    rows = [_record_to_dict(r, include_body) for r in records]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _to_json(records: Sequence[RequestRecord], include_body: bool = False) -> str:
    """JSON形式でエクスポート。"""
    rows = [_record_to_dict(r, include_body) for r in records]
    return json.dumps(rows, indent=2, ensure_ascii=False)
