"""
Web UI Application - FastAPI + Jinja2.
Launch with: llm-devproxy web  or  python -m llm_devproxy.web.app
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from llm_devproxy.core.models import ProxyConfig
from llm_devproxy.core.storage import Storage

# ── App setup ────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="llm-devproxy Dashboard", docs_url="/api/docs")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Storage は起動時に設定（run() から注入）
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage(ProxyConfig().db_path)
    return _storage


# ── Template filters ─────────────────────────────────────────

def format_cost(value: float) -> str:
    if value == 0:
        return "—"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def format_tokens(value: int) -> str:
    if value >= 1000:
        return f"{value:,}"
    return str(value)


def format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate(text: str, length: int = 80) -> str:
    if len(text) <= length:
        return text
    return text[:length] + "..."


def extract_prompt_preview(request_body: dict, max_len: int = 120) -> str:
    """リクエストボディから最後のユーザーメッセージを抽出。"""
    # OpenAI / Anthropic
    messages = request_body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = " ".join(texts)
            if content:
                return truncate(str(content), max_len)

    # Gemini
    contents = request_body.get("contents", [])
    for entry in reversed(contents):
        if entry.get("role", "user") == "user":
            parts = entry.get("parts", [])
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            text = " ".join(texts)
            if text:
                return truncate(text, max_len)

    return "(no prompt)"


def extract_response_preview(response_body: dict, max_len: int = 120) -> str:
    """レスポンスボディからアシスタントの返答を抽出。"""
    # compressed
    if response_body.get("compressed"):
        return "(compressed)"

    # OpenAI format
    choices = response_body.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content:
            return truncate(str(content), max_len)

    # Anthropic format
    content_blocks = response_body.get("content", [])
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return truncate(block.get("text", ""), max_len)

    return "(no response)"


# Register filters
templates.env.filters["format_cost"] = format_cost
templates.env.filters["format_tokens"] = format_tokens
templates.env.filters["format_time"] = format_time
templates.env.filters["truncate"] = truncate


def has_reasoning(record) -> bool:
    """推論トークンを使っているか判定"""
    return getattr(record, "reasoning_tokens", 0) > 0

templates.env.tests["reasoning"] = has_reasoning


# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """ダッシュボード（まずは履歴にリダイレクト）。"""
    storage = get_storage()
    sessions = storage.list_sessions(limit=5)
    daily_cost = storage.get_daily_cost()

    return templates.TemplateResponse("history.html", {
        "request": request,
        "page": "history",
        "sessions": sessions,
        "daily_cost": daily_cost,
        "unack_count": storage.get_unacknowledged_count(),
    })


@app.get("/history", response_class=HTMLResponse)
async def history(
    request: Request,
    q: str = Query("", description="検索キーワード"),
    session_id: str = Query("", description="セッションでフィルタ"),
    provider: str = Query("", description="プロバイダーでフィルタ"),
    model: str = Query("", description="モデルでフィルタ"),
    sort_by: str = Query("timestamp", description="ソートカラム"),
    sort_order: str = Query("desc", description="ソート順"),
    page: int = Query(1, ge=1, description="ページ番号"),
    per_page: int = Query(20, ge=5, le=100, description="1ページの件数"),
):
    """リクエスト履歴一覧・検索。"""
    storage = get_storage()

    # モデルがプロバイダーに属さない場合はリセット
    if provider and model:
        valid_models = storage.get_distinct_models(provider)
        if model not in valid_models:
            model = ""

    # SQL側でフィルタ・ソート・ページネーション
    offset = (page - 1) * per_page
    records, total = storage.list_requests(
        q=q, provider=provider, model=model,
        session_id=session_id, sort_by=sort_by, sort_order=sort_order,
        limit=per_page, offset=offset,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # プレビュー付きレコード
    enriched = []
    for r in records:
        enriched.append({
            "record": r,
            "prompt_preview": extract_prompt_preview(r.request_body),
            "response_preview": extract_response_preview(r.response_body),
        })

    # フィルタ用の選択肢（DBに実在するもののみ）
    sessions = storage.list_sessions(limit=50)
    daily_cost = storage.get_daily_cost()
    providers = storage.get_distinct_providers()
    models = storage.get_distinct_models(provider)

    return templates.TemplateResponse("history.html", {
        "request": request,
        "page_name": "history",
        "records": enriched,
        "sessions": sessions,
        "daily_cost": daily_cost,
        "unack_count": storage.get_unacknowledged_count(),
        "q": q,
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "providers": providers,
        "models": models,
        "current_page": page,
        "total_pages": total_pages,
        "total": total,
        "per_page": per_page,
    })


@app.get("/history/{record_id}", response_class=HTMLResponse)
async def history_detail(request: Request, record_id: str):
    """リクエスト詳細。"""
    storage = get_storage()
    record = storage.find_by_id(record_id)
    if not record:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)

    return templates.TemplateResponse("detail.html", {
        "request": request,
        "page_name": "detail",
        "record": record,
        "daily_cost": storage.get_daily_cost(),
        "unack_count": storage.get_unacknowledged_count(),
        "prompt_preview": extract_prompt_preview(record.request_body),
        "response_preview": extract_response_preview(record.response_body),
        "request_json": json.dumps(record.request_body, indent=2, ensure_ascii=False),
        "response_json": json.dumps(record.response_body, indent=2, ensure_ascii=False),
    })


# ── Cost page ─────────────────────────────────────────────

@app.get("/costs", response_class=HTMLResponse)
async def costs(
    request: Request,
    date_from: str = Query("", description="開始日 (YYYY-MM-DD)"),
    date_to: str = Query("", description="終了日 (YYYY-MM-DD)"),
    provider: str = Query("", description="プロバイダーでフィルタ"),
    model: str = Query("", description="モデルでフィルタ"),
    days: int = Query(30, ge=1, le=365, description="プリセット日数"),
):
    """コスト推移ダッシュボード。"""
    from datetime import datetime as dt, timedelta

    storage = get_storage()

    # 日付範囲の決定: カスタム指定があればそちら優先、なければプリセット
    if not date_from:
        date_from = (dt.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    if not date_to:
        date_to = dt.utcnow().strftime("%Y-%m-%d")

    # モデルがプロバイダーに属さない場合はリセット
    if provider and model:
        valid_models = storage.get_distinct_models(provider)
        if model not in valid_models:
            model = ""

    filter_args = dict(date_from=date_from, date_to=date_to, provider=provider, model=model)

    daily_costs = storage.get_daily_costs(**filter_args)
    by_provider = storage.get_cost_by_provider(**filter_args)
    by_model = storage.get_cost_by_model(**filter_args)
    session_costs = storage.get_session_costs(limit=20)
    daily_cost = storage.get_daily_cost()
    providers = storage.get_distinct_providers()
    models = storage.get_distinct_models(provider)

    # Chart.js用データ
    chart_labels = [d["date"] for d in daily_costs]
    chart_costs = [round(d["total_cost"], 6) for d in daily_costs]
    chart_requests = [d["request_count"] for d in daily_costs]
    chart_cached = [d["cached_count"] for d in daily_costs]

    total_cost = sum(d["total_cost"] for d in daily_costs)
    total_requests = sum(d["request_count"] for d in daily_costs)
    total_cached = sum(d["cached_count"] for d in daily_costs)
    cache_rate = (total_cached / total_requests * 100) if total_requests > 0 else 0

    return templates.TemplateResponse("costs.html", {
        "request": request,
        "page_name": "costs",
        "daily_cost": daily_cost,
        "unack_count": storage.get_unacknowledged_count(),
        "date_from": date_from,
        "date_to": date_to,
        "days": days,
        "provider": provider,
        "model": model,
        "providers": providers,
        "models": models,
        # プリセット用日付
        "today": dt.utcnow().strftime("%Y-%m-%d"),
        "yesterday": (dt.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "chart_labels": json.dumps(chart_labels),
        "chart_costs": json.dumps(chart_costs),
        "chart_requests": json.dumps(chart_requests),
        "chart_cached": json.dumps(chart_cached),
        "by_provider": by_provider,
        "by_model": by_model,
        "session_costs": session_costs,
        "total_cost": total_cost,
        "total_requests": total_requests,
        "total_cached": total_cached,
        "cache_rate": cache_rate,
    })


# ── Session comparison page ───────────────────────────────

@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page(
    request: Request,
    a: str = Query("", description="セッションA"),
    b: str = Query("", description="セッションB"),
):
    """セッション比較。"""
    storage = get_storage()
    available = storage.get_sessions_for_comparison()
    daily_cost = storage.get_daily_cost()

    session_a = None
    session_b = None
    records_a = []
    records_b = []

    if a:
        session_a = storage.get_session(a)
        if session_a:
            records_a = storage.get_requests_by_session(a)
    if b:
        session_b = storage.get_session(b)
        if session_b:
            records_b = storage.get_requests_by_session(b)

    # 比較用サマリー
    def summarize(records):
        if not records:
            return {}
        total_cost = sum(r.cost_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_reasoning = sum(r.reasoning_tokens for r in records)
        cached = sum(1 for r in records if r.is_cached)
        models = list(set(r.model for r in records))
        total_all_output = total_output + total_reasoning
        reasoning_pct = (
            total_reasoning / total_all_output * 100 if total_all_output else 0
        )
        return {
            "total_cost": total_cost,
            "total_input": total_input,
            "total_output": total_output,
            "total_reasoning": total_reasoning,
            "reasoning_pct": round(reasoning_pct, 1),
            "total_tokens": total_input + total_output + total_reasoning,
            "steps": len(records),
            "cached": cached,
            "models": models,
            "avg_cost": total_cost / len(records) if records else 0,
        }

    # enriched records with previews
    def enrich(records):
        return [{
            "record": r,
            "prompt_preview": extract_prompt_preview(r.request_body, max_len=80),
            "response_preview": extract_response_preview(r.response_body, max_len=80),
        } for r in records]

    return templates.TemplateResponse("sessions.html", {
        "request": request,
        "page_name": "sessions",
        "daily_cost": daily_cost,
        "unack_count": storage.get_unacknowledged_count(),
        "available": available,
        "a": a,
        "b": b,
        "session_a": session_a,
        "session_b": session_b,
        "records_a": enrich(records_a),
        "records_b": enrich(records_b),
        "summary_a": summarize(records_a),
        "summary_b": summarize(records_b),
    })


# ── API endpoints (for future AJAX) ─────────────────────────

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """アラート一覧。"""
    storage = get_storage()
    alerts = storage.get_alerts(limit=100)
    unack_count = storage.get_unacknowledged_count()
    daily_cost = storage.get_daily_cost()

    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "page_name": "alerts",
        "alerts": alerts,
        "unack_count": unack_count,
        "daily_cost": daily_cost,
    })


@app.post("/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str):
    storage = get_storage()
    storage.acknowledge_alert(alert_id)
    return {"ok": True}


@app.post("/alerts/acknowledge-all")
async def acknowledge_all():
    storage = get_storage()
    storage.acknowledge_all_alerts()
    return {"ok": True}


@app.get("/api/stats")
async def api_stats():
    storage = get_storage()
    daily_cost = storage.get_daily_cost()
    sessions = storage.list_sessions(limit=5)
    unack_count = storage.get_unacknowledged_count()
    return {
        "daily_cost": daily_cost,
        "unacknowledged_alerts": unack_count,
        "recent_sessions": [
            {"id": s.id, "name": s.name, "cost": s.total_cost_usd, "steps": s.step_count}
            for s in sessions
        ],
    }


@app.get("/api/export")
async def api_export(
    format: str = Query("csv", description="csv or json"),
    session_id: str = Query("", description="Filter by session"),
    provider: str = Query("", description="Filter by provider"),
    model: str = Query("", description="Filter by model"),
    include_body: bool = Query(False, description="Include full request/response body"),
    limit: int = Query(0, ge=0, description="Max records (0=all)"),
):
    """Export recorded requests as CSV or JSON download."""
    from fastapi.responses import Response
    from llm_devproxy.core.export import export_requests

    storage = get_storage()
    records, _ = storage.list_requests(
        provider=provider,
        model=model,
        session_id=session_id,
        limit=limit if limit > 0 else 10000,
        offset=0,
    )

    result = export_requests(records, format=format, include_body=include_body)

    if format == "json":
        return Response(
            content=result,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=llm_devproxy_export.json"},
        )
    else:
        return Response(
            content=result,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=llm_devproxy_export.csv"},
        )


# ── Runner ───────────────────────────────────────────────────

def run(db_path: str = ".llm_devproxy.db", host: str = "127.0.0.1", port: int = 8765):
    """Web UIを起動する。"""
    import uvicorn

    global _storage
    _storage = Storage(db_path)

    print(f"🚀 llm-devproxy Dashboard: http://{host}:{port}")
    print(f"   DB: {db_path}")
    print(f"   Press Ctrl+C to stop.\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()