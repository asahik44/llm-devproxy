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
        "prompt_preview": extract_prompt_preview(record.request_body),
        "response_preview": extract_response_preview(record.response_body),
        "request_json": json.dumps(record.request_body, indent=2, ensure_ascii=False),
        "response_json": json.dumps(record.response_body, indent=2, ensure_ascii=False),
    })


# ── API endpoints (for future AJAX) ─────────────────────────

@app.get("/api/stats")
async def api_stats():
    storage = get_storage()
    daily_cost = storage.get_daily_cost()
    sessions = storage.list_sessions(limit=5)
    return {
        "daily_cost": daily_cost,
        "recent_sessions": [
            {"id": s.id, "name": s.name, "cost": s.total_cost_usd, "steps": s.step_count}
            for s in sessions
        ],
    }


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