"""
llm-devproxy HTTP Proxy Server (FastAPI).

OpenAI / Anthropic互換のHTTPプロキシとして動作する。
base_urlをこのサーバーに向けるだけで既存コードが動く。

Usage:
    $ llm-devproxy start --port 8080

    # アプリ側の変更はbase_urlだけ
    client = openai.OpenAI(
        api_key="sk-...",
        base_url="http://localhost:8080/openai/v1",
    )
"""

import json
import os
from typing import Any, Optional

try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from ..core.engine import Engine
from ..core.models import ProxyConfig
from ..core.cost_guard import estimate_cost
from ..core.models import RequestRecord
from datetime import datetime, timezone
import hashlib


def create_app(config: Optional[ProxyConfig] = None) -> Any:
    """FastAPIアプリを生成して返す"""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required for proxy server mode.\n"
            "Install with: pip install llm-devproxy[proxy]"
        )

    config = config or ProxyConfig()
    engine = Engine(config)
    app = FastAPI(
        title="llm-devproxy",
        description="LLM development debug proxy - every call recorded",
        version="0.1.0",
    )

    # ── OpenAI互換エンドポイント ─────────────────────────────

    @app.post("/openai/v1/chat/completions")
    async def openai_chat(request: Request):
        body = await request.json()
        api_key = _extract_api_key(request)
        model = body.get("model", "gpt-4o")

        async def real_api():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                return data, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

        response_body, record = await _async_call(engine, "openai", model, body, real_api)
        return JSONResponse(content=response_body)

    # ── Anthropic互換エンドポイント ──────────────────────────

    @app.post("/anthropic/v1/messages")
    async def anthropic_messages(request: Request):
        body = await request.json()
        api_key = _extract_api_key(request)
        model = body.get("model", "claude-sonnet-4-5")

        async def real_api():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                return data, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

        response_body, record = await _async_call(engine, "anthropic", model, body, real_api)
        return JSONResponse(content=response_body)

    # ── Gemini互換エンドポイント ─────────────────────────────

    @app.post("/gemini/v1/models/{model_path:path}:generateContent")
    async def gemini_generate(model_path: str, request: Request):
        body = await request.json()
        api_key = _extract_api_key(request) or request.query_params.get("key", "")
        model = model_path  # e.g. "gemini-1.5-flash"

        # messagesフォーマットに変換（キャッシュキー用）
        normalized_body = {
            "model": model,
            "messages": _gemini_contents_to_messages(body.get("contents", [])),
            **{k: v for k, v in body.items() if k != "contents"},
        }

        async def real_api():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
                    params={"key": api_key},
                    json=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usageMetadata", {})
                return (
                    data,
                    usage.get("promptTokenCount", 0),
                    usage.get("candidatesTokenCount", 0),
                )

        response_body, record = await _async_call(
            engine, "gemini", model, normalized_body, real_api
        )
        return JSONResponse(content=response_body)

    # ── 管理エンドポイント ────────────────────────────────────

    @app.get("/devproxy/stats")
    async def stats():
        """コスト状況を返す"""
        return engine.stats()

    @app.get("/devproxy/sessions")
    async def sessions(limit: int = 20):
        """セッション一覧を返す"""
        sessions_list = engine.storage.list_sessions(limit)
        return [
            {
                "id": s.id,
                "name": s.name,
                "created_at": s.created_at.isoformat(),
                "last_accessed": s.last_accessed.isoformat(),
                "total_cost_usd": s.total_cost_usd,
                "step_count": s.step_count,
                "tags": s.tags,
            }
            for s in sessions_list
        ]

    @app.get("/devproxy/sessions/{session_name}/steps")
    async def session_steps(session_name: str, branch: str = "main"):
        """セッションの全ステップを返す"""
        session = (
            engine.storage.get_session_by_name(session_name)
            or engine.storage.get_session(session_name)
        )
        if not session:
            return JSONResponse({"error": f"Session not found: {session_name}"}, status_code=404)
        records = engine.storage.get_requests_by_session(session.id, branch)
        return [
            {
                "id": r.id,
                "step_id": r.step_id,
                "branch_name": r.branch_name,
                "timestamp": r.timestamp.isoformat(),
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "is_cached": r.is_cached,
                "tags": r.tags,
                "memo": r.memo,
            }
            for r in records
        ]

    @app.get("/devproxy/search")
    async def search(q: str, limit: int = 20):
        """プロンプト内容でキーワード検索"""
        records = engine.storage.search_requests(q, limit)
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "step_id": r.step_id,
                "timestamp": r.timestamp.isoformat(),
                "provider": r.provider,
                "model": r.model,
                "preview": _preview(r.request_body),
            }
            for r in records
        ]

    @app.post("/devproxy/sessions/{session_name}/rewind")
    async def rewind(session_name: str, step: int, branch: Optional[str] = None):
        """セッションを指定ステップに巻き戻す"""
        record = engine.rewind(session_name, step, branch)
        if not record:
            return JSONResponse({"error": "Session or step not found"}, status_code=404)
        return {
            "message": f"Rewound to step {step}",
            "step_id": record.step_id,
            "preview": _preview(record.request_body),
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "llm-devproxy"}

    return app


# ── ヘルパー関数 ─────────────────────────────────────────────

async def _async_call(engine: Engine, provider: str, model: str, body: dict, real_api_fn):
    """非同期APIコールをEngineに橋渡しする"""
    import asyncio

    loop = asyncio.get_event_loop()
    result_holder = {}
    error_holder = {}

    async def async_real_api():
        return await real_api_fn()

    # キャッシュチェック
    prompt_hash = engine.cache.make_hash(provider, model, body)
    cached = engine.cache.get(provider, model, body)

    session = engine.get_or_create_session()
    engine._step_counter += 1
    step_id = engine._step_counter

    if cached:
        record = engine.cache.build_cached_record(cached, session.id, step_id)
        engine.storage.save_request(record)
        return cached.response_body, record

    # コストチェック
    allowed, reason = engine.cost_guard.check(session.id)
    if not allowed:
        response_body = engine._mock_response(provider, reason)
        record = RequestRecord(
            session_id=session.id,
            step_id=step_id,
            branch_name="main",
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            request_body=body,
            response_body=response_body,
            is_cached=False,
            tags=["cost_limited"],
            memo=reason,
        )
        engine.storage.save_request(record)
        return response_body, record

    # 実API呼び出し
    response_body, input_tokens, output_tokens = await real_api_fn()
    cost = estimate_cost(model, input_tokens, output_tokens)

    record = RequestRecord(
        session_id=session.id,
        step_id=step_id,
        branch_name="main",
        timestamp=datetime.now(timezone.utc),
        provider=provider,
        model=model,
        prompt_hash=prompt_hash,
        request_body=body,
        response_body=response_body,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        is_cached=False,
    )
    engine.storage.save_request(record)
    engine.storage.update_session_stats(session.id, cost)

    return response_body, record


def _extract_api_key(request: Request) -> str:
    """AuthorizationヘッダーまたはX-API-KeyヘッダーからAPIキーを取得"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.headers.get("X-API-Key", request.headers.get("x-api-key", ""))


def _gemini_contents_to_messages(contents: list) -> list:
    """Gemini contentsをmessages形式に変換"""
    messages = []
    for item in contents:
        if isinstance(item, dict):
            role = item.get("role", "user")
            parts = item.get("parts", [])
            text = parts[0].get("text", "") if parts and isinstance(parts[0], dict) else str(parts)
            messages.append({"role": role, "content": text})
    return messages


def _preview(request_body: dict, max_len: int = 80) -> str:
    messages = request_body.get("messages", [])
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content = str(content[0])
        return content[:max_len] + ("..." if len(content) > max_len else "")
    return "(no messages)"


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    db_path: str = ".llm_devproxy.db",
    daily_limit_usd: float = 1.0,
    reload: bool = False,
):
    """プロキシサーバーを起動する"""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required.\n"
            "Install with: pip install 'llm-devproxy[proxy]'"
        )
    config = ProxyConfig(db_path=db_path, daily_limit_usd=daily_limit_usd)
    app = create_app(config)
    print(f"🚀 llm-devproxy server starting...")
    print(f"   http://{host}:{port}")
    print(f"   OpenAI  → http://{host}:{port}/openai/v1")
    print(f"   Anthropic → http://{host}:{port}/anthropic/v1")
    print(f"   Gemini  → http://{host}:{port}/gemini/v1")
    print(f"   Stats   → http://{host}:{port}/devproxy/stats")
    print(f"   DB      → {db_path}")
    uvicorn.run(app, host=host, port=port, reload=reload)
