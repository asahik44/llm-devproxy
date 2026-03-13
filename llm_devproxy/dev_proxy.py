"""
llm-devproxy: LLM development debug layer.

Every API call is automatically recorded.
Nothing is lost. Everything can be rewound.
"""

from typing import Any, Optional

from .core.engine import Engine
from .core.models import ProxyConfig
from .core.storage import Storage
from .providers.anthropic_wrapper import AnthropicWrapper
from .providers.openai_wrapper import OpenAIWrapper
from .providers.gemini_wrapper import GeminiWrapper


class DevProxy:
    """
    Main entry point for llm-devproxy.

    Usage (Library mode):
        from llm_devproxy import DevProxy

        proxy = DevProxy(daily_limit_usd=1.0)
        session = proxy.start_session("my_agent")

        # Wrap your existing client - that's all
        client = proxy.wrap_anthropic(anthropic.Anthropic())

        # Use exactly as before
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Check what happened
        proxy.stats()
    """

    def __init__(
        self,
        db_path: str = ".llm_devproxy.db",
        daily_limit_usd: float = 1.0,
        session_limit_usd: Optional[float] = None,
        on_exceed: str = "mock",
        cache_enabled: bool = True,
        compress_after_days: int = 30,
    ):
        config = ProxyConfig(
            db_path=db_path,
            daily_limit_usd=daily_limit_usd,
            session_limit_usd=session_limit_usd,
            on_exceed=on_exceed,
            cache_enabled=cache_enabled,
            compress_after_days=compress_after_days,
        )
        self.engine = Engine(config)
        self.storage = self.engine.storage

    # ── Session ───────────────────────────────────────────────

    def start_session(self, name: str = "", description: str = ""):
        """新しいセッションを開始"""
        return self.engine.start_session(name, description)

    def resume_session(self, name_or_id: str):
        """既存セッションを再開"""
        return self.engine.resume_session(name_or_id)

    # ── Wrap clients ──────────────────────────────────────────

    def wrap_anthropic(self, client: Any, session_id: Optional[str] = None) -> AnthropicWrapper:
        """anthropic.Anthropic() クライアントをラップ"""
        sid = session_id or (
            self.engine._current_session.id
            if self.engine._current_session else None
        )
        return AnthropicWrapper(client, self.engine, sid)

    def wrap_openai(self, client: Any, session_id: Optional[str] = None) -> OpenAIWrapper:
        """openai.OpenAI() クライアントをラップ"""
        sid = session_id or (
            self.engine._current_session.id
            if self.engine._current_session else None
        )
        return OpenAIWrapper(client, self.engine, sid)

    def wrap_gemini(self, model: Any, session_id: Optional[str] = None) -> GeminiWrapper:
        """google.generativeai.GenerativeModel をラップ"""
        sid = session_id or (
            self.engine._current_session.id
            if self.engine._current_session else None
        )
        return GeminiWrapper(model, self.engine, sid)

    # ── Rewind ────────────────────────────────────────────────

    def rewind(self, session_name_or_id: str, step: int, branch_name: Optional[str] = None):
        """
        指定セッションの指定ステップに巻き戻す。
        元の履歴は保持される。

        Example:
            proxy.rewind("my_agent", step=3)
            # → step 3の時点のプロンプトが表示される
            # → 次のAPIコールはstep 3から再開（新しいbranchとして記録）
        """
        return self.engine.rewind(session_name_or_id, step, branch_name)

    # ── History & Search ──────────────────────────────────────

    def history(self, limit: int = 20) -> list:
        """セッション一覧を表示"""
        sessions = self.storage.list_sessions(limit)
        for s in sessions:
            print(
                f"  {s.last_accessed.strftime('%Y-%m-%d %H:%M')}  "
                f"{s.name:<30}  "
                f"{s.step_count:>3} steps  "
                f"${s.total_cost_usd:.4f}"
            )
        return sessions

    def search(self, keyword: str, limit: int = 20) -> list:
        """プロンプト内容でキーワード検索"""
        records = self.storage.search_requests(keyword, limit)
        for r in records:
            session = self.storage.get_session(r.session_id)
            session_name = session.name if session else r.session_id[:8]
            content = r.request_body.get("messages", [{}])[-1].get("content", "")
            if isinstance(content, list):
                content = str(content[0])
            preview = content[:60] + "..." if len(content) > 60 else content
            print(
                f"  {r.timestamp.strftime('%Y-%m-%d %H:%M')}  "
                f"session={session_name}  step={r.step_id}  "
                f"\"{preview}\""
            )
        return records

    def show(self, session_name_or_id: str, branch: str = "main") -> list:
        """セッションの全ステップを表示"""
        session = (
            self.storage.get_session_by_name(session_name_or_id)
            or self.storage.get_session(session_name_or_id)
        )
        if not session:
            print(f"Session not found: {session_name_or_id}")
            return []

        records = self.storage.get_requests_by_session(session.id, branch)
        print(f"\n📋 Session: {session.name}")
        print(f"   Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Total cost: ${session.total_cost_usd:.4f}")
        print(f"   Steps: {session.step_count}")
        print()
        for r in records:
            content = r.request_body.get("messages", [{}])[-1].get("content", "")
            if isinstance(content, list):
                content = str(content[0])
            preview = content[:60] + "..." if len(content) > 60 else content
            cached_mark = "💾" if r.is_cached else "  "
            print(
                f"  {cached_mark} step {r.step_id:>3}  "
                f"{r.model:<25}  "
                f"${r.cost_usd:.4f}  "
                f"\"{preview}\""
            )
        return records

    def tag(self, request_id: str, tag: str):
        """リクエストにタグを追加"""
        self.storage.add_tag(request_id, tag)

    def memo(self, request_id: str, memo: str):
        """リクエストにメモを追加"""
        self.storage.add_memo(request_id, memo)

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """コスト状況のサマリーを表示"""
        s = self.engine.stats()
        print(f"\n💰 Cost Stats")
        print(f"   Daily:   ${s['daily_cost_usd']:.6f} / ${s['daily_limit_usd']} "
              f"({s['daily_usage_pct']:.1f}%)")
        print(f"   Session: ${s['session_cost_usd']:.6f}  ({s['current_session']})")
        print(f"   Steps:   {s['current_step']}")
        return s

    # ── Maintenance ───────────────────────────────────────────

    def compress_old(self):
        """古いレスポンス本文を圧縮（メタデータは保持）"""
        self.storage.compress_old_records(self.engine.config.compress_after_days)
        print(f"✅ Compressed records older than {self.engine.config.compress_after_days} days")
