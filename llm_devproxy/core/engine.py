"""
Core Engine - the heart of llm-devproxy.
All API calls flow through here: cache check → cost check → real API → record.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .cache import CacheManager
from .cost_guard import CostGuard, estimate_cost
from .alerts import AlertManager
from .models import ProxyConfig, RequestRecord, Session
from .storage import Storage


class Engine:
    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()
        self.storage = Storage(self.config.db_path)
        self.cache = CacheManager(self.storage, self.config, self.config.cache_enabled)
        self.cost_guard = CostGuard(self.config, self.storage)
        self.alert_manager = AlertManager(self.config, self.storage)

        # 現在のセッション
        self._current_session: Optional[Session] = None
        self._step_counter: int = 0

    # ── Session management ────────────────────────────────────

    def start_session(self, name: str = "", description: str = "") -> Session:
        """新しいセッションを開始（エージェントの1実行に対応）"""
        session = Session(
            name=name or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            description=description,
        )
        self.storage.save_session(session)
        self._current_session = session
        self._step_counter = 0
        return session

    def resume_session(self, session_name_or_id: str) -> Optional[Session]:
        """既存セッションを再開（rewind用）"""
        session = self.storage.get_session_by_name(session_name_or_id)
        if not session:
            session = self.storage.get_session(session_name_or_id)
        if session:
            self._current_session = session
            # 最後のstep_idから再開
            requests = self.storage.get_requests_by_session(session.id)
            self._step_counter = max((r.step_id for r in requests), default=0)
        return session

    def get_or_create_session(self, name: str = "") -> Session:
        if self._current_session:
            return self._current_session
        return self.start_session(name)

    # ── Main call method ──────────────────────────────────────

    def call(
        self,
        provider: str,
        model: str,
        request_body: dict,
        real_api_fn: Callable[[], tuple],
        session_id: Optional[str] = None,
        branch_name: str = "main",
    ) -> tuple[dict, RequestRecord]:
        """
        LLM APIを呼ぶ統一インターフェース。

        Args:
            provider: "openai" / "anthropic" / "gemini"
            model: モデル名
            request_body: APIリクエストのボディ
            real_api_fn: 実際のAPIを呼ぶ関数。
                         (response_body, input_tokens, output_tokens) または
                         (response_body, input_tokens, output_tokens, reasoning_tokens)
                         を返す
            session_id: セッションID（省略時は現在のセッション）
            branch_name: rewind後の別試みで使うブランチ名

        Returns:
            (response_body, RequestRecord)
        """
        session = self._resolve_session(session_id)
        self._step_counter += 1
        step_id = self._step_counter

        prompt_hash = self.cache.make_hash(provider, model, request_body)

        # 1. キャッシュチェック
        cached = self.cache.get(provider, model, request_body)
        if cached:
            record = self.cache.build_cached_record(cached, session.id, step_id)
            self.storage.save_request(record)
            return cached.response_body, record

        # 2. コストチェック
        allowed, reason = self.cost_guard.check(session.id)
        if not allowed:
            response_body = self._mock_response(provider, reason)
            record = RequestRecord(
                session_id=session.id,
                step_id=step_id,
                branch_name=branch_name,
                timestamp=datetime.now(timezone.utc),
                provider=provider,
                model=model,
                prompt_hash=prompt_hash,
                request_body=request_body,
                response_body=response_body,
                is_cached=False,
                tags=["cost_limited"],
                memo=reason,
            )
            self.storage.save_request(record)
            if self.config.on_exceed == "block":
                raise CostLimitExceededError(reason)
            return response_body, record

        # 3. 実APIを呼ぶ（3-tuple or 4-tuple対応）
        result = real_api_fn()
        if len(result) == 4:
            response_body, input_tokens, output_tokens, reasoning_tokens = result
        else:
            response_body, input_tokens, output_tokens = result
            reasoning_tokens = 0

        cost = estimate_cost(model, input_tokens, output_tokens, reasoning_tokens)

        # 4. 自動記録（これがコアコンセプト）
        record = RequestRecord(
            session_id=session.id,
            step_id=step_id,
            branch_name=branch_name,
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            request_body=request_body,
            response_body=response_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            is_cached=False,
        )
        self.storage.save_request(record)
        self.storage.update_session_stats(session.id, cost)

        # 4.5. セマンティックキャッシュ用embeddingを保存
        self.cache.store_semantic(record, request_body)

        # 5. アラートチェック（v0.3.0）
        self.alert_manager.evaluate(record, session.id)

        # 5.5. 推論トークン可視化（使っている場合のみ表示）
        if reasoning_tokens > 0:
            total_output = output_tokens + reasoning_tokens
            reasoning_pct = reasoning_tokens / total_output * 100 if total_output else 0
            print(
                f"🧠 Reasoning tokens: {reasoning_tokens:,} "
                f"({reasoning_pct:.0f}% of output) | "
                f"Output: {output_tokens:,} | "
                f"Cost: ${cost:.6f}"
            )

        return response_body, record

    # ── Rewind (Phase 3 core feature) ─────────────────────────

    def rewind(
        self,
        session_name_or_id: str,
        step: int,
        branch_name: Optional[str] = None,
    ) -> Optional[RequestRecord]:
        """
        指定セッションの指定ステップに巻き戻す。
        元の履歴は消えない。新しいbranchとして再実行できる。

        Returns:
            その時点のRequestRecord（プロンプトを取り出せる）
        """
        session = (
            self.storage.get_session_by_name(session_name_or_id)
            or self.storage.get_session(session_name_or_id)
        )
        if not session:
            print(f"Session not found: {session_name_or_id}")
            return None

        record = self.storage.get_request_at_step(session.id, step)
        if not record:
            print(f"Step {step} not found in session {session_name_or_id}")
            return None

        # このセッションを現在のセッションとして再開
        self._current_session = session
        self._step_counter = step - 1  # 次のcallがstep_idを+1してこのstepになる

        # branch名を設定（後から別試みを区別するため）
        new_branch = branch_name or f"rewind_step{step}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        print(f"✅ Rewound to step {step} of session '{session.name}'")
        print(f"   Branch: {new_branch}")
        print(f"   Original prompt preview: {self._preview(record.request_body)}")
        print(f"   Modify your prompt and re-run.")

        return record

    # ── Helpers ───────────────────────────────────────────────

    def _resolve_session(self, session_id: Optional[str]) -> Session:
        if session_id:
            s = self.storage.get_session(session_id)
            if s:
                return s
        if self._current_session:
            return self._current_session
        return self.start_session()

    def _mock_response(self, provider: str, reason: str) -> dict:
        msg = f"{self.config.mock_response} ({reason})"
        if provider == "anthropic":
            return {
                "content": [{"type": "text", "text": msg}],
                "model": "mock",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        # openai / gemini互換
        return {
            "choices": [{"message": {"role": "assistant", "content": msg}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _preview(self, request_body: dict, max_len: int = 80) -> str:
        messages = request_body.get("messages", [])
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, list):
                content = str(content[0])
            return content[:max_len] + ("..." if len(content) > max_len else "")
        return "(no messages)"

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """現在のコスト状況のサマリー"""
        daily = self.storage.get_daily_cost()
        session_cost = (
            self.storage.get_session_cost(self._current_session.id)
            if self._current_session else 0.0
        )
        return {
            "daily_cost_usd": daily,
            "daily_limit_usd": self.config.daily_limit_usd,
            "daily_usage_pct": daily / self.config.daily_limit_usd * 100,
            "session_cost_usd": session_cost,
            "current_session": self._current_session.name if self._current_session else None,
            "current_step": self._step_counter,
        }

    def reasoning_stats(self, session_id: Optional[str] = None) -> dict:
        """推論トークンの使用状況サマリー"""
        sid = session_id or (self._current_session.id if self._current_session else None)
        if not sid:
            return {"total_reasoning": 0, "total_output": 0, "reasoning_pct": 0}

        records = self.storage.get_requests_by_session(sid)
        total_reasoning = sum(r.reasoning_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_all_output = total_output + total_reasoning
        pct = total_reasoning / total_all_output * 100 if total_all_output else 0
        return {
            "total_reasoning": total_reasoning,
            "total_output": total_output,
            "reasoning_pct": round(pct, 1),
            "reasoning_cost_est": sum(
                r.reasoning_tokens / 1000 * 0.015  # rough estimate
                for r in records if r.reasoning_tokens > 0
            ),
        }


class CostLimitExceededError(Exception):
    pass