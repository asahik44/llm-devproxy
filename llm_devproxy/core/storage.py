"""
SQLite storage layer.
Every API call is automatically persisted. Nothing is lost.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .models import RequestRecord, Session, CostLimit, AlertRecord


class Storage:
    def __init__(self, db_path: str = ".llm_devproxy.db"):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    total_cost_usd REAL DEFAULT 0.0,
                    step_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_id INTEGER DEFAULT 0,
                    parent_id TEXT,
                    branch_name TEXT DEFAULT 'main',
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    request_body TEXT NOT NULL,
                    response_body TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    reasoning_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    is_cached INTEGER DEFAULT 0,
                    tags TEXT DEFAULT '[]',
                    memo TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS cost_limits (
                    scope TEXT PRIMARY KEY,
                    limit_usd REAL NOT NULL,
                    current_usd REAL DEFAULT 0.0,
                    action TEXT DEFAULT 'mock',
                    reset_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_requests_session
                    ON requests(session_id);
                CREATE INDEX IF NOT EXISTS idx_requests_prompt_hash
                    ON requests(prompt_hash);
                CREATE INDEX IF NOT EXISTS idx_requests_timestamp
                    ON requests(timestamp);

                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'warning',
                    category TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    details TEXT DEFAULT '{}',
                    acknowledged INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                    ON alerts(timestamp);
            """)
            # v0.3.0 migration: add reasoning_tokens column if missing
            self._migrate(conn)

    def _migrate(self, conn):
        """Apply schema migrations for backward compatibility."""
        # Check if reasoning_tokens column exists
        cursor = conn.execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in cursor.fetchall()}
        if "reasoning_tokens" not in columns:
            conn.execute(
                "ALTER TABLE requests ADD COLUMN reasoning_tokens INTEGER DEFAULT 0"
            )

    # ── Sessions ──────────────────────────────────────────────

    def save_session(self, session: Session) -> Session:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (id, name, created_at, last_accessed, description, tags,
                 total_cost_usd, step_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id, session.name,
                session.created_at.isoformat(),
                session.last_accessed.isoformat(),
                session.description,
                json.dumps(session.tags, ensure_ascii=False),
                session.total_cost_usd,
                session.step_count,
            ))
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._row_to_session(row) if row else None

    def get_session_by_name(self, name: str) -> Optional[Session]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[Session]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY last_accessed DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_session_stats(self, session_id: str, cost: float):
        with self._conn() as conn:
            conn.execute("""
                UPDATE sessions
                SET total_cost_usd = total_cost_usd + ?,
                    step_count = step_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """, (cost, datetime.now(timezone.utc).isoformat(), session_id))

    # ── Requests ──────────────────────────────────────────────

    def save_request(self, record: RequestRecord) -> RequestRecord:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO requests
                (id, session_id, step_id, parent_id, branch_name,
                 timestamp, provider, model, prompt_hash,
                 request_body, response_body,
                 input_tokens, output_tokens, reasoning_tokens, cost_usd,
                 is_cached, tags, memo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.session_id, record.step_id,
                record.parent_id, record.branch_name,
                record.timestamp.isoformat(),
                record.provider, record.model, record.prompt_hash,
                json.dumps(record.request_body, ensure_ascii=False),
                json.dumps(record.response_body, ensure_ascii=False),
                record.input_tokens, record.output_tokens,
                record.reasoning_tokens, record.cost_usd,
                int(record.is_cached),
                json.dumps(record.tags, ensure_ascii=False), record.memo,
            ))
        return record

    def find_cached(self, prompt_hash: str, model: str) -> Optional[RequestRecord]:
        """キャッシュヒットを探す（同じhash + modelの最新を返す）"""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM requests
                WHERE prompt_hash = ? AND model = ? AND is_cached = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (prompt_hash, model)).fetchone()
        return self._row_to_request(row) if row else None

    def get_requests_by_session(
        self, session_id: str, branch: str = "main"
    ) -> list[RequestRecord]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM requests
                WHERE session_id = ? AND branch_name = ?
                ORDER BY step_id ASC
            """, (session_id, branch)).fetchall()
        return [self._row_to_request(r) for r in rows]

    def get_request_at_step(
        self, session_id: str, step_id: int, branch: str = "main"
    ) -> Optional[RequestRecord]:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM requests
                WHERE session_id = ? AND step_id = ? AND branch_name = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (session_id, step_id, branch)).fetchone()
        return self._row_to_request(row) if row else None

    def search_requests(self, keyword: str, limit: int = 20) -> list[RequestRecord]:
        """プロンプト内容でキーワード検索"""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM requests
                WHERE request_body LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (f"%{keyword}%", limit)).fetchall()
        return [self._row_to_request(r) for r in rows]

    def add_tag(self, request_id: str, tag: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tags FROM requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row:
                tags = json.loads(row["tags"])
                if tag not in tags:
                    tags.append(tag)
                conn.execute(
                    "UPDATE requests SET tags = ? WHERE id = ?",
                    (json.dumps(tags, ensure_ascii=False), request_id)
                )

    def add_memo(self, request_id: str, memo: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE requests SET memo = ? WHERE id = ?",
                (memo, request_id)
            )

    # ── Cost ──────────────────────────────────────────────────

    def get_daily_cost(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as conn:
            row = conn.execute("""
                SELECT SUM(cost_usd) as total FROM requests
                WHERE timestamp LIKE ? AND is_cached = 0
            """, (f"{today}%",)).fetchone()
        return row["total"] or 0.0

    def get_session_cost(self, session_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT SUM(cost_usd) as total FROM requests
                WHERE session_id = ? AND is_cached = 0
            """, (session_id,)).fetchone()
        return row["total"] or 0.0

    # ── Compress old records ───────────────────────────────────

    def compress_old_records(self, older_than_days: int = 30):
        """古いレスポンス本文を圧縮（メタデータは保持）"""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=older_than_days)
        ).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE requests
                SET response_body = '{"compressed": true}'
                WHERE timestamp < ? AND response_body != '{"compressed": true}'
            """, (cutoff,))

    # ── Cost analytics (used by Web UI graphs) ────────────────

    def _cost_where(
        self, date_from: str = "", date_to: str = "",
        provider: str = "", model: str = "",
    ) -> tuple[str, list]:
        """コスト集計用のWHERE句を構築。"""
        conditions = []
        params: list = []
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("timestamp < ?")
            # date_toの翌日までを含める
            params.append(date_to + "T23:59:59")
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if model:
            conditions.append("model = ?")
            params.append(model)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where, params

    def get_daily_costs(self, date_from: str = "", date_to: str = "",
                        provider: str = "", model: str = "") -> list[dict]:
        """日別コスト集計。"""
        where, params = self._cost_where(date_from, date_to, provider, model)
        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT
                    DATE(timestamp) as date,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    COUNT(*) as request_count,
                    SUM(CASE WHEN is_cached = 1 THEN 1 ELSE 0 END) as cached_count
                FROM requests
                {where}
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_cost_by_provider(self, date_from: str = "", date_to: str = "",
                             provider: str = "", model: str = "") -> list[dict]:
        """プロバイダー別コスト。"""
        where, params = self._cost_where(date_from, date_to, provider, model)
        # is_cached除外を追加
        if where:
            where += " AND is_cached = 0"
        else:
            where = "WHERE is_cached = 0"
        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT
                    provider,
                    SUM(cost_usd) as total_cost,
                    COUNT(*) as request_count
                FROM requests
                {where}
                GROUP BY provider
                ORDER BY total_cost DESC
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_cost_by_model(self, date_from: str = "", date_to: str = "",
                          provider: str = "", model: str = "") -> list[dict]:
        """モデル別コスト。"""
        where, params = self._cost_where(date_from, date_to, provider, model)
        if where:
            where += " AND is_cached = 0"
        else:
            where = "WHERE is_cached = 0"
        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT
                    model,
                    provider,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    SUM(reasoning_tokens) as total_reasoning,
                    COUNT(*) as request_count
                FROM requests
                {where}
                GROUP BY model
                ORDER BY total_cost DESC
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_session_costs(self, limit: int = 20) -> list[dict]:
        """セッション別コスト集計。"""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    s.id, s.name, s.created_at,
                    s.total_cost_usd, s.step_count,
                    SUM(CASE WHEN r.is_cached = 1 THEN 1 ELSE 0 END) as cached_count
                FROM sessions s
                LEFT JOIN requests r ON s.id = r.session_id
                GROUP BY s.id
                ORDER BY s.last_accessed DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_sessions_for_comparison(self) -> list[dict]:
        """比較用セッション一覧（2件以上のステップがあるもの）。"""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, name, created_at, total_cost_usd, step_count
                FROM sessions
                WHERE step_count >= 1
                ORDER BY last_accessed DESC
                LIMIT 50
            """).fetchall()
        return [dict(r) for r in rows]

    # ── Filter helpers (used by Web UI) ─────────────────────

    def list_requests(
        self,
        q: str = "",
        provider: str = "",
        model: str = "",
        session_id: str = "",
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[RequestRecord], int]:
        """フィルタ+ソート付きリクエスト一覧。(records, total_count) を返す。"""
        conditions = []
        params: list = []

        if q:
            conditions.append("request_body LIKE ?")
            params.append(f"%{q}%")
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        # ソートカラムのホワイトリスト（SQLインジェクション防止）
        allowed_sort = {
            "timestamp", "provider", "model",
            "input_tokens", "output_tokens", "reasoning_tokens", "cost_usd",
        }
        if sort_by not in allowed_sort:
            sort_by = "timestamp"
        if sort_order not in ("asc", "desc"):
            sort_order = "desc"

        order = f"ORDER BY {sort_by} {sort_order.upper()}"

        with self._conn() as conn:
            # 総件数
            count_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM requests {where}", params
            ).fetchone()
            total = count_row["cnt"]

            # レコード取得
            rows = conn.execute(
                f"SELECT * FROM requests {where} {order} LIMIT ? OFFSET ?",
                params + [limit, offset]
            ).fetchall()

        return [self._row_to_request(r) for r in rows], total

    def get_distinct_providers(self) -> list[str]:
        """DBに存在するプロバイダー一覧を返す。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT provider FROM requests ORDER BY provider"
            ).fetchall()
        return [r["provider"] for r in rows]

    def get_distinct_models(self, provider: str = "") -> list[str]:
        """DBに存在するモデル一覧を返す。providerで絞り込み可能。"""
        with self._conn() as conn:
            if provider:
                rows = conn.execute(
                    "SELECT DISTINCT model FROM requests WHERE provider = ? ORDER BY model",
                    (provider,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT model FROM requests ORDER BY model"
                ).fetchall()
        return [r["model"] for r in rows]

    # ── Alerts (v0.3.0) ──────────────────────────────────────

    def save_alert(self, alert: AlertRecord) -> AlertRecord:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts
                (id, timestamp, level, category, message, details, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.timestamp.isoformat(),
                alert.level,
                alert.category,
                alert.message,
                json.dumps(alert.details, ensure_ascii=False),
                int(alert.acknowledged),
            ))
        return alert

    def get_alerts(
        self, limit: int = 50, unacknowledged_only: bool = False
    ) -> list[AlertRecord]:
        with self._conn() as conn:
            if unacknowledged_only:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def get_unacknowledged_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0"
            ).fetchone()
        return row["cnt"] if row else 0

    def acknowledge_alert(self, alert_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )

    def acknowledge_all_alerts(self):
        with self._conn() as conn:
            conn.execute("UPDATE alerts SET acknowledged = 1")

    def _row_to_alert(self, row) -> AlertRecord:
        return AlertRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            level=row["level"],
            category=row["category"],
            message=row["message"],
            details=json.loads(row["details"]),
            acknowledged=bool(row["acknowledged"]),
        )

    # ── Generic SQL helpers (used by semantic cache) ─────────

    def execute_sql(self, sql: str, params: tuple = ()) -> None:
        """汎用SQL実行（CREATE TABLE, INSERT, UPDATE等）。"""
        with self._conn() as conn:
            conn.execute(sql, params)

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """汎用SELECTクエリ。Row objectではなくtupleのリストを返す。"""
        with self._conn() as conn:
            conn.row_factory = None  # tupleで返す
            rows = conn.execute(sql, params).fetchall()
            conn.row_factory = sqlite3.Row  # 元に戻す
        return rows

    def find_by_id(self, record_id: str) -> Optional[RequestRecord]:
        """IDでRequestRecordを1件取得。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM requests WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_request(row) if row else None

    # ── Converters ────────────────────────────────────────────

    def _row_to_session(self, row) -> Session:
        return Session(
            id=row["id"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            description=row["description"],
            tags=json.loads(row["tags"]),
            total_cost_usd=row["total_cost_usd"],
            step_count=row["step_count"],
        )

    def _row_to_request(self, row) -> RequestRecord:
        return RequestRecord(
            id=row["id"],
            session_id=row["session_id"],
            step_id=row["step_id"],
            parent_id=row["parent_id"],
            branch_name=row["branch_name"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            provider=row["provider"],
            model=row["model"],
            prompt_hash=row["prompt_hash"],
            request_body=json.loads(row["request_body"]),
            response_body=json.loads(row["response_body"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            reasoning_tokens=row["reasoning_tokens"] if "reasoning_tokens" in row.keys() else 0,
            cost_usd=row["cost_usd"],
            is_cached=bool(row["is_cached"]),
            tags=json.loads(row["tags"]),
            memo=row["memo"],
        )