"""
Core data models for llm-devproxy.
All LLM API calls are automatically recorded - nothing is lost.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


@dataclass
class RequestRecord:
    """A single LLM API call, automatically recorded."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    step_id: int = 0
    parent_id: Optional[str] = None      # ツリー構造（どのリクエストの次か）
    branch_name: str = "main"            # rewind後の別試みはbranch名で区別

    # リクエスト情報
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = ""                   # "openai" / "anthropic" / "gemini"
    model: str = ""
    prompt_hash: str = ""                # キャッシュキー
    request_body: dict = field(default_factory=dict)
    response_body: dict = field(default_factory=dict)

    # コスト情報
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    is_cached: bool = False              # キャッシュから返したか

    # 後から追記できるメタデータ
    tags: list[str] = field(default_factory=list)
    memo: str = ""


@dataclass
class Session:
    """
    A group of related API calls (e.g., one agent run).
    Think of it like a Git repository for LLM interactions.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # 集計（自動更新）
    total_cost_usd: float = 0.0
    step_count: int = 0


@dataclass
class CostLimit:
    """Cost guard configuration."""
    scope: str = "global"               # "global" / "session" / "daily"
    limit_usd: float = 1.0
    current_usd: float = 0.0
    action: str = "mock"                # "mock" / "block" / "warn"
    reset_at: Optional[datetime] = None  # dailyリセット用


@dataclass
class ProxyConfig:
    """Main configuration for llm-devproxy."""
    # ストレージ
    db_path: str = ".llm_devproxy.db"

    # 全量自動記録（デフォルトON・これがコアコンセプト）
    auto_record: bool = True

    # キャッシュ
    cache_enabled: bool = True
    cache_mode: str = "exact"           # "exact" / "semantic"(将来)

    # コスト管理
    daily_limit_usd: float = 1.0
    session_limit_usd: Optional[float] = None
    on_exceed: str = "mock"             # "mock" / "block" / "warn"

    # ストレージ管理
    compress_after_days: int = 30       # 古いレスポンス本文を圧縮
    auto_delete: bool = False           # 完全削除は明示的にのみ

    # モック設定
    mock_response: str = "[MOCK] Cost limit reached. This is a mock response."
