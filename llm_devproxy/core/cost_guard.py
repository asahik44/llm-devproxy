"""
Cost Guard - prevents API cost explosions during development.
"""

from .models import ProxyConfig
from .storage import Storage


# 主要モデルの料金テーブル (USD per 1K tokens)
# reasoning: 推論トークンの出力単価（未対応モデルは None）
PRICING: dict[str, dict[str, float | None]] = {
    # OpenAI
    "gpt-4o":             {"input": 0.0025, "output": 0.010, "reasoning": None},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006, "reasoning": None},
    "gpt-4-turbo":        {"input": 0.010,  "output": 0.030, "reasoning": None},
    "o1":                 {"input": 0.015,  "output": 0.060, "reasoning": 0.060},
    "o1-mini":            {"input": 0.003,  "output": 0.012, "reasoning": 0.012},
    "o1-pro":             {"input": 0.150,  "output": 0.600, "reasoning": 0.600},
    "o3":                 {"input": 0.010,  "output": 0.040, "reasoning": 0.040},
    "o3-mini":            {"input": 0.001,  "output": 0.004, "reasoning": 0.004},
    "o4-mini":            {"input": 0.001,  "output": 0.004, "reasoning": 0.004},
    # Anthropic
    "claude-opus-4-5":        {"input": 0.015,  "output": 0.075, "reasoning": None},
    "claude-sonnet-4-5":      {"input": 0.003,  "output": 0.015, "reasoning": 0.015},
    "claude-haiku-4-5":       {"input": 0.00025,"output": 0.00125, "reasoning": None},
    # Google
    "gemini-1.5-pro":     {"input": 0.00125, "output": 0.005, "reasoning": None},
    "gemini-1.5-flash":   {"input": 0.000075,"output": 0.0003, "reasoning": None},
    "gemini-2.0-flash":   {"input": 0.0001,  "output": 0.0004, "reasoning": None},
    "gemini-2.5-pro":     {"input": 0.00125, "output": 0.010, "reasoning": 0.010},
    "gemini-2.5-flash":   {"input": 0.000075,"output": 0.0003, "reasoning": 0.0003},
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    """トークン数からコストを計算（推論トークン対応）"""
    pricing = PRICING.get(model)
    if not pricing:
        # 不明モデルはgpt-4o相当で概算
        pricing = {"input": 0.0025, "output": 0.010, "reasoning": None}

    cost = (
        input_tokens / 1000 * pricing["input"]
        + output_tokens / 1000 * pricing["output"]
    )

    # 推論トークンのコスト加算
    if reasoning_tokens > 0 and pricing.get("reasoning"):
        cost += reasoning_tokens / 1000 * pricing["reasoning"]

    return cost


def is_reasoning_model(model: str) -> bool:
    """推論トークンを使うモデルかどうかを判定"""
    pricing = PRICING.get(model, {})
    return pricing.get("reasoning") is not None


class CostGuard:
    def __init__(self, config: ProxyConfig, storage: Storage):
        self.config = config
        self.storage = storage

    def check(self, session_id: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        allowed=False のとき、configのactionに従ってmock/blockする。
        """
        # デイリー上限チェック
        daily = self.storage.get_daily_cost()
        if daily >= self.config.daily_limit_usd:
            return False, (
                f"Daily limit reached: ${daily:.4f} / ${self.config.daily_limit_usd}"
            )

        # セッション上限チェック
        if self.config.session_limit_usd:
            session_cost = self.storage.get_session_cost(session_id)
            if session_cost >= self.config.session_limit_usd:
                return False, (
                    f"Session limit reached: "
                    f"${session_cost:.4f} / ${self.config.session_limit_usd}"
                )

        return True, ""

    def warning_threshold(self, session_id: str) -> tuple[bool, float]:
        """80%に達したら警告（True, 使用率）を返す"""
        daily = self.storage.get_daily_cost()
        ratio = daily / self.config.daily_limit_usd if self.config.daily_limit_usd else 0
        return ratio >= 0.8, ratio
