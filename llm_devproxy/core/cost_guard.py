"""
Cost Guard - prevents API cost explosions during development.
"""

from .models import ProxyConfig
from .storage import Storage


# 主要モデルの料金テーブル (USD per 1K tokens)
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":             {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":        {"input": 0.010,  "output": 0.030},
    "o1":                 {"input": 0.015,  "output": 0.060},
    "o1-mini":            {"input": 0.003,  "output": 0.012},
    # Anthropic
    "claude-opus-4-5":        {"input": 0.015,  "output": 0.075},
    "claude-sonnet-4-5":      {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":       {"input": 0.00025,"output": 0.00125},
    # Google
    "gemini-1.5-pro":     {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash":   {"input": 0.000075,"output": 0.0003},
    "gemini-2.0-flash":   {"input": 0.0001,  "output": 0.0004},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """トークン数からコストを計算"""
    pricing = PRICING.get(model)
    if not pricing:
        # 不明モデルはgpt-4o相当で概算
        pricing = {"input": 0.0025, "output": 0.010}
    return (
        input_tokens / 1000 * pricing["input"]
        + output_tokens / 1000 * pricing["output"]
    )


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
