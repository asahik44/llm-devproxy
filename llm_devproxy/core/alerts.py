"""
Alert Manager - monitors costs and token usage, fires alerts.
"""

from datetime import datetime, timezone
from typing import Optional

from .models import AlertRecord, ProxyConfig, RequestRecord
from .storage import Storage


class AlertManager:
    """
    APIコールのたびに呼ばれ、各種閾値をチェックしてアラートを発行する。
    アラートはDB保存 + ターミナル表示の2系統。
    """

    def __init__(self, config: ProxyConfig, storage: Storage):
        self.config = config
        self.storage = storage

    def evaluate(self, record: RequestRecord, session_id: str) -> list[AlertRecord]:
        """
        1回のAPIコール後に全チェックを実行。
        発行されたアラートのリストを返す。
        """
        alerts: list[AlertRecord] = []

        # 1. デイリーコスト閾値
        alert = self._check_daily_cost()
        if alert:
            alerts.append(alert)

        # 2. セッションコスト閾値
        alert = self._check_session_cost(session_id)
        if alert:
            alerts.append(alert)

        # 3. 単一リクエストコスト
        alert = self._check_single_cost(record)
        if alert:
            alerts.append(alert)

        # 4. 推論トークン比率
        alert = self._check_reasoning_ratio(record)
        if alert:
            alerts.append(alert)

        # DB保存 + ターミナル表示
        for a in alerts:
            self.storage.save_alert(a)
            self._print_alert(a)

        return alerts

    def _check_daily_cost(self) -> Optional[AlertRecord]:
        daily = self.storage.get_daily_cost()
        limit = self.config.daily_limit_usd
        threshold = self.config.alert_daily_threshold
        ratio = daily / limit if limit > 0 else 0

        if ratio >= 1.0:
            return AlertRecord(
                level="critical",
                category="cost_daily",
                message=f"デイリー上限に到達: ${daily:.4f} / ${limit:.4f} (100%)",
                details={"daily_cost": daily, "limit": limit, "ratio": ratio},
            )
        elif ratio >= threshold:
            return AlertRecord(
                level="warning",
                category="cost_daily",
                message=f"デイリーコスト {ratio:.0%} 到達: ${daily:.4f} / ${limit:.4f}",
                details={"daily_cost": daily, "limit": limit, "ratio": ratio},
            )
        return None

    def _check_session_cost(self, session_id: str) -> Optional[AlertRecord]:
        if not self.config.session_limit_usd:
            return None

        session_cost = self.storage.get_session_cost(session_id)
        limit = self.config.session_limit_usd
        threshold = self.config.alert_session_threshold
        ratio = session_cost / limit if limit > 0 else 0

        if ratio >= 1.0:
            return AlertRecord(
                level="critical",
                category="cost_session",
                message=f"セッション上限に到達: ${session_cost:.4f} / ${limit:.4f}",
                details={"session_cost": session_cost, "limit": limit, "ratio": ratio},
            )
        elif ratio >= threshold:
            return AlertRecord(
                level="warning",
                category="cost_session",
                message=f"セッションコスト {ratio:.0%} 到達: ${session_cost:.4f} / ${limit:.4f}",
                details={"session_cost": session_cost, "limit": limit, "ratio": ratio},
            )
        return None

    def _check_single_cost(self, record: RequestRecord) -> Optional[AlertRecord]:
        threshold = self.config.alert_single_cost_usd
        if record.cost_usd >= threshold:
            return AlertRecord(
                level="warning",
                category="cost_single",
                message=(
                    f"高コストリクエスト: ${record.cost_usd:.4f} "
                    f"(model={record.model}, "
                    f"in={record.input_tokens:,}, out={record.output_tokens:,})"
                ),
                details={
                    "cost": record.cost_usd,
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "reasoning_tokens": record.reasoning_tokens,
                    "request_id": record.id,
                },
            )
        return None

    def _check_reasoning_ratio(self, record: RequestRecord) -> Optional[AlertRecord]:
        if record.reasoning_tokens == 0:
            return None

        total_output = record.output_tokens + record.reasoning_tokens
        ratio = record.reasoning_tokens / total_output if total_output > 0 else 0

        if ratio >= self.config.alert_reasoning_ratio:
            return AlertRecord(
                level="info",
                category="reasoning_ratio",
                message=(
                    f"推論トークン比率 {ratio:.0%}: "
                    f"🧠{record.reasoning_tokens:,} / "
                    f"out={record.output_tokens:,} "
                    f"(model={record.model})"
                ),
                details={
                    "reasoning_tokens": record.reasoning_tokens,
                    "output_tokens": record.output_tokens,
                    "ratio": ratio,
                    "model": record.model,
                    "request_id": record.id,
                },
            )
        return None

    def _print_alert(self, alert: AlertRecord):
        """ターミナルに目立つアラートを表示"""
        icons = {
            "critical": "🚨",
            "warning": "⚠️ ",
            "info": "ℹ️ ",
        }
        colors = {
            "critical": "\033[91m",  # red
            "warning": "\033[93m",   # yellow
            "info": "\033[96m",      # cyan
        }
        reset = "\033[0m"

        icon = icons.get(alert.level, "📢")
        color = colors.get(alert.level, "")

        print(f"{color}{icon} [{alert.level.upper()}] {alert.message}{reset}")
