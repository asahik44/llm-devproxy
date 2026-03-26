"""
LlamaIndex integration — DevProxyCallbackHandler.

Usage:
    from llm_devproxy.integrations.llamaindex import DevProxyCallbackHandler
    from llama_index.core import Settings

    handler = DevProxyCallbackHandler(daily_limit_usd=1.0)
    Settings.callback_manager.add_handler(handler)

    # All LLM calls are now automatically recorded
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install 'llm-devproxy[llamaindex]'"
    )

from llm_devproxy.core.models import ProxyConfig, RequestRecord, Session
from llm_devproxy.core.storage import Storage
from llm_devproxy.core.cost_guard import estimate_cost
from llm_devproxy.core.alerts import AlertManager


def _detect_provider(model_name: str) -> str:
    """モデル名からプロバイダーを推定。"""
    name = model_name.lower()
    if any(k in name for k in ("gpt", "o1", "o3", "o4", "davinci", "turbo")):
        return "openai"
    if any(k in name for k in ("claude", "haiku", "sonnet", "opus")):
        return "anthropic"
    if any(k in name for k in ("gemini", "palm")):
        return "gemini"
    return "unknown"


class DevProxyCallbackHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler that records all LLM calls via llm-devproxy.
    """

    def __init__(
        self,
        daily_limit_usd: float = 5.0,
        session_name: str = "llamaindex",
        db_path: str = ".llm_devproxy.db",
        alert_daily_threshold: float = 0.8,
        alert_reasoning_ratio: float = 0.7,
        alert_single_cost_usd: float = 0.10,
        verbose: bool = True,
    ):
        # LlamaIndex BaseCallbackHandler requires event_starts_to_ignore and event_ends_to_ignore
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self.verbose = verbose
        self._step_counter = 0

        config = ProxyConfig(
            daily_limit_usd=daily_limit_usd,
            alert_daily_threshold=alert_daily_threshold,
            alert_reasoning_ratio=alert_reasoning_ratio,
            alert_single_cost_usd=alert_single_cost_usd,
        )
        self._storage = Storage(db_path)
        self._config = config
        self._alert_manager = AlertManager(config, self._storage)

        self._session = Session(name=session_name, description="LlamaIndex session")
        self._storage.save_session(self._session)

        # Active event tracking
        self._active_events: Dict[str, dict] = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts."""
        if event_type == CBEventType.LLM:
            self._active_events[event_id] = {
                "start_time": time.time(),
                "payload": payload or {},
            }
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends."""
        if event_type != CBEventType.LLM:
            return

        start_info = self._active_events.pop(event_id, None)
        if not start_info:
            return

        payload = payload or {}
        self._step_counter += 1

        # Extract model info
        response = payload.get(EventPayload.RESPONSE)
        model = "unknown"
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        response_text = ""

        if response is not None:
            # Try to get model from response metadata
            raw = getattr(response, "raw", None)
            if raw:
                model = getattr(raw, "model", "unknown")
                usage = getattr(raw, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
                    # Reasoning tokens
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

            # Response text
            response_text = str(response)[:500]

        # Fallback: try getting model from start payload
        if model == "unknown":
            serialized_model = start_info.get("payload", {}).get(EventPayload.SERIALIZED, {})
            model = serialized_model.get("model", "unknown")

        provider = _detect_provider(model)
        cost = estimate_cost(model, input_tokens, output_tokens, reasoning_tokens)

        # Build messages from start payload
        messages_payload = start_info.get("payload", {}).get(EventPayload.MESSAGES, [])
        messages = []
        for msg in messages_payload:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", str(msg))
            if hasattr(role, "value"):
                role = role.value
            messages.append({"role": str(role), "content": str(content)})

        if not messages:
            # Fallback: try prompt
            prompt = start_info.get("payload", {}).get(EventPayload.PROMPT, "")
            if prompt:
                messages = [{"role": "user", "content": str(prompt)}]

        request_body = {"messages": messages}
        response_body = {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": model,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
        }

        prompt_hash = hashlib.sha256(
            json.dumps(request_body, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        record = RequestRecord(
            session_id=self._session.id,
            step_id=self._step_counter,
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
        )
        self._storage.save_request(record)
        self._storage.update_session_stats(self._session.id, cost)

        # Alerts
        alerts = self._alert_manager.check(
            self._session.id, model, cost, input_tokens, output_tokens, reasoning_tokens,
        )

        if self.verbose:
            elapsed = time.time() - start_info["start_time"]
            print(
                f"📝 [{provider}] {model} | "
                f"in={input_tokens:,} out={output_tokens:,} | "
                f"${cost:.6f} | {elapsed:.1f}s"
            )
            if reasoning_tokens > 0:
                total_out = output_tokens + reasoning_tokens
                ratio = reasoning_tokens / total_out * 100 if total_out else 0
                print(
                    f"🧠 Reasoning tokens: {reasoning_tokens:,} "
                    f"({ratio:.0f}% of output)"
                )
            for alert in alerts:
                level_icon = {"critical": "🚨", "warning": "⚠️ ", "info": "ℹ️ "}.get(alert.level, "")
                print(f"{level_icon} [{alert.level.upper()}] {alert.message}")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Required by BaseCallbackHandler."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Required by BaseCallbackHandler."""
        pass

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def total_cost(self) -> float:
        return self._storage.get_session_cost(self._session.id)

    @property
    def daily_cost(self) -> float:
        return self._storage.get_daily_cost()
