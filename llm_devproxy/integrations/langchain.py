"""
LangChain integration — DevProxyCallbackHandler.

Usage:
    from llm_devproxy.integrations.langchain import DevProxyCallbackHandler

    handler = DevProxyCallbackHandler(daily_limit_usd=1.0)
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    chain = prompt | llm | parser
    result = chain.invoke({"input": "Hello"})
    # → All API calls are automatically recorded by llm-devproxy
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install with: pip install 'llm-devproxy[langchain]'"
    )

from llm_devproxy.core.models import ProxyConfig, RequestRecord
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


def _extract_reasoning_tokens(generation: Any) -> int:
    """LLMResultのgenerationから推論トークンを抽出。"""
    info = getattr(generation, "generation_info", {}) or {}

    # OpenAI: completion_tokens_details.reasoning_tokens
    details = info.get("completion_tokens_details", {})
    if details and isinstance(details, dict):
        rt = details.get("reasoning_tokens", 0)
        if rt:
            return rt

    # Check usage metadata (some LangChain wrappers put it here)
    usage = info.get("usage", {})
    if usage:
        rt = usage.get("reasoning_tokens", 0)
        if rt:
            return rt

    return 0


class DevProxyCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that records all LLM calls via llm-devproxy.

    Supports:
    - Token counting (input, output, reasoning)
    - Cost calculation per model
    - Session / step management
    - Alert integration (daily cost, reasoning ratio, etc.)
    """

    def __init__(
        self,
        daily_limit_usd: float = 5.0,
        session_name: str = "langchain",
        db_path: str = ".llm_devproxy.db",
        alert_daily_threshold: float = 0.8,
        alert_reasoning_ratio: float = 0.7,
        alert_single_cost_usd: float = 0.10,
        verbose: bool = True,
    ):
        super().__init__()
        self.verbose = verbose
        self._step_counter = 0

        # Storage & config
        config = ProxyConfig(
            daily_limit_usd=daily_limit_usd,
            alert_daily_threshold=alert_daily_threshold,
            alert_reasoning_ratio=alert_reasoning_ratio,
            alert_single_cost_usd=alert_single_cost_usd,
        )
        self._storage = Storage(db_path)
        self._config = config
        self._alert_manager = AlertManager(config, self._storage)

        # Session
        from llm_devproxy.core.models import Session
        self._session = Session(name=session_name, description="LangChain session")
        self._storage.save_session(self._session)

        # Active call tracking (keyed by run_id)
        self._active_calls: Dict[str, dict] = {}

    # ── LLM events ──────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating."""
        model_name = kwargs.get("invocation_params", {}).get("model_name", "")
        if not model_name:
            model_name = kwargs.get("invocation_params", {}).get("model", "")
        if not model_name:
            model_name = serialized.get("kwargs", {}).get("model_name", "unknown")

        self._active_calls[str(run_id)] = {
            "start_time": time.time(),
            "model": model_name,
            "prompts": prompts,
            "tags": tags or [],
            "metadata": metadata or {},
        }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts (ChatOpenAI, etc.)."""
        model_name = kwargs.get("invocation_params", {}).get("model_name", "")
        if not model_name:
            model_name = kwargs.get("invocation_params", {}).get("model", "")
        if not model_name:
            model_name = serialized.get("kwargs", {}).get("model_name", "unknown")

        # Convert BaseMessage objects to dicts
        prompt_messages = []
        for msg_list in messages:
            for msg in msg_list:
                role = getattr(msg, "type", "user")
                content = getattr(msg, "content", str(msg))
                prompt_messages.append({"role": role, "content": content})

        self._active_calls[str(run_id)] = {
            "start_time": time.time(),
            "model": model_name,
            "messages": prompt_messages,
            "tags": tags or [],
            "metadata": metadata or {},
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes generating."""
        run_key = str(run_id)
        call_info = self._active_calls.pop(run_key, None)
        if not call_info:
            return

        model = call_info.get("model", "unknown")
        provider = _detect_provider(model)
        self._step_counter += 1

        # Extract token usage from llm_output
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        if not token_usage:
            token_usage = llm_output.get("usage", {})

        input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0
        output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0

        # Google GenAI format: usage_metadata with different key names
        usage_metadata = llm_output.get("usage_metadata", {})
        if usage_metadata and not input_tokens:
            input_tokens = usage_metadata.get("prompt_token_count", 0) or 0
            output_tokens = usage_metadata.get("candidates_token_count", 0) or 0

        # Also check generation_info on individual generations
        if not input_tokens and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    usage_meta = gen_info.get("usage_metadata", {})
                    if usage_meta:
                        input_tokens = input_tokens or usage_meta.get("prompt_token_count", 0) or 0
                        output_tokens = output_tokens or usage_meta.get("candidates_token_count", 0) or 0

        # Reasoning tokens
        reasoning_tokens = 0
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    reasoning_tokens += _extract_reasoning_tokens(gen)

        # Google GenAI: thoughts_token_count
        if not reasoning_tokens and usage_metadata:
            reasoning_tokens = usage_metadata.get("thoughts_token_count", 0) or 0
        if not reasoning_tokens and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    usage_meta = gen_info.get("usage_metadata", {})
                    if usage_meta:
                        reasoning_tokens = reasoning_tokens or usage_meta.get("thoughts_token_count", 0) or 0

        # Also check in token_usage directly (OpenAI format)
        if not reasoning_tokens:
            details = token_usage.get("completion_tokens_details", {})
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens", 0) or 0

        # Cost
        cost = estimate_cost(model, input_tokens, output_tokens, reasoning_tokens)

        # Build request/response bodies for storage
        messages = call_info.get("messages", [])
        if not messages:
            prompts = call_info.get("prompts", [])
            messages = [{"role": "user", "content": p} for p in prompts]

        response_text = ""
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    response_text += getattr(gen, "text", str(gen))

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

        # Compute prompt hash
        import hashlib, json
        prompt_hash = hashlib.sha256(
            json.dumps(request_body, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Save record
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
            tags=call_info.get("tags", []),
        )
        self._storage.save_request(record)
        self._storage.update_session_stats(self._session.id, cost)

        # Alert check
        alerts = self._alert_manager.evaluate(record, self._session.id)

        # Terminal output
        if self.verbose:
            elapsed = time.time() - call_info["start_time"]
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
                    f"({ratio:.0f}% of output) | "
                    f"Output: {output_tokens:,} | Cost: ${cost:.6f}"
                )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors out."""
        run_key = str(run_id)
        call_info = self._active_calls.pop(run_key, None)
        if self.verbose and call_info:
            model = call_info.get("model", "unknown")
            print(f"❌ [{model}] Error: {error}")

    # ── Chain events (session tracking) ─────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        pass  # Session tracking is handled at init

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes."""
        pass

    # ── Tool events ─────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts."""
        if self.verbose:
            tool_name = serialized.get("name", "unknown_tool")
            print(f"🔧 Tool: {tool_name}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes."""
        pass

    # ── Convenience properties ──────────────────────────

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def total_cost(self) -> float:
        return self._storage.get_session_cost(self._session.id)

    @property
    def daily_cost(self) -> float:
        return self._storage.get_daily_cost()
