"""
Seed data for v0.3.0 screenshots.
Usage: python seed_screenshot_data.py && llm-devproxy web
"""

import random
import uuid
from datetime import datetime, timedelta, timezone

from llm_devproxy.core.models import ProxyConfig, RequestRecord, Session, AlertRecord
from llm_devproxy.core.storage import Storage


def main():
    db_path = ".llm_devproxy_screenshot.db"
    storage = Storage(db_path)
    print(f"DB: {db_path}")

    now = datetime.now(timezone.utc)

    # ── Sessions ──
    sessions = []
    session_defs = [
        ("agent-refactor", "エージェント リファクタリング"),
        ("rag-pipeline-v2", "RAGパイプライン v2 検証"),
        ("code-review-bot", "コードレビューBot開発"),
    ]
    for name, desc in session_defs:
        s = Session(name=name, description=desc)
        storage.save_session(s)
        sessions.append(s)

    # ── Reasoning model records (the hero data) ──
    reasoning_records = [
        # o3 — 推論が重い例（スクショの主役）
        {
            "provider": "openai", "model": "o3",
            "prompt": "このPythonコードのバグを見つけて、修正案を3つ提示してください。各案のトレードオフも説明して。",
            "response": "コードを分析しました。3つの修正案を提示します...",
            "input_tokens": 2500, "output_tokens": 800, "reasoning_tokens": 4200,
        },
        # o1 — 推論比率が非常に高い例
        {
            "provider": "openai", "model": "o1",
            "prompt": "数学的帰納法を使って、すべての自然数nに対して 1+2+...+n = n(n+1)/2 を証明してください。",
            "response": "数学的帰納法による証明を示します...",
            "input_tokens": 150, "output_tokens": 400, "reasoning_tokens": 3600,
        },
        # o4-mini — 軽量推論
        {
            "provider": "openai", "model": "o4-mini",
            "prompt": "FizzBuzzを最も効率的に実装するアルゴリズムを考えて",
            "response": "最も効率的な実装方法は...",
            "input_tokens": 80, "output_tokens": 200, "reasoning_tokens": 350,
        },
        # Claude extended thinking
        {
            "provider": "anthropic", "model": "claude-sonnet-4-5",
            "prompt": "マイクロサービスアーキテクチャとモノリスの比較分析を、コスト・開発速度・スケーラビリティの観点で行って",
            "response": "3つの観点から詳細に比較分析します...",
            "input_tokens": 300, "output_tokens": 1200, "reasoning_tokens": 2800,
        },
        # Gemini thinking
        {
            "provider": "gemini", "model": "gemini-2.5-pro",
            "prompt": "分散システムにおけるCAP定理の実際の適用例を3つ挙げて、それぞれの設計判断を説明して",
            "response": "CAP定理の実際の適用例を解説します...",
            "input_tokens": 200, "output_tokens": 900, "reasoning_tokens": 1500,
        },
        # Gemini Flash thinking
        {
            "provider": "gemini", "model": "gemini-2.5-flash",
            "prompt": "Pythonのasyncioの基本的な使い方をコード付きで教えて",
            "response": "asyncioの基本パターンを説明します...",
            "input_tokens": 100, "output_tokens": 600, "reasoning_tokens": 400,
        },
    ]

    # ── Normal (non-reasoning) records ──
    normal_records = [
        {
            "provider": "openai", "model": "gpt-4o",
            "prompt": "ReactのuseEffectの使い方を解説して",
            "response": "useEffectはReactのフック関数で...",
            "input_tokens": 120, "output_tokens": 800, "reasoning_tokens": 0,
        },
        {
            "provider": "openai", "model": "gpt-4o-mini",
            "prompt": "JSONをパースするPython関数を書いて",
            "response": "以下のような関数を作成できます...",
            "input_tokens": 60, "output_tokens": 300, "reasoning_tokens": 0,
        },
        {
            "provider": "anthropic", "model": "claude-haiku-4-5",
            "prompt": "Dockerfileのベストプラクティスは？",
            "response": "Dockerfileを書く際のベストプラクティス...",
            "input_tokens": 80, "output_tokens": 500, "reasoning_tokens": 0,
        },
        {
            "provider": "gemini", "model": "gemini-2.0-flash",
            "prompt": "Git rebaseとmergeの違いを教えて",
            "response": "rebaseとmergeの主な違いは...",
            "input_tokens": 50, "output_tokens": 400, "reasoning_tokens": 0,
        },
    ]

    # ── Cached record ──
    cached_records = [
        {
            "provider": "openai", "model": "gpt-4o",
            "prompt": "ReactのuseEffectの使い方を解説して",
            "response": "useEffectはReactのフック関数で...",
            "input_tokens": 120, "output_tokens": 800, "reasoning_tokens": 0,
            "is_cached": True,
        },
    ]

    # ── Pricing for cost calculation ──
    pricing_input = {
        "o3": 0.010, "o1": 0.015, "o4-mini": 0.001,
        "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015,
        "claude-sonnet-4-5": 0.003, "claude-haiku-4-5": 0.00025,
        "gemini-2.5-pro": 0.00125, "gemini-2.5-flash": 0.000075,
        "gemini-2.0-flash": 0.0001,
    }
    pricing_output = {
        "o3": 0.040, "o1": 0.060, "o4-mini": 0.004,
        "gpt-4o": 0.010, "gpt-4o-mini": 0.0006,
        "claude-sonnet-4-5": 0.015, "claude-haiku-4-5": 0.00125,
        "gemini-2.5-pro": 0.010, "gemini-2.5-flash": 0.0003,
        "gemini-2.0-flash": 0.0004,
    }

    def calc_cost(model, inp, out, reason):
        c = inp / 1000 * pricing_input.get(model, 0.002)
        c += out / 1000 * pricing_output.get(model, 0.01)
        c += reason / 1000 * pricing_output.get(model, 0.01)  # reasoning = output rate
        return c

    def make_request_body(provider, model, prompt):
        if provider == "anthropic":
            return {
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": 4096,
            }
        elif provider == "gemini":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            return {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4096,
            }

    def make_response_body(provider, model, response, inp, out, reason):
        if provider == "anthropic":
            content = []
            if reason > 0:
                content.append({"type": "thinking", "text": "Let me analyze this step by step..." * (reason // 50)})
            content.append({"type": "text", "text": response})
            return {
                "content": content,
                "model": model,
                "usage": {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": reason},
            }
        elif provider == "gemini":
            return {
                "choices": [{"message": {"role": "model", "content": response}}],
                "model": model,
                "usage": {"prompt_tokens": inp, "completion_tokens": out, "reasoning_tokens": reason},
            }
        else:
            return {
                "choices": [{"message": {"role": "assistant", "content": response}}],
                "model": model,
                "usage": {"prompt_tokens": inp, "completion_tokens": out, "reasoning_tokens": reason},
            }

    # ── Insert records ──
    all_records = reasoning_records + normal_records + cached_records
    # Repeat some to get ~30 records spread across days
    extra = random.choices(normal_records, k=18)
    all_records.extend(extra)

    record_count = 0
    for i, rec in enumerate(all_records):
        session = sessions[i % len(sessions)]
        time_offset = timedelta(
            days=random.randint(0, 5),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        ts = now - time_offset

        is_cached = rec.get("is_cached", False)
        cost = 0.0 if is_cached else calc_cost(
            rec["model"], rec["input_tokens"], rec["output_tokens"], rec["reasoning_tokens"]
        )

        record = RequestRecord(
            session_id=session.id,
            step_id=i + 1,
            timestamp=ts,
            provider=rec["provider"],
            model=rec["model"],
            prompt_hash=str(uuid.uuid4()),
            request_body=make_request_body(rec["provider"], rec["model"], rec["prompt"]),
            response_body=make_response_body(
                rec["provider"], rec["model"], rec["response"],
                rec["input_tokens"], rec["output_tokens"], rec["reasoning_tokens"],
            ),
            input_tokens=rec["input_tokens"],
            output_tokens=rec["output_tokens"],
            reasoning_tokens=rec["reasoning_tokens"],
            cost_usd=cost,
            is_cached=is_cached,
        )
        storage.save_request(record)
        storage.update_session_stats(session.id, record.cost_usd)
        record_count += 1

    # ── Insert alerts ──
    alerts = [
        AlertRecord(
            timestamp=now - timedelta(hours=1),
            level="critical",
            category="cost_daily",
            message="デイリー上限に到達: $1.0000 / $1.0000 (100%)",
            details={"daily_cost": 1.0, "limit": 1.0, "ratio": 1.0},
        ),
        AlertRecord(
            timestamp=now - timedelta(hours=2),
            level="warning",
            category="cost_single",
            message="高コストリクエスト: $0.2120 (model=o3, in=2,500, out=800)",
            details={"cost": 0.212, "model": "o3", "input_tokens": 2500, "output_tokens": 800, "reasoning_tokens": 4200},
        ),
        AlertRecord(
            timestamp=now - timedelta(hours=3),
            level="info",
            category="reasoning_ratio",
            message="推論トークン比率 90%: 🧠3,600 / out=400 (model=o1)",
            details={"reasoning_tokens": 3600, "output_tokens": 400, "ratio": 0.9, "model": "o1"},
        ),
        AlertRecord(
            timestamp=now - timedelta(hours=4),
            level="warning",
            category="cost_daily",
            message="デイリーコスト 85% 到達: $0.8500 / $1.0000",
            details={"daily_cost": 0.85, "limit": 1.0, "ratio": 0.85},
        ),
        AlertRecord(
            timestamp=now - timedelta(hours=6),
            level="info",
            category="reasoning_ratio",
            message="推論トークン比率 70%: 🧠2,800 / out=1,200 (model=claude-sonnet-4-5)",
            details={"reasoning_tokens": 2800, "output_tokens": 1200, "ratio": 0.7, "model": "claude-sonnet-4-5"},
        ),
    ]

    # Leave some unacknowledged
    for i, alert in enumerate(alerts):
        if i >= 3:
            alert.acknowledged = True
        storage.save_alert(alert)

    print(f"\n✅ {record_count} records + {len(alerts)} alerts inserted")
    print(f"   DB: {db_path}")
    print(f"\n📸 スクショ撮影用の起動コマンド:")
    print(f"   llm-devproxy web --db {db_path}")
    print(f"   → http://127.0.0.1:8765")


if __name__ == "__main__":
    main()
