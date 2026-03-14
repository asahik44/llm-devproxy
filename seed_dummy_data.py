"""
Seed dummy data for Web UI testing.
Usage: python seed_dummy_data.py
"""

import random
import uuid
from datetime import datetime, timedelta, timezone

from llm_devproxy.core.models import ProxyConfig, RequestRecord, Session
from llm_devproxy.core.storage import Storage


def main():
    db_path = ".llm_devproxy.db"
    storage = Storage(db_path)
    print(f"DB: {db_path}")

    # ── Sessions ──
    sessions = []
    session_defs = [
        ("chatbot-dev", "チャットボット開発テスト"),
        ("rag-pipeline", "RAGパイプライン検証"),
        ("code-review", "コードレビューエージェント"),
    ]
    for name, desc in session_defs:
        s = Session(name=name, description=desc)
        storage.save_session(s)
        sessions.append(s)
        print(f"  Session: {name}")

    # ── Request templates ──
    providers = [
        ("openai", ["gpt-4o", "gpt-4o-mini"]),
        ("anthropic", ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]),
        ("gemini", ["gemini-2.0-flash"]),
    ]

    prompts_ja = [
        "Pythonでフィボナッチ数列を計算するコードを書いて",
        "FastAPIでREST APIを作る方法を教えて",
        "SQLiteのインデックスの仕組みを説明して",
        "Dockerfileのベストプラクティスは？",
        "TypeScriptの型ガードについて教えて",
        "ReactのuseEffectの使い方を解説して",
        "機械学習モデルの評価指標について説明して",
        "Git rebaseとmergeの違いは？",
        "Kubernetes Podのライフサイクルを教えて",
        "AWSのLambdaとECSの使い分けは？",
    ]

    prompts_en = [
        "Write a Python function to parse JSON safely",
        "Explain the difference between REST and GraphQL",
        "How to implement retry logic with exponential backoff?",
        "Best practices for error handling in async Python",
        "Explain CAP theorem in distributed systems",
        "How to optimize SQL queries with large datasets?",
        "What is the observer pattern? Give an example",
        "Explain JWT token authentication flow",
        "How to set up CI/CD with GitHub Actions?",
        "Compare Redis vs Memcached for caching",
    ]

    responses_short = [
        "以下のようにコードを書くことができます。まず基本的な実装から...",
        "いくつかのアプローチがあります。最も一般的な方法は...",
        "主な違いは以下の3点です。1つ目は...",
        "Here's a step-by-step approach to solve this...",
        "The key concept here is that...",
        "Let me break this down into smaller parts...",
        "There are several best practices to consider...",
        "A common pattern for this is to use...",
    ]

    all_prompts = prompts_ja + prompts_en
    now = datetime.now(timezone.utc)

    record_count = 0
    for i in range(50):
        session = random.choice(sessions)
        provider_name, models = random.choice(providers)
        model = random.choice(models)
        prompt = random.choice(all_prompts)
        response = random.choice(responses_short)

        # 時間を過去数日にばらけさせる
        time_offset = timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        ts = now - time_offset

        input_tokens = random.randint(50, 2000)
        output_tokens = random.randint(100, 4000)

        # コスト概算
        cost_per_1k_input = {"gpt-4o": 0.0025, "gpt-4o-mini": 0.00015,
                             "claude-sonnet-4-20250514": 0.003, "claude-haiku-4-5-20251001": 0.0008,
                             "gemini-2.0-flash": 0.0001}
        cost_per_1k_output = {"gpt-4o": 0.01, "gpt-4o-mini": 0.0006,
                              "claude-sonnet-4-20250514": 0.015, "claude-haiku-4-5-20251001": 0.004,
                              "gemini-2.0-flash": 0.0004}
        cost = (input_tokens / 1000 * cost_per_1k_input.get(model, 0.001)
                + output_tokens / 1000 * cost_per_1k_output.get(model, 0.002))

        is_cached = random.random() < 0.15  # 15%の確率でキャッシュヒット
        is_mock = not is_cached and random.random() < 0.05  # 5%の確率でモック

        # provider別のリクエスト形式
        if provider_name == "gemini":
            request_body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7},
            }
            response_body = {
                "candidates": [{"content": {"parts": [{"text": response}]}}],
            }
        elif provider_name == "anthropic":
            request_body = {
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            response_body = {
                "content": [{"type": "text", "text": response}],
                "model": model,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        else:  # openai
            request_body = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            response_body = {
                "choices": [{"message": {"role": "assistant", "content": response}}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            }

        tags = []
        memo = ""
        if is_mock:
            tags = ["cost_limited"]
            memo = "Daily limit exceeded"
            cost = 0.0

        record = RequestRecord(
            session_id=session.id,
            step_id=i + 1,
            timestamp=ts,
            provider=provider_name,
            model=model,
            prompt_hash=str(uuid.uuid4()),
            request_body=request_body,
            response_body=response_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0 if is_cached else cost,
            is_cached=is_cached,
            tags=tags,
            memo=memo,
        )
        storage.save_request(record)
        storage.update_session_stats(session.id, record.cost_usd)
        record_count += 1

    print(f"\n✅ {record_count} records inserted into {db_path}")
    print("   Restart Web UI and check http://127.0.0.1:8765")


if __name__ == "__main__":
    main()