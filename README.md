# llm-devproxy

[![PyPI version](https://badge.fury.io/py/llm-devproxy.svg)](https://pypi.org/project/llm-devproxy/)
[![Tests](https://github.com/asahik44/llm-devproxy/actions/workflows/test.yml/badge.svg)](https://github.com/asahik44/llm-devproxy/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


> LLM development debug layer — every API call recorded, nothing lost.

A local debug layer that solves the common pain points of LLM app development.

- **Auto-records every API call** — nothing is ever lost
- **Cache eliminates redundant costs** — same requests return from DB
- **Prevents cost explosions** — mock responses when limit is reached
- **Rewind like Git** — "go back to step 3 and try again" in seconds
- **🧠 Reasoning token visibility** — o1/o3, Claude thinking, Gemini 2.5 thinking tracked and visualized
- **🔔 Smart alerts** — cost warnings, high-spend single requests, reasoning ratio alerts
- **📥 Export** — CSV/JSON export via CLI and Web UI
- **🌊 Streaming support** — `stream=True` works transparently, recorded after completion

---

## Install
```bash
pip install llm-devproxy                  # minimal
pip install "llm-devproxy[openai]"        # with OpenAI
pip install "llm-devproxy[anthropic]"     # with Anthropic
pip install "llm-devproxy[gemini]"        # with Gemini
pip install "llm-devproxy[proxy]"         # with proxy server
pip install "llm-devproxy[all]"           # everything
```

---

## Usage — Library

### OpenAI
```python
import openai
from llm_devproxy import DevProxy

proxy = DevProxy(daily_limit_usd=1.0)
proxy.start_session("my_agent")

# Just wrap your existing client
client = proxy.wrap_openai(openai.OpenAI(api_key="sk-..."))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### Anthropic
```python
import anthropic
from llm_devproxy import DevProxy

proxy = DevProxy(daily_limit_usd=1.0)
proxy.start_session("my_agent")

client = proxy.wrap_anthropic(anthropic.Anthropic(api_key="sk-ant-..."))

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.text)
```

### Gemini
```python
import google.generativeai as genai
from llm_devproxy import DevProxy

genai.configure(api_key="AI...")
proxy = DevProxy(daily_limit_usd=1.0)
proxy.start_session("my_agent")

model = proxy.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))
response = model.generate_content("Hello")
print(response.text)
```

---

## Usage — Proxy Server
```bash
llm-devproxy start --port 8080 --limit 1.0
```

Just change `base_url` in your app — nothing else:
```python
# OpenAI
client = openai.OpenAI(
    api_key="sk-...",
    base_url="http://localhost:8080/openai/v1",
)

# Anthropic
client = anthropic.Anthropic(
    api_key="sk-ant-...",
    base_url="http://localhost:8080/anthropic/v1",
)
```

---

## CLI
```bash
# List recent sessions
llm-devproxy history

# Show all steps in a session
llm-devproxy show my_agent

# Search through recorded prompts
llm-devproxy search "keyword"

# Rewind to step 3 (original history preserved)
llm-devproxy rewind my_agent --step 3

# Rewind and start a new branch
llm-devproxy rewind my_agent --step 3 --branch new_idea

# Show cost stats
llm-devproxy stats

# Export to CSV/JSON (v0.3.0)
llm-devproxy export -f csv -o requests.csv
llm-devproxy export -f json --provider openai --model o1

# Launch web dashboard (v0.3.0)
llm-devproxy web --port 8765
```

---

## Streaming (v0.3.0)

`stream=True` just works — chunks are passed through transparently, then recorded after the stream completes.

```python
# OpenAI streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Anthropic streaming
stream = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for event in stream:
    # events pass through as-is
    pass

# All streamed responses are automatically recorded with full token counts
```

---

## Reasoning Token Tracking (v0.3.0)

Reasoning tokens from o1/o3, Claude extended thinking, and Gemini 2.5 thinking are automatically tracked, costed, and visualized.

```python
# Terminal output when reasoning tokens are used:
# 🧠 Reasoning tokens: 2,400 (83% of output) | Output: 500 | Cost: $0.034500

# Access reasoning stats
stats = proxy.engine.reasoning_stats()
# {'total_reasoning': 2400, 'total_output': 500, 'reasoning_pct': 82.8, ...}
```

The Web UI shows reasoning tokens with visual bars on every page — history, detail, costs, and session comparison.

---

## Time Travel Use Cases

### Resume an agent from the middle
```python
proxy = DevProxy()

# Rewind yesterday's run to step 8
proxy.rewind("my_agent", step=8)

# Tweak the prompt and re-run → recorded as a new branch
client = proxy.wrap_openai(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Improved prompt"}]
)
```

### Find something from days ago
```bash
llm-devproxy search "approach A"
# → session=my_agent, step=5

llm-devproxy rewind my_agent --step 5 --branch "revisit"
```

### Zero API cost in CI/CD
```python
# Same requests return from SQLite cache
# No API charges in GitHub Actions
proxy = DevProxy(cache_enabled=True)
```

---

## Config
```python
proxy = DevProxy(
    db_path=".llm_devproxy.db",  # SQLite path
    daily_limit_usd=1.0,          # daily cost limit
    session_limit_usd=None,       # per-session limit (optional)
    on_exceed="mock",             # "mock" or "block"
    cache_enabled=True,
    compress_after_days=30,

    # Alert settings (v0.3.0)
    alert_daily_threshold=0.8,    # warn at 80% of daily limit
    alert_session_threshold=0.8,  # warn at 80% of session limit
    alert_reasoning_ratio=0.7,    # warn if reasoning > 70% of output
    alert_single_cost_usd=0.10,   # warn if single request > $0.10
)
```

---

## All data stays local

Everything is stored in `.llm_devproxy.db` (SQLite) on your machine.
Nothing is sent to any external server.

---

## Roadmap

- [x] Phase 1: Cache, cost guard, auto-record everything
- [x] Phase 2: Proxy server (OpenAI/Anthropic/Gemini compatible), CLI
- [x] Phase 3: Rewind, branches, tags, memos
- [x] Phase 4: Semantic cache
- [x] Phase 5: Web UI (history browser, cost dashboard, session comparison)
- [x] Phase 6: Reasoning token tracking, alerts, CSV/JSON export, streaming
- [ ] Phase 7: Team sharing (cloud edition)

---

<details>
<summary>日本語版 README</summary>

## llm-devproxy（日本語）

LLMアプリ開発中の「あるある」をすべて解決するローカルデバッグレイヤーです。

- **API呼び出しを全量自動記録** — 保存し忘れはありえない
- **キャッシュで無駄なAPI代ゼロ** — 同じリクエストはDBから返す
- **コスト爆発を防ぐ** — 上限設定でmockレスポンスを返す
- **Gitのように巻き戻せる** — 「あのステップ3からやり直したい」が即できる
- **🧠 推論トークン可視化** — o1/o3, Claude thinking, Gemini 2.5 thinkingを追跡・可視化
- **🔔 スマートアラート** — コスト警告、高額リクエスト、推論トークン比率アラート
- **📥 エクスポート** — CSV/JSONエクスポート（CLI・Web UI両対応）
- **🌊 Streaming対応** — `stream=True` が透過的に動作、完了後に自動記録

詳しい使い方は英語版をご覧ください（内容は同じです）。

</details>

---

## License

MIT