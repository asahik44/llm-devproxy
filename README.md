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
```

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
- [ ] Phase 4: Semantic cache
- [ ] Phase 5: Web UI (history browser, cost dashboard)
- [ ] Phase 6: Team sharing (cloud edition)

---

<details>
<summary>日本語版 README</summary>

## llm-devproxy（日本語）

LLMアプリ開発中の「あるある」をすべて解決するローカルデバッグレイヤーです。

- **API呼び出しを全量自動記録** — 保存し忘れはありえない
- **キャッシュで無駄なAPI代ゼロ** — 同じリクエストはDBから返す
- **コスト爆発を防ぐ** — 上限設定でmockレスポンスを返す
- **Gitのように巻き戻せる** — 「あのステップ3からやり直したい」が即できる

詳しい使い方は英語版をご覧ください（内容は同じです）。

</details>

---

## License

MIT