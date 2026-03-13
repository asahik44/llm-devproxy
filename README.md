# llm-devproxy

> LLM development debug layer — every API call recorded, nothing lost.

LLMアプリ開発中の「あるある」をすべて解決するローカルデバッグレイヤーです。

- **API呼び出しを全量自動記録** — 保存し忘れはありえない
- **キャッシュで無駄なAPI代ゼロ** — 同じリクエストはDBから返す
- **コスト爆発を防ぐ** — 上限設定でmockレスポンスを返す
- **Gitのように巻き戻せる** — 「あのステップ3からやり直したい」が即できる

---

## インストール

```bash
pip install llm-devproxy                  # 最小構成
pip install "llm-devproxy[openai]"        # OpenAI対応
pip install "llm-devproxy[anthropic]"     # Anthropic対応
pip install "llm-devproxy[gemini]"        # Gemini対応
pip install "llm-devproxy[proxy]"         # プロキシサーバー対応
pip install "llm-devproxy[all]"           # 全部入り
```

---

## 使い方 — ライブラリ編

### OpenAI

```python
import openai
from llm_devproxy import DevProxy

proxy = DevProxy(daily_limit_usd=1.0)
proxy.start_session("my_agent")

# wrap_openai() の1行だけ追加
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

## 使い方 — プロキシサーバー編

```bash
# サーバーを起動
llm-devproxy start --port 8080 --limit 1.0
```

アプリ側は `base_url` を変えるだけ：

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

## CLI コマンド一覧

```bash
# セッション一覧
llm-devproxy history

# セッションの中身を見る
llm-devproxy show my_agent

# キーワード検索（「あのとき化合物について聞いたやつ」）
llm-devproxy search "化合物"

# step 3に巻き戻す（元の履歴は消えない）
llm-devproxy rewind my_agent --step 3

# 新しいブランチとして別の試みを記録
llm-devproxy rewind my_agent --step 3 --branch new_idea

# コスト確認
llm-devproxy stats

# タグ・メモを追加
llm-devproxy tag-cmd <request_id> "重要"
llm-devproxy memo-cmd <request_id> "このプロンプトが一番うまくいった"

# 古いレスポンス本文を圧縮（メタデータは保持）
llm-devproxy compress
```

---

## タイムトラベルのユースケース

### エージェントの途中からやり直す

```python
proxy = DevProxy()

# 昨日の実行のstep 8に巻き戻す
proxy.rewind("my_agent", step=8)

# プロンプトを改善して再実行 → 新ブランチとして自動記録
client = proxy.wrap_openai(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "改良したプロンプト"}]
)
```

### 後日思いついたアイデアを試す

```bash
# 3日前のセッションを検索
llm-devproxy search "アプローチA"
# → session=my_agent, step=5 がヒット

# そこに戻って新しい試みを始める
llm-devproxy rewind my_agent --step 5 --branch "new_approach"
```

### CI/CDでAPI代をゼロにする

```python
# 同じリクエストはSQLiteキャッシュから返す
# GitHub Actionsでも毎回課金されない
proxy = DevProxy(cache_enabled=True)
```

---

## DevProxy コンフィグ

```python
proxy = DevProxy(
    db_path=".llm_devproxy.db",  # SQLiteのパス（デフォルト）
    daily_limit_usd=1.0,          # 1日のコスト上限
    session_limit_usd=None,       # セッション上限（任意）
    on_exceed="mock",             # "mock" or "block"
    cache_enabled=True,           # キャッシュON/OFF
    compress_after_days=30,       # 古いレスポンスを圧縮する日数
)
```

---

## データはすべてローカル

すべてのデータはローカルの `.llm_devproxy.db`（SQLite）に保存されます。
外部サーバーへの送信は一切ありません。

---

## ロードマップ

- [x] Phase 1: キャッシュ・コストガード・全量自動記録
- [x] Phase 2: プロキシサーバー（OpenAI/Anthropic/Gemini互換）・CLI
- [x] Phase 3: Rewind・ブランチ・タグ・メモ
- [ ] Phase 4: セマンティックキャッシュ（意味的に近いクエリもヒット）
- [ ] Phase 5: Web UI（履歴ブラウザ・コスト可視化）
- [ ] Phase 6: チーム共有機能（クラウド版）

---

## ライセンス

MIT
