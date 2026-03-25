"""
Semantic Cache Smoke Test
=========================
セマンティックキャッシュの動作確認スクリプト。
実際のLLM APIは呼ばず、Storage + CacheManager を直接使って検証する。

使い方:
    cd /Users/hiroki/python_pj/llm-dev-proxy
    pip install -e ".[semantic-local]"   # 初回のみ
    python smoke_test_semantic.py
"""

import sys
import tempfile
import time
from pathlib import Path

# ── カラー出力ヘルパー ────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(text: str):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{RESET}\n")


def step(num: int, text: str):
    print(f"{BOLD}[Step {num}]{RESET} {text}")


def ok(text: str):
    print(f"  {GREEN}✅ {text}{RESET}")


def fail(text: str):
    print(f"  {RED}❌ {text}{RESET}")


def info(text: str):
    print(f"  {YELLOW}ℹ️  {text}{RESET}")


# ── メイン ────────────────────────────────────────────────────

def main():
    header("Semantic Cache Smoke Test")

    # ---- 依存チェック ----
    step(0, "依存パッケージの確認...")
    try:
        import numpy as np
        ok(f"numpy {np.__version__}")
    except ImportError:
        fail("numpy が見つかりません。pip install numpy")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
        ok("sentence-transformers OK")
    except ImportError:
        fail("sentence-transformers が見つかりません。")
        info("pip install -e \".[semantic-local]\"  を実行してください")
        sys.exit(1)

    try:
        from llm_devproxy.core.models import ProxyConfig, RequestRecord
        from llm_devproxy.core.storage import Storage
        from llm_devproxy.core.cache import CacheManager
        from llm_devproxy.core.semantic_cache import (
            SemanticCacheManager, normalize_prompt, cosine_similarity,
        )
        ok("llm_devproxy パッケージ OK")
    except ImportError as e:
        fail(f"llm_devproxy のインポートに失敗: {e}")
        info("pip install -e . を実行してください")
        sys.exit(1)

    # ---- セットアップ（一時DB） ----
    step(1, "一時DBでStorage・CacheManagerを初期化...")
    tmp_dir = tempfile.mkdtemp()
    db_path = str(Path(tmp_dir) / "test_semantic.db")

    config = ProxyConfig(
        db_path=db_path,
        cache_enabled=True,
        cache_mode="both",           # exact → semantic のフォールバック
        semantic_cache=True,
        semantic_backend="local",    # sentence-transformers
        similarity_threshold=0.85,
    )

    storage = Storage(db_path)
    cache = CacheManager(storage, config, enabled=True)

    if cache.semantic is None:
        fail("SemanticCacheManager の初期化に失敗")
        sys.exit(1)
    ok(f"初期化完了 (DB: {db_path})")

    # ---- テストデータ定義 ----
    PROVIDER = "openai"
    MODEL = "gpt-4o-mini"

    original_request = {
        "messages": [
            {"role": "user", "content": "Pythonでフィボナッチ数列を計算する方法を教えて"}
        ],
        "temperature": 0.7,
    }

    similar_request = {
        "messages": [
            {"role": "user", "content": "Pythonでフィボナッチ数を求めるコードを書いて"}
        ],
        "temperature": 0.7,
    }

    different_request = {
        "messages": [
            {"role": "user", "content": "東京の天気を教えて"}
        ],
        "temperature": 0.7,
    }

    exact_same_request = {
        "messages": [
            {"role": "user", "content": "Pythonでフィボナッチ数列を計算する方法を教えて"}
        ],
        "temperature": 0.7,
    }

    # ---- Step 2: normalize_prompt の動作確認 ----
    step(2, "normalize_prompt の動作確認...")
    normalized = normalize_prompt(original_request)
    print(f"  入力: {original_request['messages'][0]['content']}")
    print(f"  正規化: {normalized}")
    if "[user]" in normalized:
        ok("正規化OK")
    else:
        fail("正規化に [user] プレフィックスがない")

    # ---- Step 3: embeddingの生成テスト ----
    step(3, "Embedding生成テスト (初回はモデルDLで時間かかります)...")
    t0 = time.time()
    backend = cache.semantic.backend
    vec = backend.embed("テスト文章です")
    elapsed = time.time() - t0
    ok(f"次元数: {len(vec)}, 生成時間: {elapsed:.2f}s")

    # ---- Step 4: ダミーレコード保存 + embedding保存 ----
    step(4, "ダミーレコード + embedding を保存...")
    import uuid
    from datetime import datetime, timezone

    dummy_record = RequestRecord(
        id=str(uuid.uuid4()),
        session_id="test-session",
        step_id=1,
        provider=PROVIDER,
        model=MODEL,
        prompt_hash=cache.make_hash(PROVIDER, MODEL, original_request),
        request_body=original_request,
        response_body={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "フィボナッチ数列はこう書きます: def fib(n): ..."
                }
            }]
        },
        input_tokens=50,
        output_tokens=100,
        cost_usd=0.001,
        is_cached=False,
    )
    storage.save_request(dummy_record)
    cache.store_semantic(dummy_record, original_request)
    ok("保存完了")

    # ---- Step 5: 完全一致キャッシュテスト ----
    step(5, "完全一致キャッシュテスト（同一リクエスト）...")
    hit = cache.get(PROVIDER, MODEL, exact_same_request)
    if hit:
        ok(f"完全一致ヒット! record_id={hit.id[:8]}...")
    else:
        fail("完全一致でヒットするはずがミス")

    # ---- Step 6: セマンティックキャッシュテスト（類似） ----
    step(6, "セマンティックキャッシュテスト（類似プロンプト）...")
    print(f"  元: {original_request['messages'][0]['content']}")
    print(f"  類似: {similar_request['messages'][0]['content']}")

    # 完全一致はしないはず → semanticにフォールバック
    prompt_hash_similar = cache.make_hash(PROVIDER, MODEL, similar_request)
    exact_hit = storage.find_cached(prompt_hash_similar, MODEL)
    if exact_hit:
        info("（完全一致でもヒット — hashが衝突？）")
    else:
        info("完全一致ミス → セマンティック検索にフォールバック")

    result = cache.semantic.find_similar(PROVIDER, MODEL, similar_request)
    if result:
        record, similarity = result
        ok(f"セマンティックヒット! 類似度={similarity:.4f} (閾値={config.similarity_threshold})")
        if similarity >= config.similarity_threshold:
            ok("閾値クリア → キャッシュとして使える")
        else:
            info(f"閾値未満 → キャッシュミス扱い")
    else:
        fail("セマンティックでもヒットせず")
        info("閾値を下げるか、モデルを変えてみてください")

    # cache.get() 経由でも確認
    hit_via_cache = cache.get(PROVIDER, MODEL, similar_request)
    if hit_via_cache:
        ok("cache.get() 経由でもセマンティックヒット確認")
    else:
        info("cache.get() 経由ではミス（閾値の問題かも）")

    # ---- Step 7: 非類似プロンプトテスト ----
    step(7, "非類似プロンプトテスト（全く違う内容）...")
    print(f"  元: {original_request['messages'][0]['content']}")
    print(f"  非類似: {different_request['messages'][0]['content']}")

    result_diff = cache.semantic.find_similar(PROVIDER, MODEL, different_request)
    if result_diff:
        _, sim = result_diff
        if sim < config.similarity_threshold:
            ok(f"類似度={sim:.4f} < 閾値 → 正しくミス")
        else:
            fail(f"類似度={sim:.4f} >= 閾値 → 誤ヒット！閾値の調整が必要")
    else:
        ok("セマンティックミス（期待通り）")

    hit_diff = cache.get(PROVIDER, MODEL, different_request)
    if hit_diff is None:
        ok("cache.get() でもミス確認（期待通り）")
    else:
        fail("cache.get() でヒットしてしまった")

    # ---- Step 8: cosine_similarity の直接テスト ----
    step(8, "コサイン類似度の直接計算テスト...")
    text_a = normalize_prompt(original_request)
    text_b = normalize_prompt(similar_request)
    text_c = normalize_prompt(different_request)

    vec_a = backend.embed(text_a)
    vec_b = backend.embed(text_b)
    vec_c = backend.embed(text_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)
    sim_bc = cosine_similarity(vec_b, vec_c)

    print(f"  フィボナッチ(元) vs フィボナッチ(類似): {sim_ab:.4f}")
    print(f"  フィボナッチ(元) vs 天気:               {sim_ac:.4f}")
    print(f"  フィボナッチ(類似) vs 天気:             {sim_bc:.4f}")

    if sim_ab > sim_ac and sim_ab > sim_bc:
        ok("類似ペアが最も高い類似度 → 期待通り")
    else:
        fail("類似度の順序がおかしい")

    # ---- 結果サマリー ----
    header("テスト完了!")
    print(f"  DB: {db_path}")
    print(f"  このDBは一時ファイルなので、不要なら削除してください。")
    print()
    print(f"  {BOLD}次のステップ:{RESET}")
    print(f"  実際に使うには ProxyConfig に以下を追加:")
    print()
    print(f"    config = ProxyConfig(")
    print(f"        semantic_cache=True,")
    print(f"        semantic_backend=\"local\",   # or \"openai\"")
    print(f"        cache_mode=\"both\",           # exact → semantic fallback")
    print(f"        similarity_threshold=0.85,")
    print(f"    )")
    print()


if __name__ == "__main__":
    main()