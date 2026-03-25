"""
Pricing Manager — 3-tier model pricing resolution.

Priority: Local override > Remote JSON > Hardcoded fallback.

- Local:  ~/.llm_devproxy/pricing.json (user-editable)
- Remote: GitHub raw URL (auto-fetched, cached for 24h)
- Fallback: BUILTIN_PRICING in cost_guard.py
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

# Remote pricing URL (GitHub raw)
REMOTE_PRICING_URL = (
    "https://raw.githubusercontent.com/asahik44/llm-devproxy/main/pricing.json"
)

# Local override path
LOCAL_PRICING_DIR = Path.home() / ".llm_devproxy"
LOCAL_PRICING_PATH = LOCAL_PRICING_DIR / "pricing.json"

# Cache: fetched remote data is saved here to avoid repeated HTTP calls
CACHE_PATH = LOCAL_PRICING_DIR / ".pricing_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class PricingManager:
    """
    3階層で料金データを解決する。

    Usage:
        pm = PricingManager(builtin=BUILTIN_PRICING)
        pricing = pm.get("o3")
        # → {"input": 0.010, "output": 0.040, "reasoning": 0.040}
    """

    def __init__(
        self,
        builtin: dict[str, dict],
        remote_url: str = REMOTE_PRICING_URL,
        local_path: Optional[Path] = None,
        enable_remote: bool = True,
    ):
        self._builtin = builtin
        self._remote_url = remote_url
        self._local_path = local_path or LOCAL_PRICING_PATH
        self._enable_remote = enable_remote

        # Merged pricing dict (built once, refreshed on demand)
        self._merged: dict[str, dict] = {}
        self._loaded = False
        self._load_source: str = "builtin"  # for debugging

    def get(self, model: str) -> dict:
        """
        モデルの料金を取得。見つからない場合はgpt-4o相当のデフォルト。
        初回呼び出し時に3階層をマージする（lazy init）。
        """
        if not self._loaded:
            self._load_all()

        pricing = self._merged.get(model)
        if pricing:
            return pricing

        # 不明モデル → デフォルト
        return {"input": 0.0025, "output": 0.010, "reasoning": None}

    def get_all(self) -> dict[str, dict]:
        """全モデルのマージ済み料金を返す。"""
        if not self._loaded:
            self._load_all()
        return dict(self._merged)

    def get_source(self) -> str:
        """デバッグ用：現在の最優先ソース名。"""
        if not self._loaded:
            self._load_all()
        return self._load_source

    def reload(self):
        """料金データを再ロード。キャッシュもクリア。"""
        self._loaded = False
        self._merged = {}
        self._load_all()

    # ── Internal ──────────────────────────────────────────────

    def _load_all(self):
        """3階層をマージ。後から読んだものが上書き。"""
        # Layer 1: builtin (always available)
        self._merged = dict(self._builtin)
        self._load_source = "builtin"

        # Layer 2: remote (best-effort)
        if self._enable_remote:
            remote = self._load_remote()
            if remote:
                self._merged.update(remote)
                self._load_source = "remote"

        # Layer 3: local override (highest priority)
        local = self._load_local()
        if local:
            self._merged.update(local)
            self._load_source = "local"

        self._loaded = True

    def _load_local(self) -> Optional[dict[str, dict]]:
        """ローカル上書きファイルを読む。"""
        try:
            if self._local_path.exists():
                data = json.loads(self._local_path.read_text(encoding="utf-8"))
                models = data.get("models", data)  # "models" キーがあればそれを使用
                if isinstance(models, dict) and models:
                    return models
        except Exception as e:
            print(f"⚠️  Local pricing file error ({self._local_path}): {e}")
        return None

    def _load_remote(self) -> Optional[dict[str, dict]]:
        """リモートJSONを取得。キャッシュがTTL内なら再利用。"""
        # Check cache first
        cached = self._load_cache()
        if cached is not None:
            return cached

        # Fetch from remote
        try:
            import urllib.request
            req = urllib.request.Request(
                self._remote_url,
                headers={"User-Agent": "llm-devproxy"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                models = data.get("models", {})
                if models:
                    self._save_cache(models)
                    return models
        except Exception:
            # Network failure is expected (offline, corporate proxy, etc.)
            # Fall through silently to builtin
            pass

        return None

    def _load_cache(self) -> Optional[dict]:
        """キャッシュファイルがTTL内なら返す。"""
        try:
            if CACHE_PATH.exists():
                mtime = CACHE_PATH.stat().st_mtime
                if time.time() - mtime < CACHE_TTL_SECONDS:
                    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data:
                        return data
        except Exception:
            pass
        return None

    def _save_cache(self, models: dict):
        """キャッシュファイルに保存。"""
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps(models, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass  # Cache write failure is non-critical


def create_local_pricing_template():
    """
    ローカル上書き用のテンプレートファイルを生成する。
    CLI: llm-devproxy pricing init
    """
    LOCAL_PRICING_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_PRICING_PATH.exists():
        print(f"⚠️  Already exists: {LOCAL_PRICING_PATH}")
        return

    template = {
        "_comment": "ここに追加・上書きしたいモデルの料金を記載してください。USD per 1K tokens。",
        "models": {
            "my-custom-model": {
                "input": 0.001,
                "output": 0.002,
                "reasoning": None
            }
        }
    }
    LOCAL_PRICING_PATH.write_text(
        json.dumps(template, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✅ Created: {LOCAL_PRICING_PATH}")
    print(f"   Edit this file to add custom model pricing.")
