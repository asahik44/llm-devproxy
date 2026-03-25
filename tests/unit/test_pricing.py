"""
PricingManager tests — 3-tier pricing resolution.
"""

import json
import pytest
from pathlib import Path
from llm_devproxy.core.pricing import PricingManager
from llm_devproxy.core.cost_guard import BUILTIN_PRICING, estimate_cost, is_reasoning_model, set_pricing_manager


@pytest.fixture(autouse=True)
def reset_pricing_singleton():
    """Reset the global singleton before each test."""
    set_pricing_manager(None)
    yield
    set_pricing_manager(None)


def test_builtin_fallback():
    """リモートもローカルもない場合、ビルトインが使われること"""
    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=Path("/nonexistent/path/pricing.json"),
    )
    pricing = pm.get("o3")
    assert pricing["input"] == 0.010
    assert pricing["output"] == 0.040
    assert pricing["reasoning"] == 0.040
    assert pm.get_source() == "builtin"


def test_unknown_model_returns_default():
    """不明モデルはデフォルト値が返ること"""
    pm = PricingManager(builtin=BUILTIN_PRICING, enable_remote=False)
    pricing = pm.get("totally-unknown-model-xyz")
    assert pricing["input"] == 0.0025  # gpt-4o default
    assert pricing["output"] == 0.010


def test_local_override(tmp_path):
    """ローカルファイルがビルトインを上書きすること"""
    local_file = tmp_path / "pricing.json"
    local_file.write_text(json.dumps({
        "models": {
            "o3": {"input": 0.999, "output": 0.888, "reasoning": 0.777},
            "my-custom-model": {"input": 0.001, "output": 0.002, "reasoning": None},
        }
    }))

    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=local_file,
    )

    # o3 should be overridden
    pricing = pm.get("o3")
    assert pricing["input"] == 0.999
    assert pricing["reasoning"] == 0.777

    # Custom model should be available
    custom = pm.get("my-custom-model")
    assert custom["input"] == 0.001

    # Other builtin models should still work
    gpt4o = pm.get("gpt-4o")
    assert gpt4o["input"] == 0.0025

    assert pm.get_source() == "local"


def test_local_override_flat_format(tmp_path):
    """ローカルファイルが 'models' キーなしのフラット形式でも動くこと"""
    local_file = tmp_path / "pricing.json"
    local_file.write_text(json.dumps({
        "o3": {"input": 0.555, "output": 0.444, "reasoning": 0.333},
    }))

    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=local_file,
    )

    pricing = pm.get("o3")
    assert pricing["input"] == 0.555


def test_local_invalid_json(tmp_path):
    """ローカルファイルが壊れていてもクラッシュしないこと"""
    local_file = tmp_path / "pricing.json"
    local_file.write_text("{ invalid json !!!")

    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=local_file,
    )

    # Should fall through to builtin
    pricing = pm.get("o3")
    assert pricing["input"] == 0.010
    assert pm.get_source() == "builtin"


def test_local_empty_file(tmp_path):
    """ローカルファイルが空でもクラッシュしないこと"""
    local_file = tmp_path / "pricing.json"
    local_file.write_text("{}")

    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=local_file,
    )

    pricing = pm.get("o3")
    assert pricing["input"] == 0.010
    assert pm.get_source() == "builtin"


def test_get_all():
    """get_all()が全モデルを返すこと"""
    pm = PricingManager(builtin=BUILTIN_PRICING, enable_remote=False)
    all_pricing = pm.get_all()
    assert "o3" in all_pricing
    assert "gpt-4o" in all_pricing
    assert "claude-sonnet-4-6" in all_pricing
    assert "gemini-3.1-pro-preview" in all_pricing
    assert len(all_pricing) >= 20


def test_reload(tmp_path):
    """reload()でデータが更新されること"""
    local_file = tmp_path / "pricing.json"

    pm = PricingManager(
        builtin=BUILTIN_PRICING,
        enable_remote=False,
        local_path=local_file,
    )

    # First load: no local file → builtin
    assert pm.get("o3")["input"] == 0.010
    assert pm.get_source() == "builtin"

    # Write local file
    local_file.write_text(json.dumps({
        "models": {"o3": {"input": 0.999, "output": 0.888, "reasoning": 0.777}}
    }))

    # Reload
    pm.reload()
    assert pm.get("o3")["input"] == 0.999
    assert pm.get_source() == "local"


def test_estimate_cost_uses_pricing_manager():
    """estimate_cost() がPricingManager経由で料金を解決すること"""
    cost = estimate_cost("o3", input_tokens=1000, output_tokens=500, reasoning_tokens=2000)
    expected = 1000/1000*0.010 + 500/1000*0.040 + 2000/1000*0.040
    assert cost == pytest.approx(expected, rel=1e-3)


def test_is_reasoning_model_uses_pricing_manager():
    """is_reasoning_model() がPricingManager経由で判定すること"""
    assert is_reasoning_model("o3") is True
    assert is_reasoning_model("gpt-4o") is False
    assert is_reasoning_model("claude-sonnet-4-6") is True
    assert is_reasoning_model("gemini-3.1-pro-preview") is True
