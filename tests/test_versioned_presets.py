"""
Module-ID: tests.test_versioned_presets

Purpose: Tester système versioning presets (roundtrip save/load, resolve latest, validation).

Role in pipeline: testing

Key components: test_resolve_latest_version, test_versioned_preset_roundtrip, fixtures monkeypatch

Inputs: Versioned preset params dict, tmp_path, env var BACKTEST_PRESETS_DIR

Outputs: Loaded preset dict matched saved, version resolved

Dependencies: pytest, utils.parameters

Conventions: DEFAULT_STRATEGY_VERSION constant; roundtrip fidelité; validation avant use.

Read-if: Modification versioning system ou file structure.

Skip-if: Tests presets non critiques.
"""

import pytest

from utils.parameters import (
    DEFAULT_STRATEGY_VERSION,
    list_strategy_versions,
    load_strategy_version,
    resolve_latest_version,
    save_versioned_preset,
    validate_before_use,
)


def test_resolve_latest_version_default(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_PRESETS_DIR", str(tmp_path))
    assert resolve_latest_version("bollinger_atr") == DEFAULT_STRATEGY_VERSION


def test_versioned_preset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_PRESETS_DIR", str(tmp_path))

    params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "entry_z": 2.0,
        "atr_period": 14,
        "atr_percentile": 30,
        "k_sl": 1.5,
        "leverage": 2,
    }
    metrics = {
        "sharpe_ratio": 1.25,
        "total_return_pct": 12.5,
        "max_drawdown": 8.0,
        "win_rate": 55.0,
    }

    preset = save_versioned_preset(
        strategy_name="bollinger_atr",
        version="0.0.1",
        preset_name="winner",
        params_values=params,
        metrics=metrics,
    )

    presets = list_strategy_versions("bollinger_atr")
    assert len(presets) == 1
    assert preset.name.startswith("bollinger_atr@0.0.1__winner")

    loaded = load_strategy_version(
        strategy_name="bollinger_atr",
        version="0.0.1",
        preset_name=preset.name,
    )

    loaded_values = loaded.get_default_values()
    assert loaded.metadata.get("version") == "0.0.1"
    assert loaded_values["bb_period"] == 20
    assert loaded_values["bb_std"] == 2.0


def test_validate_before_use_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_PRESETS_DIR", str(tmp_path))

    params = {"bb_period": 20, "bb_std": 2.0}
    preset = save_versioned_preset(
        strategy_name="bollinger_atr",
        version="0.0.1",
        preset_name="winner",
        params_values=params,
    )

    with pytest.raises(ValueError):
        validate_before_use(preset, "ema_cross")
