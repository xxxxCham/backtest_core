from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import apply_dataset_quality_stage, load_ohlcv_file


def _make_ohlcv(n: int = 1000) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = np.linspace(100.0, 120.0, n)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": np.full(n, 10.0),
        },
        index=index,
    )


def test_apply_dataset_quality_stage_trims_launch_prefix_and_attaches_report() -> None:
    df = _make_ohlcv(1000)

    filtered, report = apply_dataset_quality_stage(df, "1h", symbol="TESTUSDC", enforce_quality=True)

    assert report.is_valid is True
    assert report.launch_trim_bars == 24
    assert len(filtered) == 976
    assert filtered.index[0] == df.index[24]
    assert filtered.attrs["dataset_quality"]["launch_trim_bars"] == 24


def test_apply_dataset_quality_stage_flags_large_gap() -> None:
    df = _make_ohlcv(1000)
    df = df.drop(df.index[300:560])

    filtered, report = apply_dataset_quality_stage(
        df,
        "1h",
        symbol="BROKENUSDC",
        enforce_quality=False,
    )

    assert len(filtered) > 0
    assert report.is_valid is False
    assert any(reason.startswith("coverage<") or reason.startswith("largest_gap>") for reason in report.blocking_reasons)

    with pytest.raises(ValueError):
        apply_dataset_quality_stage(df, "1h", symbol="BROKENUSDC", enforce_quality=True)


def test_load_ohlcv_file_runs_quality_stage(tmp_path) -> None:
    path = tmp_path / "FOOUSDC_1h.parquet"
    _make_ohlcv(800).to_parquet(path)

    df, report = load_ohlcv_file(path)

    assert report.is_valid is True
    assert report.symbol == "FOOUSDC"
    assert report.timeframe == "1h"
    assert "dataset_quality" in df.attrs
