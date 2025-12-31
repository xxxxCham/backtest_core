import pytest

from metrics_types import frac_to_pct, normalize_metrics, pct_to_frac


def test_pct_to_frac_conversion() -> None:
    payload = {
        "total_return_pct": 12.5,
        "max_drawdown_pct": -5.0,
        "win_rate_pct": 40.0,
    }
    out = pct_to_frac(payload)
    assert out["total_return"] == pytest.approx(0.125)
    assert out["max_drawdown"] == pytest.approx(-0.05)
    assert out["win_rate"] == pytest.approx(0.4)
    assert "total_return_pct" not in out


def test_frac_to_pct_conversion() -> None:
    payload = {
        "total_return": 0.2,
        "max_drawdown": -0.1,
        "win_rate": 0.55,
    }
    out = frac_to_pct(payload)
    assert out["total_return_pct"] == pytest.approx(20.0)
    assert out["max_drawdown_pct"] == pytest.approx(-10.0)
    assert out["win_rate_pct"] == pytest.approx(55.0)
    assert "total_return" not in out


def test_alias_normalization_pct() -> None:
    payload = {
        "total_return": 12.5,
        "max_drawdown": -5.0,
        "win_rate": 40.0,
    }
    out = normalize_metrics(payload, "pct")
    assert out["total_return_pct"] == 12.5
    assert out["max_drawdown_pct"] == -5.0
    assert out["win_rate_pct"] == 40.0
    assert "total_return" not in out


def test_invariant_violations() -> None:
    with pytest.raises(ValueError):
        normalize_metrics({"win_rate_pct": 120.0}, "pct")
    with pytest.raises(ValueError):
        normalize_metrics({"max_drawdown": 0.1}, "frac")
