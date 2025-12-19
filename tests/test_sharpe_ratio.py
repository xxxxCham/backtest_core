import numpy as np
import pandas as pd
import pytest

from backtest.performance import sharpe_ratio


def test_sharpe_ratio_standard_expected_value():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    expected = (returns.mean() * np.sqrt(252)) / returns.std(ddof=1)

    assert sharpe_ratio(returns, periods_per_year=252, method="standard") == pytest.approx(expected)


def test_sharpe_ratio_no_trades_returns_zero():
    returns = pd.Series([0.0, 0.0, 0.0])

    assert sharpe_ratio(returns, periods_per_year=252, method="standard") == 0.0


def test_sharpe_ratio_daily_resample_short_series():
    equity = pd.Series(
        [10000.0, 10100.0, 10200.0, 10150.0],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    returns = equity.pct_change().fillna(0)

    expected_returns = equity.resample("D").last().pct_change().dropna()
    expected = (expected_returns.mean() * np.sqrt(252)) / expected_returns.std(ddof=1)

    result = sharpe_ratio(
        returns,
        periods_per_year=252,
        method="daily_resample",
        equity=equity,
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_min_vol_guard():
    tiny_returns = pd.Series([1e-8] * 120)

    assert sharpe_ratio(tiny_returns, periods_per_year=252, method="standard") == 0.0


def test_sharpe_ratio_single_trade_returns_zero():
    equity = pd.Series(
        [10000.0, 10000.0, 10500.0, 10500.0],
        index=pd.date_range("2024-02-01", periods=4, freq="D"),
    )
    returns = equity.pct_change().fillna(0)

    result = sharpe_ratio(
        returns,
        periods_per_year=252,
        method="daily_resample",
        equity=equity,
    )

    assert result == 0.0


def test_sharpe_ratio_three_trades_is_computed():
    daily_changes = pd.Series([0.02, -0.01, 0.015], index=pd.date_range("2024-03-01", periods=3, freq="D"))
    equity = pd.Series(
        [10000.0] + list((10000 * (1 + daily_changes).cumprod())),
        index=pd.date_range("2024-03-01", periods=4, freq="D"),
    )
    returns = equity.pct_change().fillna(0)

    expected_returns = equity.resample("D").last().pct_change().dropna()
    expected = (expected_returns.mean() * np.sqrt(252)) / expected_returns.std(ddof=1)

    result = sharpe_ratio(
        returns,
        periods_per_year=252,
        method="daily_resample",
        equity=equity,
    )

    assert result == pytest.approx(expected)
    assert np.isfinite(result)


def test_sharpe_ratio_near_zero_volatility():
    tiny_returns = pd.Series([1e-5] * 200)
    tiny_returns.iloc[::40] = 1.2e-5

    assert sharpe_ratio(tiny_returns, periods_per_year=252, method="standard") == 0.0
