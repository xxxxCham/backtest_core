import numpy as np
import pandas as pd
import pytest

from backtest.performance import calculate_metrics


def test_calculate_metrics_annualized_return_uses_calendar_time():
    index = pd.date_range("2024-01-01", periods=1441, freq="min")
    equity = pd.Series(np.linspace(10000, 11000, len(index)), index=index)
    returns = equity.pct_change().fillna(0)

    metrics = calculate_metrics(equity, returns, pd.DataFrame(columns=["pnl"]))

    elapsed_days = (index[-1] - index[0]).total_seconds() / 86400
    years = elapsed_days / 365
    expected = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100

    assert metrics["annualized_return"] == pytest.approx(expected)


def test_volatility_annual_respects_daily_resample_on_sparse_intraday():
    daily_prices = [100.0, 102.0, 101.0]
    values: list[float] = []
    timestamps: list[pd.Timestamp] = []

    start = pd.Timestamp("2024-01-01")
    for offset, price in enumerate(daily_prices):
        day_index = pd.date_range(start + pd.Timedelta(days=offset), periods=24, freq="h")
        timestamps.extend(day_index)
        values.extend([price] * len(day_index))

    equity = pd.Series(values, index=timestamps)
    returns = equity.pct_change().fillna(0)

    metrics = calculate_metrics(
        equity,
        returns,
        pd.DataFrame(columns=["pnl"]),
        sharpe_method="daily_resample",
    )

    daily_series = pd.Series(daily_prices, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    daily_returns = daily_series.pct_change().dropna()
    expected_vol = daily_returns.std(ddof=1) * np.sqrt(252) * 100

    assert metrics["volatility_annual"] == pytest.approx(expected_vol)


def test_max_drawdown_duration_uses_timestamps():
    index = pd.date_range("2024-01-01 00:00", periods=5, freq="min")
    equity = pd.Series([100, 99, 98, 99, 100], index=index)
    returns = equity.pct_change().fillna(0)

    metrics = calculate_metrics(equity, returns, pd.DataFrame(columns=["pnl"]))

    expected_duration_days = (index[3] - index[1]).total_seconds() / 86400
    assert metrics["max_drawdown_duration_days"] == pytest.approx(expected_duration_days)
