import numpy as np
import pandas as pd

from agents.backtest_executor import BacktestExecutor, BacktestRequest
from agents.integration import run_backtest_for_agent
from backtest.engine import BacktestEngine, RunResult
from backtest.facade import UIMetrics
from backtest.storage import StoredResultMetadata
from backtest.sweep import SweepResultItem, SweepResults
from strategies.base import StrategyBase, register_strategy


@register_strategy("test_strategy_metrics")
class TestStrategy(StrategyBase):
    @property
    def required_indicators(self):
        return []

    def generate_signals(self, df, indicators, params):
        return pd.Series(0, index=df.index)


def _build_df() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    base = np.linspace(100, 109, len(index))
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )


def test_pipeline_metric_invariants() -> None:
    df = _build_df()
    engine = BacktestEngine(initial_capital=10000.0)
    result = engine.run(df=df, strategy="test_strategy_metrics", params={})

    assert "max_drawdown_pct" in result.metrics
    assert "win_rate_pct" in result.metrics
    assert "max_drawdown" not in result.metrics
    assert "win_rate" not in result.metrics
    assert "total_return" not in result.metrics
    assert -100.0 <= result.metrics["max_drawdown_pct"] <= 0.0
    assert 0.0 <= result.metrics["win_rate_pct"] <= 100.0

    metrics = run_backtest_for_agent(
        "test_strategy_metrics",
        {},
        df,
        initial_capital=10000.0,
    )
    assert -1.0 <= metrics["max_drawdown"] <= 0.0
    assert 0.0 <= metrics["win_rate"] <= 1.0

    executor = BacktestExecutor(
        backtest_fn=run_backtest_for_agent,
        strategy_name="test_strategy_metrics",
        data=df,
        validation_fn=None,
    )
    request = BacktestRequest(parameters={})
    result_agent = executor.run(request)
    assert -1.0 <= result_agent.max_drawdown <= 0.0
    assert 0.0 <= result_agent.win_rate <= 1.0


def _make_run_result(metrics: dict) -> RunResult:
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    equity = pd.Series([100.0, 105.0], index=index)
    returns = pd.Series([0.0, 0.05], index=index)
    trades = pd.DataFrame(columns=["pnl"])
    return RunResult(
        equity=equity,
        returns=returns,
        trades=trades,
        metrics=metrics,
    )


def test_ui_metrics_to_dict_canonical_keys() -> None:
    metrics = {
        "total_return": 5.0,
        "max_drawdown": -10.0,
        "win_rate": 55.0,
        "sharpe_ratio": 1.2,
        "total_pnl": 100.0,
    }
    result = _make_run_result(metrics)
    payload = UIMetrics.from_run_result(result).to_dict()

    assert "total_return_pct" in payload
    assert "max_drawdown_pct" in payload
    assert "win_rate_pct" in payload
    assert "total_return" not in payload
    assert "max_drawdown" not in payload
    assert "win_rate" not in payload


def test_stored_result_metadata_round_trip_normalizes_metrics() -> None:
    data = {
        "run_id": "run_123",
        "timestamp": "2025-01-01T00:00:00",
        "strategy": "test",
        "symbol": "TEST",
        "timeframe": "1h",
        "params": {},
        "metrics": {
            "total_return": 7.0,
            "max_drawdown": -8.0,
            "win_rate": 60.0,
            "sharpe_ratio": 1.5,
        },
        "n_bars": 100,
        "n_trades": 10,
        "period_start": "2025-01-01",
        "period_end": "2025-01-02",
        "duration_sec": 1.0,
    }
    metadata = StoredResultMetadata.from_dict(data)
    assert "total_return_pct" in metadata.metrics
    assert "max_drawdown_pct" in metadata.metrics
    assert "win_rate_pct" in metadata.metrics
    assert "total_return" not in metadata.metrics

    payload = metadata.to_dict()
    assert "total_return_pct" in payload["metrics"]
    assert "total_return" not in payload["metrics"]


def test_sweep_best_metrics_canonical_keys() -> None:
    item = SweepResultItem(
        params={},
        metrics={"total_return": 3.0, "max_drawdown": -5.0},
        success=True,
    )
    results = SweepResults(
        items=[item],
        best_params={},
        best_metrics={"win_rate": 50.0},
        total_time=0.1,
        n_completed=1,
        n_failed=0,
    )

    assert "total_return_pct" in results.items[0].metrics
    assert "max_drawdown_pct" in results.items[0].metrics
    assert "win_rate_pct" in results.best_metrics
