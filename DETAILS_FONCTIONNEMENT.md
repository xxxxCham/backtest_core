# Details de fonctionnement - backtest_core

This document is the canonical reference for runtime behavior, entrypoints,
data handling, metrics, and LLM orchestration logs. It replaces overlapping
historical docs while keeping them for traceability.

## Entrypoints

- CLI: `python -m backtest_core` -> `cli.main` (see `cli/__init__.py`)
- UI: `streamlit run ui/app.py`
- Known broken scripts in `pyproject.toml`: `backtest-ui`, `backtest-demo`
  (missing module/main). Use the CLI or Streamlit entrypoint above.

## CLI commands (current)

- list (strategies, indicators, data, presets)
- info (strategy, indicator)
- backtest
- sweep
- optuna
- validate
- export
- visualize

## Data loading

- Search order for data root:
  1) `BACKTEST_DATA_DIR`
  2) `TRADX_DATA_ROOT`
  3) `D:/ThreadX_big/data/crypto/processed/parquet`
  4) `data/sample_data`
  5) fallback candidates in `data/loader.py` (cwd and repo paths)
- File discovery is recursive and cached (`_scan_data_files`).
- Normalization (`_normalize_ohlcv`) enforces:
  - columns: open, high, low, close, volume (lowercase)
  - DatetimeIndex (from index or time column)
  - timezone forced to UTC (localize if naive, convert if tz-aware)
  - sorted index and NaN rows dropped
  - float64 for OHLCV

## Execution flow

- CLI/UI/agents -> `backtest.engine.BacktestEngine`
- `_calculate_indicators` -> `indicators.registry.calculate_indicator`
  - exceptions are logged and values set to None
- `StrategyBase.generate_signals`
- `backtest.simulator` or `backtest.simulator_fast`
- `backtest.performance.calculate_metrics`

## Metrics and Sharpe

- `calculate_metrics` exposes `total_return_pct`, `max_drawdown`, `win_rate`
  already in percent (no extra *100 in the CLI).
- Annualized return uses calendar duration when `equity.index` is datetime;
  otherwise falls back to `periods_per_year`.
- Sharpe defaults to `daily_resample`:
  - requires `equity` and DatetimeIndex
  - resamples equity to daily and uses 252 for annualization
  - guards: >=3 samples and >=3 non-zero returns
  - min annualized vol = 0.1%, clamp to +/-20
- Volatility annualization:
  - daily resample with 252 when `sharpe_method=daily_resample`
  - otherwise std(returns) * sqrt(periods_per_year)
- Max drawdown duration uses timestamps when available; otherwise uses bar
  counts and `periods_per_year`.

## LLM orchestration logs

- Logger: `agents/orchestration_logger.py`
- Viewer UI: `ui/orchestration_viewer.py`
- Integration: pass `orchestration_logger` to
  `agents.integration.create_optimizer_from_engine`
- UI (LLM mode) creates a logger and renders logs in Streamlit.
- Logs are saved as JSON in the current working directory with
  `OrchestrationLogger.save_to_file()`.
- Detailed API reference: `docs/ORCHESTRATION_LOGS.md`

## Quick tests

- `python -m pytest tests/test_sharpe_ratio.py`
- `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py`
- Expect a warning if `.pytest_cache` is not writable.

## Legacy docs (historical)

The following files are historical and may be out of date. Keep them for
traceability but treat this document as the source of truth.

- `ORCHESTRATION_COMPLETE.md`
- `LIVRAISON_ORCHESTRATION_LOGS.md`
- `AUDIT.md`
- `CHANGELOG_AUDIT.md`
- `FIX_PLAN.md`
