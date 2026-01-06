from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestEngine  # noqa: E402
from data.loader import load_ohlcv  # noqa: E402
from strategies.base import get_strategy  # noqa: E402


@dataclass
class ProfileConfig:
    strategy: str
    symbol: str
    timeframe: str
    runs: int
    bars: int
    initial_capital: float
    seed: int
    use_real_data: bool
    start: Optional[str]
    end: Optional[str]
    output_path: str
    summary_path: Optional[str]
    sort_by: str
    top_n: int
    verbose_logs: bool


def _timeframe_to_freq(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if not tf:
        return "1h"
    num = ""
    unit = ""
    for ch in tf:
        if ch.isdigit():
            num += ch
        else:
            unit += ch
    if not num or unit not in {"m", "h", "d"}:
        return "1h"
    if unit == "m":
        return f"{num}min"
    return f"{num}{unit}"


def _generate_synthetic_ohlcv(bars: int, seed: int, timeframe: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    freq = _timeframe_to_freq(timeframe)
    dates = pd.date_range("2024-01-01", periods=bars, freq=freq)

    prices = 100 + rng.standard_normal(bars).cumsum() * 0.5
    highs = prices + np.abs(rng.standard_normal(bars) * 0.3)
    lows = prices - np.abs(rng.standard_normal(bars) * 0.3)
    opens = prices + rng.standard_normal(bars) * 0.1
    volumes = rng.integers(1000, 10000, size=bars)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


def _load_data(config: ProfileConfig) -> Tuple[pd.DataFrame, str]:
    if config.use_real_data:
        try:
            df = load_ohlcv(
                config.symbol,
                config.timeframe,
                start=config.start,
                end=config.end,
            )
            if df is not None and not df.empty:
                return df, "real"
        except Exception:
            pass
    df = _generate_synthetic_ohlcv(config.bars, config.seed, config.timeframe)
    return df, "synthetic"


def _get_default_params(strategy_key: str) -> Dict[str, float]:
    strategy_class = get_strategy(strategy_key)
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_key}")
    instance = strategy_class()
    params = dict(getattr(instance, "default_params", {}) or {})
    if "leverage" not in params:
        params["leverage"] = 1
    return params


def _run_batch(
    engine: BacktestEngine,
    df: pd.DataFrame,
    strategy: str,
    params: Dict[str, float],
    runs: int,
    symbol: str,
    timeframe: str,
    seed: int,
    silent_mode: bool,
) -> None:
    for i in range(runs):
        engine.run(
            df=df,
            strategy=strategy,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            seed=seed + i,
            silent_mode=silent_mode,
        )


def run_profile(config: ProfileConfig) -> float:
    df, data_source = _load_data(config)
    params = _get_default_params(config.strategy)

    engine = BacktestEngine(initial_capital=config.initial_capital)
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    _run_batch(
        engine=engine,
        df=df,
        strategy=config.strategy,
        params=params,
        runs=config.runs,
        symbol=config.symbol,
        timeframe=config.timeframe,
        seed=config.seed,
        silent_mode=not config.verbose_logs,
    )
    profiler.disable()
    elapsed = time.perf_counter() - start

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(output_path))

    if config.summary_path:
        summary_path = Path(config.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("PROFILE SUMMARY\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"strategy: {config.strategy}\n")
            handle.write(f"runs: {config.runs}\n")
            handle.write(f"bars: {len(df)}\n")
            handle.write(f"data_source: {data_source}\n")
            handle.write(f"elapsed_sec: {elapsed:.3f}\n")
            handle.write(f"output: {output_path}\n")
            handle.write("\n")
            stats = pstats.Stats(profiler, stream=handle)
            stats.strip_dirs().sort_stats(config.sort_by).print_stats(config.top_n)

    return elapsed


def _parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Run cProfile against a batch of backtests.",
    )
    parser.add_argument("--strategy", default="ema_cross")
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="profile_run.pstats")
    parser.add_argument("--summary", default="profile_run.txt")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--verbose-logs", action="store_true")

    args = parser.parse_args()
    summary_path = None if args.no_summary else args.summary

    return ProfileConfig(
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        runs=args.runs,
        bars=args.bars,
        initial_capital=args.initial_capital,
        seed=args.seed,
        use_real_data=args.use_real_data,
        start=args.start,
        end=args.end,
        output_path=args.output,
        summary_path=summary_path,
        sort_by=args.sort,
        top_n=args.top,
        verbose_logs=args.verbose_logs,
    )


def main() -> None:
    config = _parse_args()
    elapsed = run_profile(config)
    print(
        "Profile complete. "
        f"Elapsed {elapsed:.2f}s. "
        f"Stats: {config.output_path}"
    )
    if config.summary_path:
        print(f"Summary: {config.summary_path}")


if __name__ == "__main__":
    main()
