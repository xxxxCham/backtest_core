from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_backtest_cprofile import ProfileConfig, run_profile  # noqa: E402


def _parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Run a large cProfile batch for backtests.",
    )
    parser.add_argument("--strategy", default="ema_cross")
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--bars", type=int, default=15000)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="profiles/profile_big.pstats")
    parser.add_argument("--summary", default="profiles/profile_big.txt")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--top", type=int, default=60)
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
        "Big profile complete. "
        f"Elapsed {elapsed:.2f}s. "
        f"Stats: {config.output_path}"
    )
    if config.summary_path:
        print(f"Summary: {config.summary_path}")


if __name__ == "__main__":
    main()
