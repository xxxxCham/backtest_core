"""Check saved backtest runs and find best results."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.storage import ResultStorage


def main():
    storage = ResultStorage()
    all_results = storage.list_results()

    print(f"📊 Total saved runs: {len(all_results)}")
    print("\n🏆 Top 10 runs by PNL:")
    print("-" * 80)

    sorted_by_pnl = sorted(
        all_results,
        key=lambda x: x.metrics.get('pnl', 0),
        reverse=True
    )[:10]

    for i, meta in enumerate(sorted_by_pnl, 1):
        pnl = meta.metrics.get('pnl', 0)
        sharpe = meta.metrics.get('sharpe_ratio', 0)
        print(f"{i}. {meta.run_id[:12]}... | {meta.strategy:20} | "
              f"PNL={pnl:10.2f} | Sharpe={sharpe:6.4f} | "
              f"{meta.symbol} {meta.timeframe}")
        if i <= 3:
            print(f"   Params: {meta.params}")

    print("\n" + "=" * 80)
    print("\n🎯 Top 10 runs by Sharpe Ratio:")
    print("-" * 80)

    sorted_by_sharpe = sorted(
        all_results,
        key=lambda x: x.metrics.get('sharpe_ratio', 0),
        reverse=True
    )[:10]

    for i, meta in enumerate(sorted_by_sharpe, 1):
        pnl = meta.metrics.get('pnl', 0)
        sharpe = meta.metrics.get('sharpe_ratio', 0)
        print(f"{i}. {meta.run_id[:12]}... | {meta.strategy:20} | "
              f"PNL={pnl:10.2f} | Sharpe={sharpe:6.4f} | "
              f"{meta.symbol} {meta.timeframe}")


if __name__ == "__main__":
    main()
