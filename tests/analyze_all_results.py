#!/usr/bin/env python
"""Analyse exhaustive de tous les résultats de backtests."""

import json
from pathlib import Path

import pandas as pd


def main():
    results_dir = Path('d:/backtest_core/backtest_results')

    print('=' * 80)
    print('ANALYSE EXHAUSTIVE - TOUTES LES CONFIGURATIONS PROMETTEUSES')
    print('=' * 80)

    # 1. SWEEP EMA CROSS
    print('\n' + '=' * 80)
    print('1. SWEEP EMA CROSS (sweep_20251230_231247)')
    print('=' * 80)

    df_ema = pd.read_parquet(results_dir / 'sweep_20251230_231247/all_results.parquet')
    profitable_ema = df_ema[df_ema['total_return_pct'] > 0].drop_duplicates(
        subset=['fast_period', 'slow_period']
    ).sort_values('total_return_pct', ascending=False)

    print(f'\nNombre de configs profitables: {len(profitable_ema)}/10000')

    for i, (idx, row) in enumerate(profitable_ema.iterrows(), 1):
        print(f'\n📊 Config #{i}:')
        print(f'   Paramètres: fast_period={int(row["fast_period"])}, slow_period={int(row["slow_period"])}')
        print(f'   PnL: ${row["total_pnl"]:.2f} | Return: {row["total_return_pct"]:.2f}%')
        print(f'   Sharpe: {row["sharpe_ratio"]:.2f} | Calmar: {row["calmar_ratio"]:.2f}')
        print(f'   Win Rate: {row["win_rate"]:.1f}% | Profit Factor: {row["profit_factor"]:.2f}')
        print(f'   Max DD: {row["max_drawdown"]:.2f}% | Trades: {int(row["total_trades"])}')
        print(f'   Avg Win: ${row["avg_win"]:.2f} | Avg Loss: ${row["avg_loss"]:.2f}')
        print(f'   Largest Win: ${row["largest_win"]:.2f} | Largest Loss: ${row["largest_loss"]:.2f}')
        print(f'   Avg Trade Duration: {row["avg_trade_duration_hours"]:.1f}h')

    # 2. RUNS INDIVIDUELS PROFITABLES
    print('\n' + '=' * 80)
    print('2. RUNS INDIVIDUELS PROFITABLES')
    print('=' * 80)

    with open(results_dir / 'index.json') as f:
        index = json.load(f)

    profitable_runs = {
        run_id: data for run_id, data in index.items()
        if data['metrics']['total_pnl'] > 0
    }

    print(f'\nNombre de runs profitables: {len(profitable_runs)}/{len(index)}')

    for i, (run_id, data) in enumerate(
        sorted(profitable_runs.items(),
               key=lambda x: x[1]['metrics']['total_pnl'],
               reverse=True), 1
    ):
        print(f'\n📈 Run #{i} - {run_id}:')
        print(f'   Stratégie: {data["strategy"]}')
        print(f'   Période: {data["period_start"]} → {data["period_end"]}')
        print(f'   Symbole: {data["symbol"]} | Timeframe: {data["timeframe"]}')
        print('\n   PARAMÈTRES COMPLETS:')
        for param, value in data['params'].items():
            print(f'      {param}: {value}')

        m = data['metrics']
        print('\n   MÉTRIQUES:')
        print(f'      PnL: ${m["total_pnl"]:.2f} | Return: {m["total_return_pct"]:.2f}%')
        print(f'      CAGR: {m.get("annualized_return", 0):.2f}% | Sharpe: {m["sharpe_ratio"]:.2f}')
        print(f'      Win Rate: {m["win_rate_pct"]:.2f}% | Profit Factor: {m["profit_factor"]:.2f}')
        print(f'      Max DD: {m["max_drawdown_pct"]:.2f}% | Trades: {m["total_trades"]}')
        print(f'      Avg Win: ${m["avg_win"]:.2f} | Avg Loss: ${m["avg_loss"]:.2f}')
        print(f'      Largest Win: ${m["largest_win"]:.2f} | Largest Loss: ${m["largest_loss"]:.2f}')
        print(f'      Expectancy: ${m["expectancy"]:.2f} | Risk/Reward: {m["risk_reward_ratio"]:.2f}')

    # 3. RÉSUMÉ GLOBAL
    print('\n' + '=' * 80)
    print('3. RÉSUMÉ GLOBAL')
    print('=' * 80)

    print('\nStratégies testées:')
    print(f'  - EMA Cross: 10,000 combinaisons → {len(profitable_ema)} profitables ({len(profitable_ema)/100:.1f}%)')
    print(f'  - BollingerATR & variants: {len(index)} runs → {len(profitable_runs)} profitables ({len(profitable_runs)/len(index)*100:.1f}%)')

    if len(profitable_ema) > 0:
        best_ema = profitable_ema.iloc[0]
        print('\n🏆 MEILLEUR RÉSULTAT GLOBAL:')
        print(f'   Stratégie: EMA Cross ({int(best_ema["fast_period"])}/{int(best_ema["slow_period"])})')
        print(f'   Return: {best_ema["total_return_pct"]:.2f}% | Sharpe: {best_ema["sharpe_ratio"]:.2f} | Calmar: {best_ema["calmar_ratio"]:.2f}')


if __name__ == '__main__':
    main()
