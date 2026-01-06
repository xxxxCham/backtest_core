#!/usr/bin/env python
"""
Script de test complet - Lance plusieurs backtests sur différentes stratégies.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from backtest.engine import BacktestEngine
from strategies import list_strategies


def load_sample_data():
    """Charge les données de test."""
    sample_file = Path(__file__).parent / "data" / "sample_data" / "BTCUSDT_1h_6months.csv"
    df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def run_simple_backtest(strategy_name, df, params=None):
    """Lance un backtest simple avec paramètres par défaut."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {strategy_name.upper()}")
    print(f"{'='*80}")

    engine = BacktestEngine(
        initial_capital=10000.0
    )

    # Paramètres incluant les frais et slippage
    full_params = {
        "fees_bps": 10.0,
        "slippage_bps": 5.0,
    }
    if params:
        full_params.update(params)

    start_time = time.time()
    result = engine.run(
        df=df,
        strategy=strategy_name,
        params=full_params
    )
    duration = time.time() - start_time

    metrics = result.metrics

    # Extraire le PnL correctement (peut être dans 'pnl' ou 'total_pnl')
    pnl = metrics.get('total_pnl', metrics.get('pnl', 0))
    if pnl == 0:
        # Calculer depuis total_return_pct et capital initial
        return_pct = metrics.get('total_return_pct', 0)
        pnl = (return_pct / 100.0) * 10000.0

    print(f"⏱️  Durée: {duration:.2f}s")
    print(f"📊 PnL: ${pnl:,.2f}")
    print(f"📈 Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"📉 Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"⚡ Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"🎯 Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"🔄 Trades: {metrics.get('total_trades', 0)}")
    print(f"💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    return result


def main():
    print(f"\n{'='*80}")
    print("🚀 TEST COMPLET DES STRATÉGIES DE BACKTEST")
    print(f"{'='*80}\n")

    # Charger les données
    print("📥 Chargement des données...")
    df = load_sample_data()
    print(f"✅ {len(df)} barres chargées ({df.index[0]} → {df.index[-1]})\n")

    # Lister les stratégies
    strategies = sorted(list_strategies())
    print(f"📋 Stratégies disponibles: {len(strategies)}")
    for s in strategies:
        print(f"   • {s}")

    # Configurations de test pour chaque stratégie
    test_configs = {
        "ema_cross": [
            {"fast_period": 12, "slow_period": 26},
            {"fast_period": 15, "slow_period": 50},
        ],
        "macd_cross": [
            {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        ],
        "rsi_reversal": [
            {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
        ],
        "bollinger_atr": [
            {"bb_period": 20, "bb_std": 2.0, "atr_period": 14},
        ],
    }

    # Résultats globaux
    all_results = []

    # Tester chaque stratégie
    for strategy in strategies:
        if strategy not in test_configs:
            print(f"\n⚠️  {strategy}: Pas de config de test définie, skip")
            continue

        for params in test_configs[strategy]:
            try:
                result = run_simple_backtest(strategy, df, params)

                # Extraire le PnL correctement
                pnl = result.metrics.get('total_pnl', result.metrics.get('pnl', 0))
                if pnl == 0:
                    return_pct = result.metrics.get('total_return_pct', 0)
                    pnl = (return_pct / 100.0) * 10000.0

                all_results.append({
                    "strategy": strategy,
                    "params": params,
                    "pnl": pnl,
                    "return_pct": result.metrics.get("total_return_pct", 0),
                    "sharpe": result.metrics.get("sharpe_ratio", 0),
                    "max_dd": result.metrics.get("max_drawdown_pct", 0),
                    "trades": result.metrics.get("total_trades", 0),
                    "win_rate": result.metrics.get("win_rate_pct", 0),
                })
            except Exception as e:
                print(f"❌ Erreur: {e}")
                import traceback
                traceback.print_exc()

    # Résumé final
    print(f"\n\n{'='*80}")
    print("📊 RÉSUMÉ FINAL - TOUTES STRATÉGIES")
    print(f"{'='*80}\n")

    if all_results:
        # Trier par PnL
        sorted_results = sorted(all_results, key=lambda x: x["pnl"], reverse=True)

        print(f"{'Stratégie':<20} {'Params':<30} {'PnL':>12} {'Return%':>10} {'Sharpe':>8} {'Trades':>8}")
        print("-" * 80)

        for r in sorted_results:
            params_str = str(r["params"])[:28] + ".." if len(str(r["params"])) > 30 else str(r["params"])
            print(f"{r['strategy']:<20} {params_str:<30} ${r['pnl']:>10,.2f} {r['return_pct']:>9.2f}% {r['sharpe']:>7.2f} {r['trades']:>8}")

        # Statistiques
        print(f"\n{'='*80}")
        profitable = [r for r in all_results if r["pnl"] > 0]
        print(f"✅ Configs profitables: {len(profitable)}/{len(all_results)} ({len(profitable)/len(all_results)*100:.1f}%)")

        if profitable:
            best = sorted_results[0]
            print("\n🏆 MEILLEURE CONFIG:")
            print(f"   Stratégie: {best['strategy']}")
            print(f"   Paramètres: {best['params']}")
            print(f"   PnL: ${best['pnl']:,.2f}")
            print(f"   Return: {best['return_pct']:.2f}%")
            print(f"   Sharpe: {best['sharpe']:.2f}")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Trades: {best['trades']}")
    else:
        print("❌ Aucun résultat disponible")

    print(f"\n{'='*80}")
    print("✅ Tests terminés !")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
