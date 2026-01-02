#!/usr/bin/env python
"""
Script CLI pour lancer un backtest en mode grille.

Usage:
    python run_grid_backtest.py --strategy atr_channel --symbol BTCUSDC --timeframe 30m
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
from data.loader import load_ohlcv
from itertools import product
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Lancer un backtest en mode grille")
    parser.add_argument("--strategy", default="atr_channel", help="Nom de la stratégie")
    parser.add_argument("--symbol", default="BTCUSDC", help="Symbole (ex: BTCUSDC)")
    parser.add_argument("--timeframe", default="30m", help="Timeframe (ex: 1h, 30m, 1d)")
    parser.add_argument("--start-date", default="2024-12-01", help="Date de début")
    parser.add_argument("--end-date", default="2024-12-31", help="Date de fin")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--max-combos", type=int, default=100, help="Nombre max de combinaisons")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"🚀 BACKTEST MODE GRILLE")
    print(f"{'='*80}")
    print(f"Stratégie: {args.strategy}")
    print(f"Symbole: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Période: {args.start_date} → {args.end_date}")
    print(f"Capital initial: ${args.initial_capital:,.2f}")
    print(f"{'='*80}\n")

    # Charger les données
    print("📥 Chargement des données...")
    try:
        df = load_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start_date,
            end=args.end_date
        )
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1

    print(f"✅ Données chargées: {len(df)} barres")
    print(f"   Période: {df.index[0]} → {df.index[-1]}\n")

    # Définir la grille de paramètres selon la stratégie
    if args.strategy == "atr_channel":
        param_grid = {
            "atr_period": [10, 14, 20],
            "atr_multiplier": [1.5, 2.0, 2.5],
            "ema_fast": [8, 12],
            "ema_slow": [21, 34],
        }
    elif args.strategy == "bollinger_atr":
        param_grid = {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "entry_z": [1.5, 2.0, 2.5],
            "atr_period": [10, 14],
            "atr_percentile": [20, 30, 40],
            "k_sl": [1.0, 1.5, 2.0],
            "leverage": [1, 2],
        }
    else:
        print(f"❌ Stratégie '{args.strategy}' non configurée dans ce script")
        return 1

    # Générer toutes les combinaisons
    param_names = list(param_grid.keys())
    param_values_lists = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values_lists))

    print(f"📊 Grille de paramètres:")
    for name, values in param_grid.items():
        print(f"   {name}: {values}")
    print(f"\n   Total combinaisons: {len(combinations)}")

    if len(combinations) > args.max_combos:
        print(f"⚠️  Limite de {args.max_combos} combinaisons appliquée")
        combinations = combinations[:args.max_combos]

    print(f"\n{'='*80}")
    print("🔄 EXÉCUTION DES BACKTESTS")
    print(f"{'='*80}\n")

    # Créer l'engine
    engine = BacktestEngine(initial_capital=args.initial_capital)

    # Stocker les résultats
    results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        try:
            result = engine.run(
                df=df,
                strategy=args.strategy,
                params=params,
                symbol=args.symbol,
                timeframe=args.timeframe,
            )

            if result and result.metrics:
                metrics = result.metrics
                results.append({
                    "combo": i,
                    "params": str(params),
                    "pnl": metrics.get("total_pnl", 0),
                    "sharpe": metrics.get("sharpe_ratio", 0),
                    "max_dd": metrics.get("max_drawdown_pct", 0),
                    "win_rate": metrics.get("win_rate_pct", 0),
                    "trades": metrics.get("total_trades", 0),
                })

                # Afficher progression
                if i % 10 == 0 or i == len(combinations):
                    print(f"[{i:3d}/{len(combinations)}] "
                          f"PnL: ${metrics.get('total_pnl', 0):>10,.2f} | "
                          f"Sharpe: {metrics.get('sharpe_ratio', 0):>6.2f} | "
                          f"Trades: {metrics.get('total_trades', 0):>4d}")

        except Exception as e:
            print(f"[{i:3d}/{len(combinations)}] ❌ Erreur: {e}")
            continue

    print(f"\n{'='*80}")
    print("📊 RÉSULTATS TOP 10")
    print(f"{'='*80}\n")

    if not results:
        print("❌ Aucun résultat valide")
        return 1

    # Trier par Sharpe ratio
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    print(results_df.head(10).to_string(index=False))

    print(f"\n{'='*80}")
    best = results_df.iloc[0]
    print(f"🏆 MEILLEURE COMBINAISON:")
    print(f"   Paramètres: {best['params']}")
    print(f"   PnL: ${best['pnl']:,.2f}")
    print(f"   Sharpe: {best['sharpe']:.3f}")
    print(f"   Max DD: {best['max_dd']:.2f}%")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Trades: {best['trades']}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
