#!/usr/bin/env python3
"""
Script CLI pour lancer un grid search mini sur la stratégie ATR Channel.

Usage:
    python tools/run_atr_grid_mini.py [--n-bars N] [--workers W]

Options:
    --n-bars N      Nombre de barres OHLCV (défaut: 500)
    --workers W     Nombre de workers parallèles (défaut: 4)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.sweep import SweepEngine
from strategies.atr_channel import ATRChannelStrategy


def generate_test_data(n_bars: int = 500, volatility: float = 0.02) -> pd.DataFrame:
    """
    Génère des données OHLCV synthétiques réalistes.

    Args:
        n_bars: Nombre de barres à générer
        volatility: Volatilité des prix (0.02 = 2% par barre)

    Returns:
        DataFrame OHLCV avec index datetime
    """
    print(f"📊 Génération de {n_bars} barres OHLCV...")

    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')

    # Prix avec tendance et bruit
    np.random.seed(42)
    trend = np.linspace(100, 120, n_bars)  # Tendance haussière modérée
    noise = np.cumsum(np.random.randn(n_bars) * volatility * 100)
    close = trend + noise

    # OHLC réalistes
    high = close + np.abs(np.random.randn(n_bars)) * volatility * 100
    low = close - np.abs(np.random.randn(n_bars)) * volatility * 100
    open_price = close + (np.random.randn(n_bars) * volatility * 50)
    volume = np.random.randint(10000, 100000, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)

    df.index.name = 'timestamp'

    print(f"   Période: {df.index[0]} → {df.index[-1]}")
    print(f"   Prix: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    print(f"   Range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    return df


def main():
    """Exécute le grid search mini sur ATR Channel."""
    parser = argparse.ArgumentParser(description="Grid search mini sur stratégie ATR Channel")
    parser.add_argument("--n-bars", type=int, default=500, help="Nombre de barres OHLCV")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de workers parallèles")

    args = parser.parse_args()

    print("=" * 80)
    print("📈 GRID SEARCH MINI - STRATÉGIE ATR CHANNEL")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"   Barres OHLCV: {args.n_bars}")
    print(f"   Workers: {args.workers}")

    # Étape 1: Générer les données
    df = generate_test_data(n_bars=args.n_bars)

    # Étape 2: Créer la stratégie
    strategy = ATRChannelStrategy()

    # Étape 3: Définir la grille de paramètres (mini: 3x3 = 9 combinaisons)
    param_grid = {
        'atr_period': [10, 14, 20],      # 3 valeurs
        'atr_mult': [1.5, 2.0, 2.5],     # 3 valeurs
        'leverage': [1],                  # Fixe
    }

    n_combinations = len(param_grid['atr_period']) * len(param_grid['atr_mult'])
    print("\n📋 Grille de paramètres:")
    print(f"   atr_period: {param_grid['atr_period']}")
    print(f"   atr_mult: {param_grid['atr_mult']}")
    print(f"   leverage: {param_grid['leverage']}")
    print(f"   Total: {n_combinations} combinaisons")

    # Étape 4: Lancer le sweep
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT DU GRID SEARCH")
    print("=" * 80)

    engine = SweepEngine(
        max_workers=args.workers,
        use_processes=True,
        initial_capital=10000.0,
        auto_save=True,
    )

    start_time = time.perf_counter()

    try:
        results = engine.run_sweep(
            df=df,
            strategy=strategy,
            param_grid=param_grid,
            show_progress=True,
        )

        elapsed = time.perf_counter() - start_time

        print("\n" + "=" * 80)
        print("✅ GRID SEARCH TERMINÉ")
        print("=" * 80)

        print(f"\n⏱️  Temps total: {elapsed:.1f}s")
        print(f"📊 Combinaisons testées: {results.n_completed}/{n_combinations}")
        print(f"❌ Combinaisons échouées: {results.n_failed}")

        # Afficher les meilleurs résultats
        print("\n" + "=" * 80)
        print("🏆 TOP 5 - MEILLEURS PARAMÈTRES (Sharpe Ratio)")
        print("=" * 80)

        top_5 = results.get_top_n(n=5, metric="sharpe_ratio")

        for i, row in top_5.iterrows():
            print(f"\n#{i+1}:")
            print(f"   atr_period: {row.get('atr_period', 'N/A')}")
            print(f"   atr_mult: {row.get('atr_mult', 'N/A')}")
            print(f"   Sharpe: {row.get('sharpe_ratio', 0):.2f}")
            print(f"   Return: {row.get('total_return', 0)*100:.1f}%")
            print(f"   Max DD: {row.get('max_drawdown', 0)*100:.1f}%")
            print(f"   Trades: {row.get('total_trades', 0):.0f}")

        # Sauvegarder les résultats
        output_file = Path("backtest_results") / f"atr_grid_mini_{int(time.time())}.csv"
        output_file.parent.mkdir(exist_ok=True)

        df_results = results.to_dataframe()
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_file}")

        return 0

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ Erreur après {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Grid search interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
