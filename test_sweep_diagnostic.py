"""
Script de diagnostic pour identifier les goulots d'étranglement du sweep Numba.
Teste avec un nombre modéré de combos pour observer le comportement.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("DIAGNOSTIC SWEEP NUMBA - Identification des goulots")
print("=" * 80)

# Générer des données synthétiques pour le test
print("\n[1/5] Génération de données synthétiques...")
n_bars = 125031  # Même taille que dans les logs
np.random.seed(42)

# Simuler des prix OHLCV réalistes
base_price = 50000.0
returns = np.random.randn(n_bars) * 0.01
prices = base_price * np.exp(np.cumsum(returns))

timestamps = pd.date_range('2020-01-01', periods=n_bars, freq='1H')
df = pd.DataFrame({
    'open': prices,
    'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
    'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
    'close': prices,
    'volume': np.random.uniform(100, 10000, n_bars),
}, index=timestamps)

print(f"  ✓ Données générées: {len(df):,} bars ({df.index[0]} à {df.index[-1]})")

# Créer une grille de test MODÉRÉE (pas 1.7M combos!)
print("\n[2/5] Création grille de test (10,000 combos pour diagnostic)...")
param_grid = []
for bb_period in range(15, 50, 5):  # 7 valeurs
    for bb_std in [1.5, 2.0, 2.5, 3.0]:  # 4 valeurs
        for entry_z in [1.5, 2.0, 2.5, 3.0]:  # 4 valeurs
            for leverage in [1, 2, 3]:  # 3 valeurs
                for k_sl in [1.0, 1.5, 2.0]:  # 3 valeurs
                    param_grid.append({
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'entry_z': entry_z,
                        'leverage': leverage,
                        'k_sl': k_sl,
                    })
                    if len(param_grid) >= 10000:
                        break
                if len(param_grid) >= 10000:
                    break
            if len(param_grid) >= 10000:
                break
        if len(param_grid) >= 10000:
            break
    if len(param_grid) >= 10000:
        break

print(f"  ✓ Grille créée: {len(param_grid):,} combos")

# Lancer le sweep Numba
print("\n[3/5] Lancement sweep Numba...")
print("  [ATTENTION] Observer les logs pour identifier le goulot\n")

from backtest.sweep_numba import run_numba_sweep

start_global = time.perf_counter()

try:
    results = run_numba_sweep(
        df=df,
        strategy_key='bollinger_atr',
        param_grid=param_grid,
        initial_capital=10000.0,
        fees_bps=10.0,
        slippage_bps=5.0,
    )

    elapsed_global = time.perf_counter() - start_global

    print(f"\n[4/5] Sweep terminé!")
    print(f"  Temps total: {elapsed_global:.2f}s")
    print(f"  Throughput: {len(param_grid)/elapsed_global:,.0f} bt/s")

    # Analyser les résultats
    print(f"\n[5/5] Analyse des résultats...")
    print(f"  Résultats retournés: {len(results):,}")

    if results:
        pnls = [r['total_pnl'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        print(f"  Best PnL: ${max(pnls):,.2f}")
        print(f"  Best Sharpe: {max(sharpes):.2f}")

    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("=" * 80)

except Exception as e:
    elapsed_global = time.perf_counter() - start_global
    print(f"\n❌ ERREUR après {elapsed_global:.2f}s:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("⚠️ DIAGNOSTIC INCOMPLET - Analyser l'erreur ci-dessus")
    print("=" * 80)
