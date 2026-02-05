"""
Test avec une grille MASSIVE pour reproduire le blocage (500K+ combos).
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("TEST GRILLE MASSIVE - Reproduction du blocage (500K combos)")
print("=" * 80)

# Générer données
print("\n[1/4] Génération données synthétiques...")
n_bars = 125031
np.random.seed(42)
base_price = 50000.0
returns = np.random.randn(n_bars) * 0.01
prices = base_price * np.exp(np.cumsum(returns))

timestamps = pd.date_range('2020-01-01', periods=n_bars, freq='h')
df = pd.DataFrame({
    'open': prices,
    'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
    'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
    'close': prices,
    'volume': np.random.uniform(100, 10000, n_bars),
}, index=timestamps)

print(f"  ✓ {len(df):,} bars générées")

# Créer GROSSE grille (500K combos pour voir le goulot)
print("\n[2/4] Création GROSSE grille (500K combos)...")
param_grid = []
target_size = 500000

for bb_period in range(10, 100):  # 90 valeurs
    for bb_std in np.arange(1.0, 4.0, 0.25):  # 12 valeurs
        for entry_z in np.arange(1.0, 4.0, 0.25):  # 12 valeurs
            for leverage in [1, 2, 3, 5]:  # 4 valeurs
                for k_sl in np.arange(0.5, 3.0, 0.5):  # 5 valeurs
                    param_grid.append({
                        'bb_period': int(bb_period),
                        'bb_std': float(bb_std),
                        'entry_z': float(entry_z),
                        'leverage': float(leverage),
                        'k_sl': float(k_sl),
                    })
                    if len(param_grid) >= target_size:
                        break
                if len(param_grid) >= target_size:
                    break
            if len(param_grid) >= target_size:
                break
        if len(param_grid) >= target_size:
            break
    if len(param_grid) >= target_size:
        break

print(f"  ✓ Grille créée: {len(param_grid):,} combos")
print(f"  Estimation mémoire résultats: {len(param_grid) * 200 / 1024**2:.1f} MB")

# Lancer le sweep
print("\n[3/4] Lancement sweep Numba...")
print("  [OBSERVATION] Le kernel Numba devrait être rapide...")
print("  [OBSERVATION] MAIS la construction des résultats va BLOQUER!\n")

from backtest.sweep_numba import run_numba_sweep

start_global = time.perf_counter()
checkpoint_kernel = None

# Interception pour mesurer le temps du kernel vs construction
import logging
logging.basicConfig(level=logging.INFO)

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

    print(f"\n[4/4] ✅ Sweep terminé (si on arrive ici)!")
    print(f"  Temps total: {elapsed_global:.2f}s")
    print(f"  Résultats: {len(results):,}")
    print(f"  Throughput: {len(param_grid)/elapsed_global:,.0f} bt/s")

except KeyboardInterrupt:
    elapsed_global = time.perf_counter() - start_global
    print(f"\n⚠️ INTERROMPU après {elapsed_global:.2f}s")
    print("  → Le blocage se situe APRÈS le kernel Numba")
    print("  → Problème identifié: boucle Python sur 500K dicts")

except Exception as e:
    elapsed_global = time.perf_counter() - start_global
    print(f"\n❌ ERREUR après {elapsed_global:.2f}s:")
    print(f"  {type(e).__name__}: {e}")

print("\n" + "=" * 80)
