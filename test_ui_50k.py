"""
Test Streamlit UI avec grille 50K combos (au lieu de 1.7M)
Pour diagnostiquer si le problème vient de la taille de la grille
"""
import os
import sys
import time
import numpy as np
import pandas as pd

# Config Numba
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

sys.path.insert(0, 'd:/backtest_core')

from backtest.sweep_numba import run_numba_sweep

print("="*80)
print("TEST GRILLE MOYENNE - 50K combos")
print("="*80)

# Générer données
print("\n[1/3] Génération données...")
n_bars = 125031
df = pd.DataFrame({
    'close': np.random.randn(n_bars) * 10 + 100,
    'high': np.random.randn(n_bars) * 10 + 101,
    'low': np.random.randn(n_bars) * 10 + 99,
}, index=pd.date_range('2020-01-01', periods=n_bars, freq='1h'))
print(f"  ✓ {n_bars:,} bars")

# Grille 50K
print("\n[2/3] Création grille 50K...")
param_grid = []
for bb_period in range(15, 50, 2):  # 18 valeurs
    for bb_std in np.arange(1.5, 3.5, 0.2):  # 10 valeurs
        for entry_z in [1.5, 2.0, 2.5, 3.0]:  # 4 valeurs
            for leverage in [1, 2, 3]:  # 3 valeurs
                for k_sl in [1.5, 2.0, 2.5, 3.0, 4.0]:  # 5 valeurs
                    param_grid.append({
                        'bb_period': bb_period,
                        'bb_std': float(bb_std),
                        'entry_z_buy': entry_z,
                        'entry_z_sell': entry_z,
                        'leverage': leverage,
                        'k_sl': k_sl,
                    })

print(f"  ✓ {len(param_grid):,} combos")

# Sweep Numba
print("\n[3/3] Sweep Numba 50K combos...")
print("  Si ça bloque ici → problème avec grilles moyennes aussi")
print("  Si ça passe → problème uniquement avec 1.7M combos\n")

start = time.perf_counter()
try:
    results = run_numba_sweep(
        df=df,
        strategy_key='bollinger_bands',
        param_grid=param_grid,
        initial_capital=10000,
    )
    elapsed = time.perf_counter() - start

    print(f"\n✅ SUCCÈS - {len(results):,} résultats en {elapsed:.1f}s")
    print(f"   Throughput: {len(results)/elapsed:,.0f} bt/s")

except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
