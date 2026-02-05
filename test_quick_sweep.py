"""
Test rapide pour reproduire le problème de l'utilisateur avec 507 combos.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("TEST RAPIDE - 507 combos (comme utilisateur)")
print("=" * 80)

# Données synthétiques
print("\n[1/3] Génération données...")
n_bars = 10000  # Plus petit pour aller vite
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

# Créer grille 507 combos
print("\n[2/3] Création grille 507 combos...")
param_grid = []
for bb_period in range(15, 35, 2):  # 10 valeurs
    for bb_std in [1.5, 2.0, 2.5]:  # 3 valeurs
        for entry_z in [1.5, 2.0, 2.5]:  # 3 valeurs
            for leverage in [1, 2]:  # 2 valeurs
                for k_sl in [1.0, 1.5, 2.0]:  # 3 valeurs
                    param_grid.append({
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'entry_z': entry_z,
                        'leverage': leverage,
                        'k_sl': k_sl,
                    })
                    if len(param_grid) >= 507:
                        break
                if len(param_grid) >= 507:
                    break
            if len(param_grid) >= 507:
                break
        if len(param_grid) >= 507:
            break
    if len(param_grid) >= 507:
        break

print(f"  ✓ Grille créée: {len(param_grid):,} combos")

# Callback pour feedback
def progress_cb(completed, total, best_result):
    print(f"  [CALLBACK] {completed}/{total} combos traités", flush=True)

# Lancer le sweep
print("\n[3/3] Lancement sweep Numba...")
print("  Si ça bloque, on a reproduit le problème!\n")

from backtest.sweep_numba import run_numba_sweep

start = time.perf_counter()

try:
    results = run_numba_sweep(
        df=df,
        strategy_key='bollinger_atr',
        param_grid=param_grid,
        initial_capital=10000.0,
        fees_bps=10.0,
        slippage_bps=5.0,
        progress_callback=progress_cb,
    )

    elapsed = time.perf_counter() - start

    print(f"\n✅ TERMINÉ en {elapsed:.2f}s")
    print(f"  Résultats: {len(results):,}")
    print(f"  Throughput: {len(param_grid)/elapsed:,.0f} bt/s")

    if results:
        pnls = [r['total_pnl'] for r in results]
        print(f"  Best PnL: ${max(pnls):,.2f}")

except Exception as e:
    elapsed = time.perf_counter() - start
    print(f"\n❌ ERREUR après {elapsed:.2f}s:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
