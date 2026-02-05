"""
Test avec 1.7M combos (comme dans les logs de l'utilisateur) pour valider l'optimisation.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("TEST FINAL - 1.7M COMBOS (validation optimisation)")
print("=" * 80)

# Générer données (même taille que les logs)
print("\n[1/3] Génération données (125K bars)...")
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

# Créer grille 1.7M combos
print("\n[2/3] Création grille 1.7M combos (comme dans les logs)...")
param_grid = []
target_size = 1771561  # Exactement comme dans les logs

print("  Génération en cours... (peut prendre 10-30s)", flush=True)
for bb_period in range(10, 200):
    for bb_std in np.arange(0.5, 5.0, 0.1):
        for entry_z in np.arange(0.5, 5.0, 0.1):
            for leverage in [1, 2, 3, 5, 10]:
                for k_sl in np.arange(0.5, 4.0, 0.25):
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

    # Feedback tous les 100K combos
    if len(param_grid) % 100000 == 0:
        print(f"  ... {len(param_grid):,} combos générés", flush=True)

print(f"  ✓ Grille créée: {len(param_grid):,} combos")
print(f"  Taille estimée: {len(param_grid) * 200 / 1024**2:.0f} MB en mémoire")

# Lancer le sweep
print("\n[3/3] Lancement sweep Numba (1.7M combos)...")
print("  ATTENDU:")
print("    • Kernel Numba: ~250-300s (comme avant)")
print("    • Construction: ~2-5s (au lieu de bloquer 10 min!)")
print()

from backtest.sweep_numba import run_numba_sweep

start_total = time.perf_counter()

try:
    results = run_numba_sweep(
        df=df,
        strategy_key='bollinger_atr',
        param_grid=param_grid,
        initial_capital=10000.0,
        fees_bps=10.0,
        slippage_bps=5.0,
    )

    elapsed_total = time.perf_counter() - start_total

    print(f"\n{'='*80}")
    print("✅ SUCCÈS - Sweep 1.7M TERMINÉ!")
    print(f"{'='*80}")
    print(f"  Temps total: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Résultats: {len(results):,}")
    print(f"  Throughput: {len(param_grid)/elapsed_total:,.0f} bt/s")

    if results:
        pnls = [r['total_pnl'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        print(f"\n  Meilleurs résultats:")
        print(f"    • Best PnL: ${max(pnls):,.2f}")
        print(f"    • Best Sharpe: {max(sharpes):.2f}")

    print(f"\n{'='*80}")

except KeyboardInterrupt:
    elapsed_total = time.perf_counter() - start_total
    print(f"\n⚠️ INTERROMPU après {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

except Exception as e:
    elapsed_total = time.perf_counter() - start_total
    print(f"\n❌ ERREUR après {elapsed_total:.1f}s:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
