"""Benchmark de la fonction sharpe_ratio() après désactivation logging"""
import time
import pandas as pd
import numpy as np
from backtest.performance import sharpe_ratio

# Générer des séries de returns réalistes
np.random.seed(42)

def generate_returns(n_points=100):
    """Génère des returns aléatoires réalistes"""
    returns = np.random.randn(n_points) * 0.01  # 1% volatilité quotidienne
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    return pd.Series(returns, index=dates)

def generate_equity(n_points=100):
    """Génère une courbe d'equity réaliste"""
    returns = np.random.randn(n_points) * 0.01
    equity = 10000 * (1 + returns).cumprod()
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    return pd.Series(equity, index=dates)

# Nombre d'appels (équivalent à plusieurs backtests dans une grid search)
N_CALLS = 1000

print(f"Benchmark sharpe_ratio(): {N_CALLS} appels")
print("-" * 60)

# Test 1: sharpe_ratio standard (sans run_id)
print("Test 1: Sans run_id (pas de logging)")
returns_list = [generate_returns(100) for _ in range(N_CALLS)]

start = time.perf_counter()
for returns in returns_list:
    _ = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252)
elapsed1 = time.perf_counter() - start

calls_per_sec1 = N_CALLS / elapsed1 if elapsed1 > 0 else 0
print(f"  Temps: {elapsed1:.3f}s")
print(f"  Appels/sec: {calls_per_sec1:.0f}")
print(f"  Temps par appel: {elapsed1/N_CALLS*1000:.2f}ms")

# Test 2: sharpe_ratio avec run_id (logging potentiel)
print(f"\nTest 2: Avec run_id (logging activé)")
returns_list = [generate_returns(100) for _ in range(N_CALLS)]
equity_list = [generate_equity(100) for _ in range(N_CALLS)]

start = time.perf_counter()
for i, (returns, equity) in enumerate(zip(returns_list, equity_list)):
    _ = sharpe_ratio(
        returns,
        risk_free=0.0,
        periods_per_year=252,
        method='daily_resample',
        equity=equity,
        run_id=f"test_{i}"
    )
elapsed2 = time.perf_counter() - start

calls_per_sec2 = N_CALLS / elapsed2 if elapsed2 > 0 else 0
print(f"  Temps: {elapsed2:.3f}s")
print(f"  Appels/sec: {calls_per_sec2:.0f}")
print(f"  Temps par appel: {elapsed2/N_CALLS*1000:.2f}ms")

# Analyse
print("\n" + "=" * 60)
print("ANALYSE:")
print(f"  Ralentissement avec logging: {elapsed2/elapsed1:.1f}x")
print(f"  Impact par appel: {(elapsed2-elapsed1)/N_CALLS*1000:.2f}ms")

if calls_per_sec2 > 500:
    print(f"\n✅ Excellent: {calls_per_sec2:.0f} appels/sec - Logging désactivé efficacement")
elif calls_per_sec2 > 300:
    print(f"\n✅ Bon: {calls_per_sec2:.0f} appels/sec - Performance acceptable")
elif calls_per_sec2 > 100:
    print(f"\n⚠️ Moyen: {calls_per_sec2:.0f} appels/sec - Encore de l'overhead")
else:
    print(f"\n❌ Lent: {calls_per_sec2:.0f} appels/sec - Logging encore actif")

print("\nNOTE: Pour backtests complets, attendez ~1/10 de ces chiffres")
print("      (indicateurs, signaux, simulation, métriques additionnelles)")
