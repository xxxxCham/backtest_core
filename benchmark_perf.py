"""Benchmark rapide pour mesurer les runs/sec du BacktestEngine"""
import time
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
from backtest.indicators import compute_indicators
from backtest.signal_generator import generate_signals

# Générer données OHLCV synthétiques
np.random.seed(42)
n_bars = 1000
dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')

prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
df = pd.DataFrame({
    'open': prices + np.random.randn(n_bars) * 0.1,
    'high': prices + abs(np.random.randn(n_bars) * 0.3),
    'low': prices - abs(np.random.randn(n_bars) * 0.3),
    'close': prices,
    'volume': np.random.randint(1000, 10000, n_bars)
}, index=dates)

# Calculer indicateurs
df = compute_indicators(df, indicators=['sma_short', 'sma_long', 'rsi', 'atr'])

# Paramètres de test
params_list = [
    {'k_entry': 0.5, 'k_tp': 1.0, 'k_sl': 0.5, 'trailing_stop': False},
    {'k_entry': 0.8, 'k_tp': 1.5, 'k_sl': 0.8, 'trailing_stop': False},
    {'k_entry': 1.0, 'k_tp': 2.0, 'k_sl': 1.0, 'trailing_stop': True},
    {'k_entry': 0.6, 'k_tp': 1.2, 'k_sl': 0.6, 'trailing_stop': False},
    {'k_entry': 0.7, 'k_tp': 1.8, 'k_sl': 0.7, 'trailing_stop': True},
]

# Benchmark
n_runs = len(params_list)
print(f"Benchmark: {n_runs} backtests sur {n_bars} barres")
print("-" * 50)

start_time = time.perf_counter()

for i, params in enumerate(params_list, 1):
    signals = generate_signals(df, strategy='crossing')

    engine = BacktestEngine(
        df=df,
        signals=signals,
        initial_capital=10000,
        k_entry=params['k_entry'],
        k_tp=params['k_tp'],
        k_sl=params['k_sl'],
        trailing_stop=params['trailing_stop'],
        logging_mode=None  # Désactiver tous les logs
    )

    result = engine.run()

    if result and 'metrics' in result:
        sharpe = result['metrics'].get('sharpe_ratio', 0)
        print(f"Run {i}/{n_runs} - Sharpe: {sharpe:.2f}")

elapsed = time.perf_counter() - start_time
runs_per_sec = n_runs / elapsed if elapsed > 0 else 0

print("-" * 50)
print(f"Temps total: {elapsed:.2f}s")
print(f"Performance: {runs_per_sec:.1f} runs/sec")
print(f"Barres/sec: {(n_runs * n_bars / elapsed):.0f}")

# Diagnostic
if runs_per_sec < 50:
    print("\n⚠️ Performance < 50 runs/sec - Logging encore actif ?")
elif runs_per_sec < 80:
    print("\n⚠️ Performance < 80 runs/sec - Autres optimisations nécessaires")
else:
    print(f"\n✅ Performance restaurée ({runs_per_sec:.1f} runs/sec)")
