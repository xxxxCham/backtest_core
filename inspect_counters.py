"""Inspecter la structure des PerfCounters"""
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
from utils.config import Config

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

config = Config()
engine = BacktestEngine(initial_capital=10000, config=config)

result = engine.run(
    df=df,
    strategy='ema_cross',
    params={'k_sl': 0.5},
    symbol='TEST',
    timeframe='5min'
)

print("Structure de result.meta['perf_counters']:")
print("=" * 70)
import json
print(json.dumps(result.meta['perf_counters'], indent=2, default=str))
