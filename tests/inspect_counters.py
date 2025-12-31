"""
Module-ID: inspect_counters

Purpose: Inspecter la structure des PerfCounters du BacktestEngine - vérifier clefs et formats.

Role in pipeline: performance inspection

Key components: engine.counters, affichage JSON

Inputs: Engine instance avec backtest exécuté

Outputs: Compteurs JSON (time_indicators, time_signals, time_trading, etc.)

Dependencies: json, numpy, pandas, backtest.engine

Conventions: Compteurs structures en dict Python

Read-if: Vérifier quels compteurs sont capturés.

Skip-if: Vous n'avez pas besoin d'inspecter compteurs.
"""
import json

import numpy as np
import pandas as pd

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
print(json.dumps(result.meta['perf_counters'], indent=2, default=str))
