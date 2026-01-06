"""
Test rapide pour vérifier le fix du Sharpe Ratio.

Génère des données synthétiques 1h sur 6 mois et vérifie que le Sharpe varie.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402

print("=" * 80)
print("🧪 TEST FIX SHARPE RATIO")
print("=" * 80)
print()

# Générer données synthétiques 1h sur 6 mois
np.random.seed(42)
start_date = pd.Timestamp('2024-01-01', tz='UTC')
periods = 24 * 180  # 6 mois = ~180 jours x 24h
dates = pd.date_range(start=start_date, periods=periods, freq='1h')

# Prix avec tendance + bruit
price = 40000 + np.cumsum(np.random.randn(periods) * 100)
price = pd.Series(price, index=dates)

# OHLC
df = pd.DataFrame({
    'open': price,
    'high': price * 1.01,
    'low': price * 0.99,
    'close': price,
    'volume': np.random.randint(1000, 10000, size=periods)
}, index=dates)

print("📊 Données synthétiques générées:")
print(f"   {len(df)} barres sur {(dates[-1] - dates[0]).days} jours")
print("   Timeframe: 1h")
print()

# Tester plusieurs combinaisons
test_params = [
    {'atr_period': 14, 'atr_mult': 2.0, 'leverage': 1},
    {'atr_period': 20, 'atr_mult': 3.0, 'leverage': 1},
    {'atr_period': 10, 'atr_mult': 1.5, 'leverage': 1},
]

sharpe_values = []

print("Test sur 3 combinaisons de paramètres:")
for i, params in enumerate(test_params, 1):
    engine = BacktestEngine(initial_capital=10000)
    # atr_channel n'existe plus, utiliser ema_cross avec params adaptés
    ema_params = {'fast_period': params['atr_period'], 'slow_period': params['atr_period'] * 2, 'leverage': params['leverage']}
    result = engine.run(df=df, strategy='ema_cross', params=ema_params, timeframe='1h')

    sharpe = result.metrics['sharpe_ratio']
    pnl = result.metrics['total_pnl']
    trades = result.metrics['total_trades']

    sharpe_values.append(sharpe)
    print(f"   Test {i}: Sharpe={sharpe:.4f}, PnL={pnl:+.2f}, Trades={trades}")

print()

# Analyse
unique_sharpes = set(sharpe_values)
print("📊 Analyse des résultats:")
print(f"   Sharpe values: {[f'{s:.4f}' for s in sharpe_values]}")
print(f"   Valeurs uniques: {len(unique_sharpes)}")

if len(unique_sharpes) == 1:
    print(f"   ❌ ÉCHEC: Toutes les valeurs sont identiques ({sharpe_values[0]:.4f})")
    print("      Le fix n'a PAS résolu le problème")
elif sharpe_values[0] == 0.0:
    print("   ⚠️  Sharpe = 0.0 (pas assez de données après daily_resample?)")
else:
    print("   ✅ SUCCÈS: Sharpe varie correctement !")
    print(f"      Min: {min(sharpe_values):.4f}, Max: {max(sharpe_values):.4f}")

print()
print("=" * 80)
