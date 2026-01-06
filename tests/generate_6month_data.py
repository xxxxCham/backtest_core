"""
Génère BTCUSDT_1h_6months.csv avec 6 mois de données (4320 barres).
"""

import numpy as np
import pandas as pd

print("Génération de BTCUSDT_1h_6months.csv...")

np.random.seed(42)
start_date = pd.Timestamp('2024-08-03', tz='UTC')
periods = 4326  # Correspondant à vos données réelles
dates = pd.date_range(start=start_date, periods=periods, freq='h')

# Prix réaliste BTC: 40k → 50k sur 6 mois
price_base = 40000
trend = np.linspace(0, 10000, periods)
noise = np.random.randn(periods).cumsum() * 200
price = price_base + trend + noise

df = pd.DataFrame({
    'open': price,
    'high': price + np.abs(np.random.randn(periods) * 150),
    'low': price - np.abs(np.random.randn(periods) * 150),
    'close': price + np.random.randn(periods) * 50,
    'volume': np.random.randint(10000, 100000, size=periods)
}, index=dates)

df['high'] = df[['open', 'high', 'close']].max(axis=1)
df['low'] = df[['open', 'low', 'close']].min(axis=1)

output_path = 'data/sample_data/BTCUSDT_1h_6months.csv'
df.to_csv(output_path)

print(f"✅ Créé: {output_path}")
print(f"   {len(df)} barres")
print(f"   {df.index[0]} → {df.index[-1]}")
print(f"   {(df.index[-1] - df.index[0]).days} jours")
print(f"   Prix: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
