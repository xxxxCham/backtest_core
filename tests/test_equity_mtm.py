"""
Test simple du mark-to-market dans l'equity curve.
"""

import pandas as pd

from backtest.simulator_fast import calculate_equity_fast

# Données minimales : 10 barres
dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
df = pd.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'volume': [1000] * 10
}, index=dates)

# Trade simple : entrée barre 2, sortie barre 8
trades_df = pd.DataFrame({
    'entry_ts': [dates[2]],
    'exit_ts': [dates[8]],
    'pnl': [500],  # Profit de $500
    'price_entry': [102],
    'price_exit': [108],
    'size': [100],  # 100 unités
    'side': ['LONG']
})

print("=" * 60)
print("TEST MARK-TO-MARKET")
print("=" * 60)
print()
print("Données:")
print(f"  Prix: {df['close'].tolist()}")
print("  Trade: Entry @ barre 2 (prix=102), Exit @ barre 8 (prix=108)")
print("  Size: 100, P&L réalisé: $500")
print()

equity = calculate_equity_fast(df, trades_df, initial_capital=10000)

print("Equity curve (attendue avec mark-to-market):")
print("  Barre 0-1: 10000 (pas de position)")
print("  Barre 2: 10000 + (102-102)*100 = 10000 (entrée)")
print("  Barre 3: 10000 + (103-102)*100 = 10100 (MTM)")
print("  Barre 4: 10000 + (104-102)*100 = 10200 (MTM)")
print("  ...")
print("  Barre 7: 10000 + (107-102)*100 = 10500 (MTM)")
print("  Barre 8-9: 10500 (position fermée, capital réalisé)")
print()

print("Equity curve OBTENUE:")
for i, (idx, val) in enumerate(equity.items()):
    print(f"  Barre {i}: {val:.2f}")
print()

# Vérifier si le mark-to-market fonctionne
if equity.iloc[3] > equity.iloc[2]:
    print("✅ Mark-to-market fonctionne (equity augmente pendant position ouverte)")
else:
    print("❌ Mark-to-market NE fonctionne PAS (equity plate)")

# Vérifier les valeurs attendues
expected_3 = 10000 + (103 - 102) * 100  # 10100
actual_3 = equity.iloc[3]
print("\nVérification barre 3:")
print(f"  Attendu: {expected_3}")
print(f"  Obtenu: {actual_3}")
if abs(actual_3 - expected_3) < 1:
    print("  ✅ CORRECT")
else:
    print("  ❌ INCORRECT")
