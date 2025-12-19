"""
Test pour reproduire le bug des "37.5 trades" dans le grid search.

Vérifie que total_trades est toujours un entier, jamais un float.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine

print("=" * 80)
print("🧪 TEST GRID SEARCH - VÉRIFICATION DU TYPE DE total_trades")
print("=" * 80)
print()

# Charger données
try:
    df = pd.read_csv('data/sample_data/BTCUSDT_1h_6months.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ Fichier BTCUSDT_1h_6months.csv introuvable")
    sys.exit(1)

print(f"📊 Données: {len(df)} barres")
print()

# Simuler un petit grid search
param_grid = [
    {'atr_period': 10, 'atr_mult': 1.5, 'leverage': 1},
    {'atr_period': 14, 'atr_mult': 2.0, 'leverage': 1},
    {'atr_period': 20, 'atr_mult': 2.5, 'leverage': 1},
    {'atr_period': 28, 'atr_mult': 3.0, 'leverage': 1},
    {'atr_period': 30, 'atr_mult': 4.0, 'leverage': 1},
]

print(f"🔬 Grille de {len(param_grid)} combinaisons")
print()

results_list = []

for i, params in enumerate(param_grid, 1):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(df=df, strategy='atr_channel', params=params, timeframe='1h')

    metrics = result.metrics
    total_trades = metrics["total_trades"]

    # Vérifier le type
    print(f"{i}. {params}")
    print(f"   total_trades = {total_trades}")
    print(f"   Type = {type(total_trades).__name__}")

    if not isinstance(total_trades, (int, np.integer)):
        print(f"   ❌ PROBLÈME: total_trades n'est pas un int!")
    else:
        print(f"   ✅ Type correct")

    # Simuler ce que fait l'UI
    result_dict = {
        "params": str(params),
        "total_pnl": metrics["total_pnl"],
        "sharpe": metrics["sharpe_ratio"],
        "max_dd": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "trades": metrics["total_trades"],  # Comme dans l'UI
        "profit_factor": metrics["profit_factor"]
    }

    results_list.append(result_dict)
    print()

# Créer DataFrame comme dans l'UI
print("=" * 80)
print("📋 CRÉATION DU DATAFRAME (comme dans l'UI)")
print("=" * 80)
print()

results_df = pd.DataFrame(results_list)

print("Colonnes du DataFrame:")
print(results_df.dtypes)
print()

print("Valeurs de 'trades':")
print(results_df['trades'].values)
print()

# Vérifier si il y a des floats
trades_values = results_df['trades'].values
if any(isinstance(x, float) and not x.is_integer() for x in trades_values):
    print("❌ PROBLÈME DÉTECTÉ: Des valeurs fractionnaires dans 'trades'!")
    for i, val in enumerate(trades_values):
        if isinstance(val, float) and not val.is_integer():
            print(f"   Ligne {i}: {val}")
else:
    print("✅ Toutes les valeurs de 'trades' sont des entiers")

print()

# Afficher le DataFrame
print("Top 5 résultats:")
print(results_df.sort_values("sharpe", ascending=False).head())
print()

# Test avec des clés dupliquées (peut-être la cause?)
print("=" * 80)
print("🧪 TEST: Doublons avec groupby")
print("=" * 80)
print()

# Ajouter intentionnellement un doublon
duplicate = results_list[0].copy()
results_list_with_dup = results_list + [duplicate]

print(f"Liste avec doublon: {len(results_list_with_dup)} éléments")

df_with_dup = pd.DataFrame(results_list_with_dup)
print("\nDataFrame brut:")
print(df_with_dup['trades'].values)

# Si on groupe par params et fait une moyenne
grouped = df_with_dup.groupby('params').mean(numeric_only=True)
print("\nAprès groupby().mean():")
print(grouped['trades'].values)

if any(not x.is_integer() for x in grouped['trades'].values):
    print("\n⚠️  CAUSE POTENTIELLE: groupby().mean() crée des fractions!")
    print("   Solution: Utiliser .first() ou .last() au lieu de .mean()")
else:
    print("\n✅ Pas de fractions")

print()
print("=" * 80)
