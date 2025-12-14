"""
Script de test pour vérifier l'intégration des graphiques.

Usage:
    python test_charts_integration.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

# Test d'import
print("🧪 Test des imports...")
try:
    from ui.components.charts import (
        render_equity_and_drawdown,
        render_ohlcv_with_trades,
        render_equity_curve,
        render_comparison_chart,
    )
    print("✅ Tous les imports de charts fonctionnent")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Générer des données de test
print("\n📊 Génération des données de test...")

# Equity curve de test (croissance avec volatilité)
dates = pd.date_range("2025-01-01", periods=100, freq="1H")
equity = pd.Series(
    10000 + np.cumsum(np.random.randn(100) * 50 + 10),
    index=dates,
    name="equity"
)

print(f"✅ Equity: {len(equity)} points")
print(f"   Min: ${equity.min():.2f}, Max: ${equity.max():.2f}")

# OHLCV de test
ohlcv_data = {
    "open": 100 + np.random.randn(100).cumsum(),
    "high": 105 + np.random.randn(100).cumsum(),
    "low": 95 + np.random.randn(100).cumsum(),
    "close": 100 + np.random.randn(100).cumsum(),
    "volume": np.random.randint(1000, 10000, 100),
}
df = pd.DataFrame(ohlcv_data, index=dates)

print(f"✅ OHLCV: {len(df)} barres")

# Trades de test
trades_data = {
    "entry_ts": [dates[10], dates[30], dates[60]],
    "exit_ts": [dates[20], dates[40], dates[70]],
    "side": ["long", "short", "long"],
    "price_entry": [df.iloc[10]["close"], df.iloc[30]["close"], df.iloc[60]["close"]],
    "price_exit": [df.iloc[20]["close"], df.iloc[40]["close"], df.iloc[70]["close"]],
}
trades_df = pd.DataFrame(trades_data)

print(f"✅ Trades: {len(trades_df)} trades")

# Test des fonctions (sans Streamlit)
print("\n🔍 Validation des fonctions...")

# Test 1: Validation que les fonctions existent et ont les bonnes signatures
try:
    # Ces fonctions attendent un contexte Streamlit, donc on ne peut pas les exécuter
    # Mais on peut vérifier qu'elles existent et ont les bonnes signatures
    import inspect

    sig_equity = inspect.signature(render_equity_and_drawdown)
    assert "equity" in sig_equity.parameters
    assert "initial_capital" in sig_equity.parameters
    print("✅ render_equity_and_drawdown: signature valide")

    sig_ohlcv = inspect.signature(render_ohlcv_with_trades)
    assert "df" in sig_ohlcv.parameters
    assert "trades_df" in sig_ohlcv.parameters
    print("✅ render_ohlcv_with_trades: signature valide")

    sig_curve = inspect.signature(render_equity_curve)
    assert "equity" in sig_curve.parameters
    print("✅ render_equity_curve: signature valide")

    sig_compare = inspect.signature(render_comparison_chart)
    assert "results_list" in sig_compare.parameters
    print("✅ render_comparison_chart: signature valide")

except AssertionError as e:
    print(f"❌ Erreur de signature: {e}")
    sys.exit(1)

print("\n✅ Tous les tests passent!")
print("\n📌 Pour tester visuellement, lancez:")
print("   streamlit run ui/app.py")
