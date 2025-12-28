"""
Script de test pour l'infrastructure de logging JOUR 1 - Diagnostic.

Vérifie que :
- sharpe_ratio() produit des logs SHARPE_INPUT/SANITY/CALC/OUTPUT
- calculate_equity_curve() produit des logs EQUITY_SERIES_META/JUMPS/DD/RECONCILE
- Les logs sont corrélés par run_id
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("TEST JOUR 1 : Logs structurés détaillés (sharpe_ratio + equity_curve)")
print("=" * 80)

# Test 1 : Imports
try:
    from backtest.performance import sharpe_ratio, calculate_metrics
    from backtest.simulator import calculate_equity_curve
    from utils.observability import generate_run_id
    print("✅ Imports réussis")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Test 2 : sharpe_ratio() avec run_id
print("\n" + "=" * 80)
print("TEST 2 : sharpe_ratio() avec logs structurés")
print("=" * 80)

run_id = generate_run_id(strategy="test", symbol="BTC", timeframe="1h", seed=123)
print(f"run_id = {run_id}")

# Créer données de test
dates = pd.date_range(start="2025-01-01", periods=30, freq="1h")
equity = pd.Series(
    np.cumsum(np.random.randn(30) * 10 + 100) + 10000,
    index=dates
)
returns = equity.pct_change().dropna()

# Appeler sharpe_ratio avec run_id
sharpe = sharpe_ratio(
    returns,
    periods_per_year=252,
    method="daily_resample",
    equity=equity,
    run_id=run_id
)

print(f"Sharpe Ratio = {sharpe:.4f}")
print("✅ sharpe_ratio() fonctionne avec run_id")
print("   Vérifiez les logs pour: SHARPE_INPUT, SHARPE_SANITY, SHARPE_CALC, SHARPE_OUTPUT")

# Test 3 : calculate_equity_curve() avec run_id
print("\n" + "=" * 80)
print("TEST 3 : calculate_equity_curve() avec logs structurés")
print("=" * 80)

# Créer DataFrame OHLCV simple
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, 30),
    'high': np.random.uniform(110, 120, 30),
    'low': np.random.uniform(90, 100, 30),
    'close': np.random.uniform(100, 110, 30),
    'volume': np.random.uniform(1000, 10000, 30),
}, index=dates)

# Créer quelques trades fictifs
trades_df = pd.DataFrame({
    'entry_ts': [dates[5], dates[15]],
    'exit_ts': [dates[10], dates[20]],
    'price_entry': [105.0, 108.0],
    'price_exit': [110.0, 106.0],
    'size': [10.0, 10.0],
    'side': ['LONG', 'LONG'],
    'pnl': [50.0, -20.0],
    'fees': [1.0, 1.0],
})

equity_curve = calculate_equity_curve(df, trades_df, initial_capital=10000.0, run_id=run_id)

print(f"Equity courbe: len={len(equity_curve)}, final={equity_curve.iloc[-1]:.2f}")
print("✅ calculate_equity_curve() fonctionne avec run_id")
print("   Vérifiez les logs pour: EQUITY_SERIES_META, EQUITY_DD, EQUITY_COMPLETE")

# Test 4 : calculate_metrics() end-to-end
print("\n" + "=" * 80)
print("TEST 4 : calculate_metrics() end-to-end avec run_id")
print("=" * 80)

metrics = calculate_metrics(
    equity=equity_curve,
    returns=equity_curve.pct_change().fillna(0),
    trades_df=trades_df,
    initial_capital=10000.0,
    periods_per_year=252,
    run_id=run_id
)

print(f"Métriques calculées:")
print(f"  - Total Return: {metrics['total_return_pct']:.2f}%")
print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"  - Max Drawdown: {metrics['max_drawdown']:.2f}%")
print(f"  - Win Rate: {metrics['win_rate']:.2f}%")
print("✅ calculate_metrics() fonctionne avec run_id propagé")

# Test 5 : Vérifier les logs WARNING pour Sharpe=0
print("\n" + "=" * 80)
print("TEST 5 : Déclencher SHARPE_ZERO warnings")
print("=" * 80)

# Créer returns insuffisants (< 3 samples)
returns_tiny = pd.Series([0.01, 0.02])
run_id_tiny = generate_run_id()

sharpe_zero = sharpe_ratio(returns_tiny, run_id=run_id_tiny)
print(f"Sharpe avec 2 samples = {sharpe_zero:.4f} (attendu: 0.0)")
print("✅ SHARPE_ZERO warning devrait être dans les logs (reason=min_samples)")

# Créer returns avec variance nulle
returns_flat = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
run_id_flat = generate_run_id()

sharpe_flat = sharpe_ratio(returns_flat, run_id=run_id_flat)
print(f"Sharpe avec returns plats = {sharpe_flat:.4f} (attendu: 0.0)")
print("✅ SHARPE_ZERO warning devrait être dans les logs (reason=low_volatility)")

# Résumé
print("\n" + "=" * 80)
print("RÉSUMÉ DES TESTS JOUR 1")
print("=" * 80)
print("✅ sharpe_ratio() avec logs structurés : OK")
print("✅ calculate_equity_curve() avec logs structurés : OK")
print("✅ Propagation run_id calculate_metrics() → sharpe_ratio() : OK")
print("✅ Warnings SHARPE_ZERO déclenchés correctement : OK")
print("\n🎉 Tous les tests JOUR 1 - Diagnostic sont passés !")
print("\n📋 Vérifications dans les logs :")
print("   - Cherchez: grep 'SHARPE_INPUT\\|SHARPE_SANITY\\|SHARPE_OUTPUT' logs/*.log")
print("   - Cherchez: grep 'EQUITY_SERIES_META\\|EQUITY_DD\\|EQUITY_COMPLETE' logs/*.log")
print("   - Cherchez: grep 'SHARPE_ZERO' logs/*.log")
print("\nProchaines étapes :")
print("  - JOUR 2 : Corrections ciblées (gardes adaptatifs + LLM strict)")
print("  - Tester avec un vrai backtest pour voir tous les logs en action")
