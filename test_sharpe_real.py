"""
Test rapide du Sharpe sur un backtest réel avec données 15m.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
import pandas as pd

print("=" * 70)
print("TEST SHARPE RATIO SUR BACKTEST RÉEL")
print("=" * 70)
print()

# Charger les données sample
print("1. Chargement des données sample BTCUSDT 1m...")
df = pd.read_csv(
    "data/sample_data/BTCUSDT_1m_sample.csv",
    index_col=0,
    parse_dates=True
)
print(f"   ✓ {len(df)} barres chargées ({df.index[0]} à {df.index[-1]})")
print(f"   ✓ Période: {(df.index[-1] - df.index[0]).days} jours")
print()

# Créer le moteur et lancer backtest
print("2. Exécution du backtest avec stratégie bollinger_atr...")
engine = BacktestEngine(initial_capital=10000)

params = {
    "period": 20,
    "std_dev": 2.0,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "k_sl": 1.5,
    "k_tp": 2.5
}

result = engine.run(
    df=df,
    strategy="bollinger_atr",
    params=params,
    symbol="BTCUSDT",
    timeframe="1m"
)

print(f"   ✓ Backtest terminé en {result.meta['duration_sec']:.2f}s")
print()

# Afficher les métriques clés
print("3. Métriques de performance:")
print("-" * 70)
metrics = result.metrics

print(f"   Total PnL:           ${metrics['total_pnl']:>12,.2f}")
print(f"   Total Return:        {metrics['total_return_pct']:>12.2f}%")
print(f"   Annualized Return:   {metrics['annualized_return']:>12.2f}%")
print()
print(f"   Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}  ← CORRIGÉ")
print(f"   Sortino Ratio:       {metrics['sortino_ratio']:>12.2f}")
print(f"   Max Drawdown:        {metrics['max_drawdown']:>12.2f}%")
print(f"   Volatilité Ann.:     {metrics['volatility_annual']:>12.2f}%")
print()
print(f"   Total Trades:        {metrics['total_trades']:>12d}")
print(f"   Win Rate:            {metrics['win_rate']:>12.1f}%")
print(f"   Profit Factor:       {metrics['profit_factor']:>12.2f}")
print(f"   Avg Win:             ${metrics['avg_win']:>12,.2f}")
print(f"   Avg Loss:            ${metrics['avg_loss']:>12,.2f}")
print()

# Validation du Sharpe
print("4. Validation du Sharpe:")
print("-" * 70)
sharpe = metrics['sharpe_ratio']

if -10 < sharpe < 10:
    print(f"   ✅ Sharpe = {sharpe:.2f} est dans la plage attendue [-10, +10]")
else:
    print(f"   ⚠️  Sharpe = {sharpe:.2f} est hors plage (suspect)")

if sharpe != 3.49 and sharpe != -3.49:
    print(f"   ✅ Sharpe ≠ ±3.49 (l'ancien bug est corrigé)")
else:
    print(f"   ❌ Sharpe = ±3.49 (bug toujours présent !)")

# Interpréter le Sharpe
if sharpe < 0:
    interpretation = "Stratégie perdante"
elif sharpe < 1:
    interpretation = "Stratégie faible"
elif sharpe < 2:
    interpretation = "Stratégie correcte"
elif sharpe < 3:
    interpretation = "Stratégie bonne"
else:
    interpretation = "Stratégie excellente (ou peu de données)"

print(f"   📊 Interprétation: {interpretation}")
print()

# Test avec plusieurs paramètres pour vérifier que Sharpe varie
print("5. Test variabilité du Sharpe (3 backtests avec params différents):")
print("-" * 70)

test_params = [
    {"period": 10, "std_dev": 1.5, "k_sl": 1.0, "k_tp": 2.0},
    {"period": 20, "std_dev": 2.0, "k_sl": 1.5, "k_tp": 2.5},
    {"period": 30, "std_dev": 2.5, "k_sl": 2.0, "k_tp": 3.0}
]

sharpes = []
for i, p in enumerate(test_params, 1):
    r = engine.run(df=df, strategy="bollinger_atr", params=p, symbol="BTCUSDT", timeframe="1m")
    s = r.metrics['sharpe_ratio']
    sharpes.append(s)
    pnl = r.metrics['total_pnl']
    print(f"   Test {i}: Sharpe = {s:6.2f} | PnL = ${pnl:>10,.2f} | Params: {p}")

print()
unique_sharpes = len(set([round(s, 1) for s in sharpes]))
if unique_sharpes >= 2:
    print(f"   ✅ Sharpe varie entre tests ({unique_sharpes} valeurs distinctes)")
    print(f"   ✅ Min: {min(sharpes):.2f}, Max: {max(sharpes):.2f}, Range: {max(sharpes)-min(sharpes):.2f}")
else:
    print(f"   ⚠️  Sharpe trop constant entre tests")

print()
print("=" * 70)
print("✅ TEST TERMINÉ")
print("=" * 70)
