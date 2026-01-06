"""
Script de diagnostic pour analyser les anomalies de Sharpe Ratio.

Analyse:
1. Pourquoi le Sharpe est toujours à 6.96
2. Pourquoi les résultats sont quasi tous positifs
3. Pourquoi le Top 10 ne semble pas trié

Usage:
    python tools/diagnose_sharpe_anomaly.py
"""

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine

print("=" * 80)
print("🔍 DIAGNOSTIC DES ANOMALIES")
print("=" * 80)
print()

# ============================================================================
# Charger les données 1h
# ============================================================================
print("📊 Chargement des données BTCUSDT_1h_sample.csv...")
df = pd.read_csv('data/sample_data/BTCUSDT_1h_sample.csv', index_col=0, parse_dates=True)
print(f"   ✅ {len(df)} barres chargées ({df.index[0]} → {df.index[-1]})")
print(f"   📅 Durée: {(df.index[-1] - df.index[0]).days} jours")
print()

# ============================================================================
# ANOMALIE #1: Sharpe Ratio toujours à 6.96
# ============================================================================
print("=" * 80)
print("🔍 ANOMALIE #1: Sharpe Ratio toujours à 6.96")
print("=" * 80)
print()

# Tester plusieurs combinaisons de paramètres
test_params = [
    {'atr_period': 14, 'atr_mult': 2.0, 'leverage': 1},
    {'atr_period': 20, 'atr_mult': 3.0, 'leverage': 1},
    {'atr_period': 30, 'atr_mult': 2.5, 'leverage': 1},
    {'atr_period': 10, 'atr_mult': 1.5, 'leverage': 1},
    {'atr_period': 25, 'atr_mult': 4.0, 'leverage': 1},
]

sharpe_values = []
results_details = []

for i, params in enumerate(test_params, 1):
    print(f"Test {i}/5: {params}")

    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(df=df, strategy='atr_channel', params=params)

    sharpe = result.metrics['sharpe_ratio']
    pnl = result.metrics['total_pnl']
    trades = result.metrics['total_trades']

    sharpe_values.append(sharpe)
    results_details.append({
        'params': params,
        'sharpe': sharpe,
        'pnl': pnl,
        'trades': trades
    })

    print(f"   Sharpe={sharpe:.4f}, PnL={pnl:.2f}, Trades={trades}")

    # Analyser en détail le calcul du Sharpe
    if i == 1:  # Premier test seulement
        print("\n   🔬 Analyse détaillée du calcul Sharpe:")
        equity = result.equity
        returns = result.equity.pct_change().dropna()

        print(f"   Equity samples: {len(equity)}")
        print(f"   Returns samples: {len(returns)}")
        print(f"   Returns mean: {returns.mean():.6f}")
        print(f"   Returns std: {returns.std():.6f}")

        # Resample en quotidien (méthode daily_resample)
        if isinstance(equity.index, pd.DatetimeIndex):
            equity_daily = equity.resample('D').last().dropna()
            returns_daily = equity_daily.pct_change().dropna()

            print(f"   Daily equity samples: {len(equity_daily)}")
            print(f"   Daily returns samples: {len(returns_daily)}")
            print(f"   Daily returns mean: {returns_daily.mean():.6f}")
            print(f"   Daily returns std: {returns_daily.std():.6f}")

            # Calcul manuel du Sharpe
            if len(returns_daily) >= 60:
                periods_per_year = 252
                mean_excess = returns_daily.mean()
                std_returns = returns_daily.std(ddof=1)
                sharpe_manual = (mean_excess * np.sqrt(periods_per_year)) / std_returns

                print(f"   Sharpe calculé manuellement: {sharpe_manual:.4f}")

                if abs(sharpe_manual) > 20.0:
                    sharpe_clamped = np.sign(sharpe_manual) * 20.0
                    print(f"   ⚠️  Sharpe clampé de {sharpe_manual:.4f} → {sharpe_clamped:.4f}")
            else:
                print(f"   ⚠️  Pas assez de jours ({len(returns_daily)} < 60), Sharpe devrait être 0.0")

    print()

# Analyse des valeurs de Sharpe
print("📊 Analyse des valeurs de Sharpe obtenues:")
print(f"   Valeurs: {sharpe_values}")
print(f"   Unique values: {set(sharpe_values)}")
print(f"   Toutes identiques? {len(set(sharpe_values)) == 1}")

if len(set(sharpe_values)) == 1:
    print(f"   ⚠️  ANOMALIE CONFIRMÉE: Toutes les valeurs sont {sharpe_values[0]:.4f}")
else:
    print(f"   ✅ Sharpe varie entre {min(sharpe_values):.4f} et {max(sharpe_values):.4f}")

print()

# ============================================================================
# ANOMALIE #2: Résultats quasi tous positifs
# ============================================================================
print("=" * 80)
print("🔍 ANOMALIE #2: Résultats quasi tous positifs")
print("=" * 80)
print()

print("Analyse des 5 tests ci-dessus:")
positive_count = sum(1 for r in results_details if r['pnl'] > 0)
negative_count = sum(1 for r in results_details if r['pnl'] < 0)
zero_count = sum(1 for r in results_details if r['pnl'] == 0)

print(f"   Résultats positifs: {positive_count}/5 ({positive_count/5*100:.0f}%)")
print(f"   Résultats négatifs: {negative_count}/5 ({negative_count/5*100:.0f}%)")
print(f"   Résultats nuls: {zero_count}/5 ({zero_count/5*100:.0f}%)")

if positive_count / 5 > 0.8:
    print(f"   ⚠️  ANOMALIE POSSIBLE: {positive_count/5*100:.0f}% de résultats positifs")
    print("   Cela peut indiquer:")
    print("      - Biais dans les données (période haussière)")
    print("      - Stratégie avec biais long uniquement")
    print("      - Problème dans le calcul du P&L")

print()

# Analyse de la période des données
print("📈 Analyse de la tendance des données:")
initial_price = df['close'].iloc[0]
final_price = df['close'].iloc[-1]
market_return = (final_price - initial_price) / initial_price * 100

print(f"   Prix initial: ${initial_price:,.2f}")
print(f"   Prix final: ${final_price:,.2f}")
print(f"   Market return: {market_return:+.2f}%")

if market_return > 10:
    print(f"   ✅ Période haussière (+{market_return:.1f}%), biais positif normal pour stratégie long")
elif market_return < -10:
    print(f"   📉 Période baissière ({market_return:.1f}%), résultats positifs suspects")
else:
    print(f"   ➡️  Période neutre ({market_return:.1f}%)")

print()

# ============================================================================
# ANOMALIE #3: Top 10 mal trié
# ============================================================================
print("=" * 80)
print("🔍 ANOMALIE #3: Top 10 mal trié")
print("=" * 80)
print()

# Créer un DataFrame comme dans l'UI
df_results = pd.DataFrame([
    {
        'params': str(r['params']),
        'sharpe': r['sharpe'],
        'total_pnl': r['pnl'],
        'trades': r['trades']
    }
    for r in results_details
])

print("DataFrame AVANT tri:")
print(df_results.to_string(index=False))
print()

# Trier par sharpe descendant
df_sorted = df_results.sort_values('sharpe', ascending=False)

print("DataFrame APRÈS tri par sharpe (descendant):")
print(df_sorted.to_string(index=False))
print()

# Vérifier si le tri est correct
sharpes_before = df_results['sharpe'].tolist()
sharpes_after = df_sorted['sharpe'].tolist()
sharpes_expected = sorted(sharpes_before, reverse=True)

if sharpes_after == sharpes_expected:
    print("   ✅ Tri correct")
else:
    print("   ⚠️  ANOMALIE: Tri incorrect!")
    print(f"   Attendu: {sharpes_expected}")
    print(f"   Obtenu: {sharpes_after}")

print()

# ============================================================================
# RÉSUMÉ ET RECOMMANDATIONS
# ============================================================================
print("=" * 80)
print("📋 RÉSUMÉ")
print("=" * 80)
print()

print("Anomalies détectées:")
if len(set(sharpe_values)) == 1:
    print("   ❌ #1: Sharpe Ratio constant")
    print("      → Vérifier le calcul avec daily_resample")
    print("      → Vérifier MIN_SAMPLES_FOR_SHARPE et MIN_DAYS_FOR_SHARPE")
else:
    print("   ✅ #1: Sharpe Ratio varie normalement")

if positive_count / 5 > 0.8:
    print(f"   ⚠️  #2: {positive_count/5*100:.0f}% de résultats positifs")
    if market_return > 10:
        print("      → Normal pour période haussière")
    else:
        print("      → Suspect, vérifier le calcul du P&L")
else:
    print("   ✅ #2: Distribution de P&L normale")

if sharpes_after == sharpes_expected:
    print("   ✅ #3: Tri fonctionne correctement")
else:
    print("   ❌ #3: Tri défaillant")
    print("      → Vérifier le code de tri dans ui/app.py")

print()
print("=" * 80)
