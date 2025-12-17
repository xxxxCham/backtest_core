"""
Script de diagnostic v2 pour vérifier la correction du Sharpe avec daily_resample.
"""
import numpy as np
import pandas as pd
from backtest.performance import sharpe_ratio, calculate_metrics

print("=" * 60)
print("TEST: Sharpe avec daily_resample (equity sparse)")
print("=" * 60)

# Simuler une equity qui ne change qu'aux trades
n_bars = 10000  # 10000 minutes = ~7 jours
dates = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
equity_sparse = pd.Series(10000.0, index=dates)

# Seulement 20 trades sur 10000 barres
np.random.seed(42)
trade_indices = np.random.choice(range(100, n_bars-100), size=20, replace=False)
trade_indices.sort()

for i in trade_indices:
    pnl = np.random.normal(100, 200)  # Gains/pertes variables
    equity_sparse[i:] += pnl

print(f"Equity: {len(equity_sparse)} barres sur {(dates[-1] - dates[0]).days} jours")
print(f"Nombre de trades: {len(trade_indices)}")
print(f"PnL final: ${equity_sparse.iloc[-1] - equity_sparse.iloc[0]:.2f}")
print()

# Calculer returns (la plupart seront 0.0)
returns_sparse = equity_sparse.pct_change().fillna(0)
print(f"Returns non-zéro: {(returns_sparse != 0).sum()}/{len(returns_sparse)}")
print()

# Comparer les 3 méthodes
print("COMPARAISON DES MÉTHODES:")
print("-" * 60)

# 1. Méthode standard (problématique)
sharpe_std = sharpe_ratio(
    returns_sparse,
    periods_per_year=365*24*60,  # Minutes
    method="standard"
)
print(f"1. Standard (365*24*60 minutes):    Sharpe = {sharpe_std:7.2f}  ⚠️ GONFLÉ")

# 2. Méthode trading_days (incomplet)
sharpe_td = sharpe_ratio(
    returns_sparse,
    periods_per_year=252,
    method="trading_days"
)
print(f"2. Trading days (252):              Sharpe = {sharpe_td:7.2f}  ⚠️ ENCORE GONFLÉ")

# 3. Méthode daily_resample (CORRECT)
sharpe_daily = sharpe_ratio(
    returns_sparse,
    periods_per_year=252,
    method="daily_resample",
    equity=equity_sparse
)
print(f"3. Daily resample (252):            Sharpe = {sharpe_daily:7.2f}  ✓ CORRECT")

print()
print("=" * 60)
print("EXPLICATION:")
print("=" * 60)
print("Méthode 1 (standard): Utilise tous les returns par minute,")
print("   dont 99.8% sont à 0. sqrt(525600) ≈ 725 amplifie tout.")
print()
print("Méthode 2 (trading_days): Filtre les zéros mais utilise")
print("   periods_per_year=252 sur seulement 20 returns dispersés")
print("   sur 7 jours. C'est incorrect.")
print()
print("Méthode 3 (daily_resample): Resample l'equity en quotidien")
print("   (~7 points), calcule returns quotidiens (~6 returns),")
print("   puis Sharpe avec periods_per_year=252. CORRECT.")
print()
print("Le Sharpe devrait être dans [-3, +3] pour la plupart des")
print("stratégies. Au-delà, c'est souvent un artéfact mathématique.")
print("=" * 60)
print()

# Test avec calculate_metrics complet
print("=" * 60)
print("TEST: calculate_metrics avec daily_resample")
print("=" * 60)

trades_df = pd.DataFrame({
    'pnl': [np.random.normal(100, 200) for _ in range(20)],
    'entry_ts': pd.date_range('2024-01-01', periods=20, freq='6h'),
    'exit_ts': pd.date_range('2024-01-01 01:00', periods=20, freq='6h')
})

metrics = calculate_metrics(
    equity=equity_sparse,
    returns=returns_sparse,
    trades_df=trades_df,
    initial_capital=10000.0,
    periods_per_year=252,
    sharpe_method="daily_resample"
)

print(f"Sharpe:          {metrics['sharpe_ratio']:7.2f}")
print(f"Sortino:         {metrics['sortino_ratio']:7.2f}")
print(f"Total PnL:       ${metrics['total_pnl']:,.2f}")
print(f"Total Return:    {metrics['total_return_pct']:.2f}%")
print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
print(f"Win Rate:        {metrics['win_rate']:.1f}%")
print()
print("✓ Le Sharpe devrait maintenant être dans une plage réaliste")
print("  et varier selon la performance réelle de la stratégie.")
print("=" * 60)
