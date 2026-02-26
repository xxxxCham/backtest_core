# Leaderboard Builder - session 20260224_231824_dans_le_march_bitcoin_par_rapport_l_euro

Objective: Dans le marché Bitcoin par rapport à l'euro (0GUSDC) en un heure, je vais utiliser les indicateurs fibonacci_levels, stoch_rsi, macd et atr pour une stratégie de trading basée sur la mean-reversion. J'intègrerai également le filtre horaire de liquidité dans mon modèle pour éviter les changements soudains d'ordre volumétrique important pendant l'heure considérée.

Pour ma stratégie, j'ai identifié la famille robuste des breakout qui correspondent aux courbes d'accréditation-désaccroisement et de convergence-divergence (AO) et le filtre horaire pour éviter les changements soudains. J'utiliserai ensuite l'indicateur macd, qui est reconnu comme un indicateur efficace de tendance avec une moyenne mobile externe (EMA), ainsi que l'ATR et le stoch_rsi pour mesurer la volatilité.

Lorsque les paires d'indicateurs seront satisfaits, j'appliquerai un stop loss à 1% sous le niveau des entrées (SL) et mon objectif serait de
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 51.45 | 0.485 | +48.09% | -40.15% | 1.06 | 336 | continue | overtrading |
| 2 | 2 | -100.00 | 0.453 | -24.77% | -96.02% | 0.93 | 97 | continue | ruined |
| 3 | 8 | -100.00 | -20.000 | -65.16% | -100.00% | 0.89 | 181 | continue | ruined |