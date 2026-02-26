# Leaderboard Builder - session 20260224_212140_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : EMA + OBV + STOCHASTIC. Entrées : EMA(fast) = n écarts de prix, OBV > SMA(OBV), STOC > Niveles fibo(EMA). Sorties : LONG/SHORT entre S1 et R1 en fonction des signaux entrantes. Risk management: Stop Loss à 5% sous le prix d'entrée pour la longue position, TP = stop loss + 2*slippage. Hypothese testable : Les combinaisons de EMA+OBV+STOCHASTIC sont plus probables dans une dynamique en croissance que les signaux individuels.
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -0.239 | -76.82% | -76.88% | 0.82 | 225 | continue | high_drawdown |
| 2 | 3 | -100.00 | -20.000 | -504.43% | -100.00% | 0.67 | 1055 | continue | ruined |
| 3 | 5 | -100.00 | -20.000 | -244.90% | -100.00% | 0.75 | 519 | continue | ruined |