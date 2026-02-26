# Leaderboard Builder - session 20260225_151601_strat_gie_de_mean_reversion_sur_maticusd

Objective: Stratégie de Mean-reversion sur MATICUSDC 30m. Indicateurs : WILLIAMS_R + STOCH_RSI + STOCHASTIC + ATR. Entrée quand le prix touche la bande extrême de WILLIAMS_R avec STOCH_RSI en zone de survente/surachat. Sortie quand le prix revient vers la moyenne (WILLIAMS_R neutre). Trailing stop à 1.2x ATR, take-profit à 2.9x ATR.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -20.000 | -570.59% | -100.00% | 0.58 | 1717 | continue | ruined |
| 3 | 3 | -100.00 | -20.000 | -489.27% | -100.00% | 0.57 | 1304 | continue | ruined |
| 4 | 5 | -100.00 | -20.000 | -478.55% | -100.00% | 0.57 | 1302 | continue | ruined |
| 5 | 6 | -100.00 | -20.000 | -423.11% | -100.00% | 0.60 | 1164 | stop | ruined |