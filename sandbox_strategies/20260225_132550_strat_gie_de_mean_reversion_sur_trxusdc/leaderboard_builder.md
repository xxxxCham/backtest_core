# Leaderboard Builder - session 20260225_132550_strat_gie_de_mean_reversion_sur_trxusdc

Objective: Stratégie de Mean-reversion sur TRXUSDC 1w. Indicateurs : CCI + BOLLINGER + ATR. Entrée quand le prix touche la bande extrême de CCI avec BOLLINGER en zone de survente/surachat. Take-profit au retour à la bande médiane, stop si BOLLINGER continue dans la tendance. Stop-loss dynamique basé sur ATR (2.4x), ratio risk/reward 2.5:1.
Status: failed
Best Sharpe: 0.909
Best Continuous Score: 14.37

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 14.37 | 0.718 | +40.97% | -70.72% | 1.11 | 19 | continue | high_drawdown |
| 2 | 1 | -70.52 | 0.669 | -12.30% | -84.43% | 0.97 | 18 | continue | high_drawdown |
| 3 | 2 | -78.52 | 0.475 | -16.30% | -64.63% | 0.88 | 12 | continue | high_drawdown |
| 4 | 4 | -100.00 | -20.000 | -102.31% | -100.00% | 0.72 | 20 | continue | ruined |
| 5 | 5 | -100.00 | -20.000 | -128.12% | -100.00% | 0.49 | 16 | continue | ruined |
| 6 | 7 | -100.00 | 0.000 | -53.74% | -53.74% | 0.00 | 1 | continue | insufficient_trades |
| 7 | 8 | -100.00 | 0.909 | -49.04% | -97.15% | 0.87 | 21 | continue | ruined |