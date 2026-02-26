# Leaderboard Builder - session 20260225_093627_objectif_de_strat_gie_de_trading_sur_le

Objective: Objectif de stratégie de trading : Sur le marché DOGEUSDC 4h, utiliser l'indicateur fibonacci_levels et la combinaison EMA + OBV. Les entrées seront les niveaux fibos, l'EMA de 10 jours et l'OBV quotidienne. La sortie sera le Vwap ajusté en fonction des conditions d'entrée. Le stop loss (SL) sera à -2% du VWAP optimal, tandis que le take profit (TP) sera a +3%. Pour la gestion de risque, nous utiliserons un SL/TP de 1%/2%, et nous allouerons une place maximale pour chaque position de 0.5%. Cette stratégie doit être testée sur les dernières deux années de données, avec une volatilité de l'ordre de 10%, et un trend dominant actuellement observé dans le marché DOGEUSDC.
Status: failed
Best Sharpe: 0.391
Best Continuous Score: 17.70

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 76.17 | 0.674 | +72.98% | -31.70% | 1.18 | 212 | continue | approaching_target |
| 2 | 6 | 17.70 | 0.391 | +32.58% | -57.90% | 1.05 | 267 | continue | high_drawdown |
| 3 | 7 | -24.10 | 0.221 | -4.57% | -43.92% | 0.99 | 208 | continue | needs_work |
| 4 | 1 | -100.00 | -20.000 | -166.33% | -100.00% | 0.63 | 127 | continue | ruined |