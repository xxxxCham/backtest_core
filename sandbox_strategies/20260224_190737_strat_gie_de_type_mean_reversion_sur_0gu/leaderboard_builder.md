# Leaderboard Builder - session 20260224_190737_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h avec filtrage par regime trend vs range. Indicateurs : EMA + BOLLINGER + VORTEX. Entrées : crossover EMA vers le haut avec prix en dessous de la bande inférieure Bollinger, VORTEX croissant depuis un minimum local. Sorties : prix franchissant la bande supérieure Bollinger ou VORTEX décroissant. Risk management : SL dynamique basé sur ATR, TP fixe à 2x le SL, mode_risk_rotation activé selon volatilité (ATR > moyenne 20 périodes = SL serré, sinon SL large).
Status: success
Best Sharpe: 1.622
Best Continuous Score: 100.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 100.00 | 1.622 | +42.49% | -14.84% | 1.65 | 32 | accept | target_reached |
| 2 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 4 | 3 | -56.32 | -0.501 | -2.71% | -7.30% | 0.63 | 4 | continue | insufficient_trades |
| 5 | 4 | -100.00 | -1.436 | -36.99% | -46.74% | 0.54 | 31 | continue | wrong_direction |