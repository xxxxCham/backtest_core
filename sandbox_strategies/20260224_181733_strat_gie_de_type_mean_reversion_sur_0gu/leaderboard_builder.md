# Leaderboard Builder - session 20260224_181733_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h. Indicateurs : WILLIAMS_R + VORTIC + ATR. Entrées : when Williams %R croise la zone de survente (80) avec une augmentation de l'ATR sur 2 périodes, confirmé par un creux de Vortex. Sorties : sortie sur le croisement de la moyenne mobile EMA(20) ou lorsqu’un retour de Vortex dépasse le seuil de 0.5. Risk management : SL fixé sur le maximum/minimum du candle précédent, TP dynamique basé sur l’écart-type de la Bollinger Band sur 5 périodes, avec rotation du profil risque selon le niveau d’ATR global sur 10 périodes. Hypothèse testable : Les mouvements de réversion après un signal de survente de Williams_R sont plus probables lorsque l’ATR augmente et que Vortex confirme une tendance faible.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -100.00 | -20.000 | -188.26% | -100.00% | 0.67 | 446 | stop | ruined |