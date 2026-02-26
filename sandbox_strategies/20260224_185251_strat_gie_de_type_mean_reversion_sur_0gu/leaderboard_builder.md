# Leaderboard Builder - session 20260224_185251_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h avec un angle de nouveauté "mode_inverse". Indicateurs : EMA + BOLLINGER + VORTEX. Entrées : lorsque le prix touche la bande supérieure de Bollinger et que VORTEX croise sa ligne de référence, mais uniquement si le regime est detecté comme range par le filtre Supertrend. Sorties : sortie sur le crossing de la moyenne mobile EMA ou à 2x le ATR. Risk management : SL fixé sur la bande inférieure de Bollinger, TP fixé sur le niveau de pivot haute. Mode_microstructure : ajoute un filtre de liquidité minimal sur le volume et une contrainte horaire de 1h-1h UTC.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 4 | 2 | -93.87 | -1.587 | -6.46% | -6.98% | 0.28 | 7 | continue | needs_work |