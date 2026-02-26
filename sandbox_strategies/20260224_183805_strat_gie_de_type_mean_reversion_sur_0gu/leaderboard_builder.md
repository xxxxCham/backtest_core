# Leaderboard Builder - session 20260224_183805_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h. Indicateurs : STOCHASTIC + WILLIAMS_R + VOLUME_OSCILLATOR. Entrées : lorsque le STOCHASTIC croise en dessous de 20 avec un volume supérieur à la moyenne mobile sur 5 périodes, et que WILLIAMS_R est inférieur à -80, avec un volume oscillateur positif. Sorties : sortie en cours de session si le STOCHASTIC dépasse 80 ou si le prix touche la moyenne de Bollinger supérieure. Risk management : SL à 1.5x ATR, TP à 2x le SL, avec gating par volatilité implicite (si VIX > 20, ne pas entrer).
Status: failed
Best Sharpe: 0.220
Best Continuous Score: -15.56

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | -15.56 | 0.220 | -3.88% | -38.59% | 0.98 | 76 | stop | needs_work |
| 2 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 4 | 3 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 5 | 1 | -82.69 | -1.084 | -24.62% | -33.11% | 0.75 | 44 | continue | wrong_direction |
| 6 | 5 | -100.00 | -20.000 | -104.11% | -100.00% | 0.78 | 260 | continue | ruined |