# Leaderboard Builder - session 20260224_180850_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h. Indicateurs : STOCHASTIC + KELTNER + VOLUME_OSCILLATOR. Entrées : when STOCHASTIC crosses below 20 and KELTNER channel contracts with VOLUME_OSCILLATOR > 1.5, en mode inverse si le regime est range. Sorties : quand STOCHASTIC croise au-dessus de 80 ou SL atteint. Risk management : SL dynamique basé sur ATR, TP fixe à 1.5x SL, avec rotation du profile risque selon volatilité du jour.
Status: failed
Best Sharpe: -0.134
Best Continuous Score: -26.79

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -26.79 | -0.134 | -8.51% | -24.07% | 0.81 | 29 | stop | needs_work |
| 2 | 1 | -100.00 | -20.000 | -142.56% | -100.00% | 0.77 | 462 | continue | ruined |
| 3 | 2 | -100.00 | -20.000 | -275.48% | -100.00% | 0.83 | 1320 | continue | ruined |