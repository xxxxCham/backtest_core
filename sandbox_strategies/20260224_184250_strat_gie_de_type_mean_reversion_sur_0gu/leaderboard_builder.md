# Leaderboard Builder - session 20260224_184250_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h avec filtrage par regime trend vs range basé sur le supertrend et le donchian. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA vers le haut avec bande de bollinger étroite et volume supérieur à la moyenne sur 1h, suivi d'une confirmation contrarienne partielle via le stochastic. Sorties : sortie sur la première intersection du STOCHASTIC avec le seuil de surachat ou sur le supertrend. Risk management : SL dynamique basé sur l'ATR, TP fixe à 2x le SL, avec adaptation du seuil d'entrée selon le contexte de volatilité du VIX.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 1 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 4 | 3 | -100.00 | -20.000 | -77.52% | -100.00% | 0.85 | 357 | continue | ruined |