# Leaderboard Builder - session 20260225_111914_strat_gie_de_type_mean_reversion_sur_fro

Objective: Stratégie de type mean-reversion sur FRONTUSDC 1h. Indicateurs : EMA + VOLUME_OSCILLATOR + WILLIAMS_R + ATR. Entrées : lorsque le prix touche le niveau de support avec un volume oscillant supérieur à la moyenne mobile sur 20 périodes et que le WILLIAMS_R est inférieur à -80. Sorties : sortie à la croisée de la moyenne mobile sur 50 périodes ou lorsque WILLIAMS_R dépasse -20. Risk management : SL fixé à partir de l'ATR avec rotation dynamique du profil risque selon la volatilité observée sur 10 périodes, TP défini comme un ratio de l'écart-type du prix sur 14 périodes.
Status: success
Best Sharpe: 0.998
Best Continuous Score: 39.70

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 100.00 | 1.681 | +138.46% | -13.81% | 2.35 | 65 | accept | target_reached |
| 2 | 5 | 39.70 | 0.998 | +33.15% | -55.12% | 1.08 | 269 | continue | high_drawdown |
| 3 | 2 | 13.02 | 0.307 | +0.79% | -1.64% | 1.25 | 3 | continue | insufficient_trades |
| 4 | 7 | -10.67 | -0.059 | -1.66% | -13.58% | 0.95 | 21 | continue | needs_work |