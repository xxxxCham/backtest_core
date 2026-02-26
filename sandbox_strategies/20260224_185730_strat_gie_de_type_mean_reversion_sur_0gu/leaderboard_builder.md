# Leaderboard Builder - session 20260224_185730_strat_gie_de_type_mean_reversion_sur_0gu

Objective: Stratégie de type mean-reversion sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : lorsque le prix touche la bande inférieure de Bollinger et que VOLUME_OSCILLATOR croise au-dessus de sa moyenne sur 10 périodes, avec confirmation par EMA 50. Sorties : sortie à la première croisée au-dessus de la bande supérieure de Bollinger ou après 4 périodes si le volume diminue. Risk management : SL à 1.5x ATR, TP à 2x ATR, avec gating par volatilité implicite si le niveau de volatilité est inférieur à la moyenne sur 20 périodes.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 4 | 6 | -52.87 | 0.000 | -2.76% | -2.76% | 0.00 | 1 | stop | insufficient_trades |