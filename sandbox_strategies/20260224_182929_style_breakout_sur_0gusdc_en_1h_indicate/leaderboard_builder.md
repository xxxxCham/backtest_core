# Leaderboard Builder - session 20260224_182929_style_breakout_sur_0gusdc_en_1h_indicate

Objective: Style breakout sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50/200 vers le haut avec confirmation Bollinger en mode expansion et Volume Oscillator > 0. Sorties : croisement EMA 50/200 à la baisse avec RSI > 70 ou perte de 20% du niveau d'entrée. Risk management : SL dynamique basé sur ATR avec TP fixe à 2x le SL, gating par volatilité implicite via Keltner Channel, confirmation contrarienne partielle requise via ROC > 0.5 pour éviter les faux signaux.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -140.88% | -100.00% | 0.64 | 156 | continue | ruined |
| 2 | 3 | -100.00 | -20.000 | -113.18% | -100.00% | 0.83 | 237 | stop | ruined |