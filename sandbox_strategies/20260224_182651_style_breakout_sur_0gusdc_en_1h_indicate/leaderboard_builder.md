# Leaderboard Builder - session 20260224_182651_style_breakout_sur_0gusdc_en_1h_indicate

Objective: Style breakout sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50/200 vers le haut avec prix sortant du canal Bollinger inférieur et volume oscillateur positif croissant. Sorties : crossing EMA 50/200 vers le bas ou perte de 50% du volume oscillateur. Risk management : SL à 1.5x ATR, TP à 2x ATR avec ajustement dynamique selon la force du mouvement. Filtre horaire : uniquement entrées entre 08h00 et 16h00 UTC pour maximiser la liquidité. Filtre anti-faux-signaux : confirmation par le crossing de deux niveaux de Fibonacci sur le même timeframe.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -85.65 | -0.702 | -33.00% | -42.94% | 0.78 | 69 | continue | wrong_direction |