# Leaderboard Builder - session 20260224_175117_style_momentum_sur_0gusdc_en_1h_indicate

Objective: Style momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50 vers le haut avec prix à l'intérieur de la bande de Bollinger et volume oscillateur croissant depuis 3 périodes. Sorties :跌破 bande supérieure de Bollinger ou RSI > 80 avec décroissance du volume oscillateur. Risk management : SL à 1.5x ATR, TP à 2x ATR, mode_counter_consensus requis : confirmation contrarienne partielle sur timeframe 1h avant entrée, mode_risk_rotation : profil risque serré si volatilité < 0.5%, large si > 0.8%.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -0.584 | -70.56% | -78.19% | 0.71 | 145 | continue | high_drawdown |
| 2 | 3 | -100.00 | -20.000 | -400.18% | -100.00% | 0.66 | 568 | stop | ruined |