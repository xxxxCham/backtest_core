# Leaderboard Builder - session 20260224_192150_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50/200 vers le haut avec bande de Bollinger en expansion et volume oscillateur positif. Sorties : crossover EMA 50/200 vers le bas ou perte de 20% sur 1h. Risk management : SL à 1.5x ATR, TP à 2x ATR. Filtre horaire : uniquement les sessions 1h-1h UTC pour éviter les faux signaux liés à la liquidité matinale. Confirmation contrarienne : nécessite un retour de la bande de Bollinger vers le haut après une violation pour confirmer la tendance.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | -6.31 | 0.726 | +9.58% | -67.99% | 1.04 | 129 | continue | high_drawdown |
| 2 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 1 | -85.65 | -0.702 | -33.00% | -42.94% | 0.78 | 69 | continue | wrong_direction |
| 4 | 5 | -100.00 | -20.000 | -108.21% | -100.00% | 0.76 | 266 | continue | ruined |