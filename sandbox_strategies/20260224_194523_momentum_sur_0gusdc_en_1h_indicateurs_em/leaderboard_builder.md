# Leaderboard Builder - session 20260224_194523_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50/200 vers le haut avec prix sortant de bande Bollinger inférieure, confirmation volume oscillant positif. Sorties : croisement EMA 50/200 vers le bas ou prix touchant bande supérieure Bollinger. Risk management : SL dynamique basé sur ATR, TP fixe à 1.5x le SL, filtre anti-faux-signaux par confirmation contrarienne partielle (RSI > 70 après entrée long), mode_counter_consensus activé.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 3 | 1 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 4 | 4 | -100.00 | -20.000 | -83.73% | -100.00% | 0.85 | 205 | continue | ruined |