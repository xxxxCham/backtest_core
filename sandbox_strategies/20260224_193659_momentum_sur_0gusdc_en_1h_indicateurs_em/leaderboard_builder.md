# Leaderboard Builder - session 20260224_193659_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : croisement EMA 50/200 vers le haut avec prix à l'intérieur de la bande Bollinger et volume oscillateur > 0.5. Sorties : croisement EMA 50/200 vers le bas ou prix sortant de la bande Bollinger. Risk management : SL à 1.5x ATR, TP à 2x ATR. Mode inverse : tester logique inverse avec filtre de regime de volatilité via ADX.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 3 | 2 | -85.65 | -0.702 | -33.00% | -42.94% | 0.78 | 69 | continue | wrong_direction |