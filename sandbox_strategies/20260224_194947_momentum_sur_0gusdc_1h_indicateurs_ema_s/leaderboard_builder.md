# Leaderboard Builder - session 20260224_194947_momentum_sur_0gusdc_1h_indicateurs_ema_s

Objective: Momentum sur 0GUSDC 1h. Indicateurs : EMA + STOCHASTIC + VOLUME_OSCILLATOR. Entrées : croisement haussier EMA(20) sur EMA(50) avec stochastic > 80 et volume oscillateur positif. Sorties : croisement baissier EMA(20) sous EMA(50) ou stochastic < 20. Risk management : SL basé sur ATR(14) avec TP fixe à 1.5x le SL, filtré par regime de volatilité via Keltner Channel.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -1.13 | 0.179 | -3.08% | -25.21% | 0.98 | 69 | continue | needs_work |
| 2 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |