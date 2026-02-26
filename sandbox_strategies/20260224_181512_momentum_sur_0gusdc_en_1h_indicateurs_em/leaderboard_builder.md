# Leaderboard Builder - session 20260224_181512_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : croisement EMA 50/200 vers le haut avec bande de Bollinger en expansion et volume oscillateur > 0. Sorties : croisement EMA 50/200 vers le bas ou perte de 20% sur le volume oscillateur. Risk management : SL dynamique basé sur ATR, TP à 1.5x le SL avec ajustement si prix touche bande de Bollinger. Mode_inverse appliqué : entrée sur tendance inverse de EMA 200 avec confirmation contrarienne de MACD. Filtre horaire : liquidité minimale de 100k sur les 1h précedentes.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -20.000 | -108.18% | -100.00% | 0.68 | 246 | continue | ruined |