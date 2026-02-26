# Leaderboard Builder - session 20260224_195337_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50/200 vers le haut avec prix qui sort du canal Bollinger inférieur et volume oscillateur croît de 15% sur 3 périodes. Sorties : atteinte du niveau supérieur de Bollinger ou retour en dessous de l'EMA 200. Risk management : SL fixé à la moyenne mobile exponentielle de 10 périodes sous le point d'entrée, TP à 1.5 fois le SL avec ajustement dynamique selon ATR. Mode risk rotation : adaptation du SL/TP selon l'indice Fear & Greed. Mode counter consensus : confirmation contrarienne partielle requise via un retournement du signal Stochastic sur timeframe 1h. Hypothèse : les mouvements de tendance sont amplifiés par une forte volatilité confirmée par le volume oscillateur, avec une asymétrie longue lorsque le prix est en dessous du canal Bollinger.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 8 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 4 | 5 | -100.00 | -20.000 | -102.10% | -100.00% | 0.45 | 237 | continue | ruined |