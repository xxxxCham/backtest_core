# Leaderboard Builder - session 20260224_180610_momentum_sur_0gusdc_1h_indicateurs_ema_b

Objective: Momentum sur 0GUSDC 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50 vers le haut avec prix sortant de bande Bollinger inférieure, confirmation Volume Oscillator > 0.5. Sorties : RSI > 70 ou retour sous bande Bollinger supérieure. Risk management : SL à 1.5x ATR, TP à 2x ATR. Filtre de session : uniquement les heures 1h-1h UTC, liquidité minimale 1h. Hypothèse : les impulsions de volume élevé dans un contexte de volatilité croissante indiquent des mouvements de tendance durables.
Status: failed
Best Sharpe: -0.990
Best Continuous Score: -82.10

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -82.10 | -0.990 | -10.51% | -21.56% | 0.51 | 6 | continue | needs_work |
| 2 | 1 | -85.65 | -0.702 | -33.00% | -42.94% | 0.78 | 69 | continue | wrong_direction |
| 3 | 3 | -100.00 | -1.854 | -11.95% | -14.45% | 0.20 | 5 | stop | needs_work |