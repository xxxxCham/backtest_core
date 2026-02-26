# Leaderboard Builder - session 20260224_174710_style_momentum_sur_0gusdc_en_1h_indicate

Objective: Style momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : crossover EMA 50 vers le haut avec prix sortant de bande Bollinger inférieure, confirmé par volume oscillant positif. Sorties : RSI > 70 ou跌破 bande supérieure Bollinger. Risk management : SL à 1.5x ATR, TP à 2x ATR avec ajustement dynamique selon supertrend. Mode inverse appliqué : tester entrée sur croisement EMA basse avec filtre sur tendance inverse via ADX < 20. Mode counter-consensus : exiger confirmation contrarienne sur timeframe 1h via MACD et STOCHASTIC. Hypothèse : les mouvements de tendance forte sont amplifiés par volatilité croissante dans les zones de support.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.66 | 0.000 | -0.90% | -0.90% | 0.00 | 2 | continue | insufficient_trades |
| 2 | 3 | -100.00 | -20.000 | -1475.87% | -100.00% | 0.57 | 4864 | stop | ruined |