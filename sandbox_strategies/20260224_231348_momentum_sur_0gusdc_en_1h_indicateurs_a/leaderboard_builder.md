# Leaderboard Builder - session 20260224_231348_momentum_sur_0gusdc_en_1h_indicateurs_a

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : AROON + EMA, MACD + ROC. Entrées : adx > 35 & bollinger_upperband < 0.8 & atr < 2% & ema(close) > ema(open)*2 & macd > signal; MomentumRSI < 70 & close[1h] > open[1h]; Sorties : AroonUp = True & MACD_signalline > Close, BollingerLowerband > -1.5 * EMA(close,-8). Risk management: SL=3% & TP=6%. Hypothese testable: Si l'AROON + EMA et le MACD + ROC sont positifs, 0GUSDC peut avoir une tendance forte en 1h avec un risque raisonnable.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | -100.00 | -20.000 | -225.88% | -100.00% | 0.78 | 1127 | continue | ruined |
| 2 | 6 | -100.00 | -20.000 | -84.69% | -100.00% | 0.87 | 476 | stop | ruined |