# Leaderboard Builder - session 20260224_223742_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : EMA + MACD + ROC. Entrées : adx>50, amplitude_hunter>80, aroon<30, atr>=2*sma(close,20), bollinger bands (upper=kelterner(high,timeframe)) + stochastic %K>70 and %D < 20. Sorties : vwap within Bollinger Bands with obv > mean_obv * 1.5. Risk management: SL = close - atr*3, TP = close + (close-sl)*0.02. Hypothese testable: La stratégie basée sur les EMA, ROC et MACD offre des gains en accélération avec un ADX élevé et une riche dynamique à court terme sur le marché USDC.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 5 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 6 | -100.00 | -20.000 | -1160.98% | -100.00% | 0.49 | 3503 | stop | ruined |