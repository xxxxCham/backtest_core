# Leaderboard Builder - session 20260224_233322_objectif_cr_er_une_strat_gie_de_trading

Objective: Objectif: Créer une stratégie de trading qui profite du marché 0GUSDC dans le timeframe 1h en utilisant les indicateurs suivants : adx, amplitude_hunter, aroon, atr, bollinger, cci, donchian, ema, fear_greed, fibonacci_levels, ichimoku, keltner, macd, mfi, momentum, obv, onchain_smoothing, pi_cycle, pivot_points, psar, roc, rsi, sma, standard_deviation, stoch_rsi, stochastic, supertrend, volume_oscillator, vortex, vwap, williams_r.

Les anciens ancrages performants identifiés dans les sessions précédentes sont : familles robustes detectees: breakout, momentum, mean-reversion ; combinaisons indicateurs deja positives: EMA + OBV, BOLLINGER + STOCHASTIC, EMA ; angles de nouveaute deja prometteurs: curated.

Contraintes de diversification imposées incluent : intègre au moins un axe 'hors sentiers battus' parmi : asymetrie long/short (seuils differents), combinaison de signaux contradictoires avec vote majoritaire, filtre horaire de liquid
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -0.639 | -92.75% | -96.19% | 0.71 | 136 | continue | ruined |
| 2 | 5 | -100.00 | -0.263 | -52.98% | -82.69% | 0.91 | 494 | continue | overtrading |
| 3 | 6 | -100.00 | -0.017 | -58.07% | -79.35% | 0.86 | 207 | stop | high_drawdown |