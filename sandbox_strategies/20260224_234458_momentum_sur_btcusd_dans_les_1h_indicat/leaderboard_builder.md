# Leaderboard Builder - session 20260224_234458_momentum_sur_btcusd_dans_les_1h_indicat

Objective: [Momentum] sur BTCUSD dans les 1h. Indicateurs : EMA + ATR + ROC. Entrées : RSI > 70, Momentum > 65. Sorties : TP = SCP * (1 - retracement de courbe Bollinger), SL = TP - (TP * volatilité). Diversification: Asymetrie long/short (>25%), filtre horaire alternant entre liquidite basse et haute selon les conditions moyennes. Hypothèse testable : Combinaison d'indicateurs de trend et de momentum pour anticiper une tendance forte et prolongée sur le marché BTCUSD dans les 1h.
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 14.87 | 1.029 | +24.51% | -76.85% | 1.08 | 104 | continue | high_drawdown |
| 2 | 1 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 3 | 3 | -100.00 | -20.000 | -104.11% | -100.00% | 0.78 | 260 | continue | ruined |