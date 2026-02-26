# Leaderboard Builder - session 20260224_212415_objectif_de_strat_gie_de_trading_style_r

Objective: Objectif de stratégie de trading
Style relativement agressif sur le marché Bitcoin en 1h. Indicateurs : EMA(fast) + OBV + MACD, entrées basées sur la volatilité implicite (80% bande 9 définition), sorties testant des zones de confluence fibo et pivot points. Gating par volatilite implique/(realise)/1h liquidité > avgLiquid, filtre anti-false signals sur RSI(7). Risk management: SL/TP=0.1*.5%. Hypothèse testable : "Le MACD en combinaison avec OBV et EMA offre une stratégie de trading agressive efficace sur le marché Bitcoin en 1h".
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -83.15 | 0.363 | -31.55% | -72.91% | 0.97 | 738 | continue | overtrading |
| 2 | 1 | -100.00 | -20.000 | -96.50% | -100.00% | 0.80 | 175 | continue | ruined |