# Leaderboard Builder - session 20260224_205223_objectif_strat_gie_de_trading_sur_le_mar

Objective: Objectif: Stratégie de trading sur le marché 0GUSDC en 1h qui utilise des indicateurs inusuals pour trouver les moments où l'amplitude est haute (High_Aroon, high_ADX) et la volatilité est faible (low_ATR). Combinant ces deux facteurs avec un signal de trend puissant (EMA Crossover) et une combinaison des moyennes mobiles efficaces (Bollinger Band + Stochastic), nous avons l'intention d'obtenir des signaux pour acheter ou vendre les crypto-monnaies.

Les entrées seront basées sur un filtre horaire de liquidité élevé et une gating par volatilité implicite/réelle, avec un SL/TP adapté à la volatilité pour réduire les pertes maximales. La stratégie sera testée sous plusieurs configurations des filtres sur le temps (long vs court), le trend ou non (trend filter) et l'utilisation d'un signal de confirmation partiellement contraire pour éviter un comportement trop linéaire.

Notre objectif est de trouver une stratégie unique qui exploite des axes non usuel du marché, gère efficacement la volatilité et apporte
Status: success
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 89.58 | 1.174 | +34.17% | -22.35% | 1.30 | 45 | accept | target_reached |
| 2 | 1 | -27.55 | 0.001 | -4.68% | -18.31% | 0.92 | 17 | continue | needs_work |