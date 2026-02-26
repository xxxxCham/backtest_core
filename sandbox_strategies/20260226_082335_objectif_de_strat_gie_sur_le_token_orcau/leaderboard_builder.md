# Leaderboard Builder - session 20260226_082335_objectif_de_strat_gie_sur_le_token_orcau

Objective: Objectif de stratégie: Sur le token ORCAUSDC dans le timeframe 4h, j'utilisera les indicateurs EMA et ROC pour une robuste detection des familles robustes en mode breakout, mode_inverse ou mean_reversion. J'ajouterai ensuite l'indicateur BOLLINGER + STOCHASTIC en combinaison positive EMA + OBV dans le mode_hors_sentiers battus avec un filtre anti-faux-signaisls pour une gestion de risque non lineaire (SL/TP adaptes a la volatilite). Je vais également prendre en compte les comportements aleatoires imposés par mode_offbeat et mode inverse. Afin d'éviter des formulations generiques, j'ajouterai une condition additionnelle pour filtrer l'indicateur RSI entre 30/70. En fin de stratégie, je vais utiliser un SL/TP en fonction de la volatilité implicite/realisee et tester ma hypothese avec un risque management diversifié.
Status: max_iterations
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 12.86 | 0.253 | +4.73% | -33.93% | 1.03 | 43 | continue | marginal |