# Leaderboard Builder - session 20260226_080245_strat_gie_sur_twtusdc_30m_objectif_sur_l

Objective: Stratégie sur TWTUSDC 30m. Objectif: "Sur le marché de {token} avec un timeframe d'30m, lancer une stratégie de trading qui combine les indicateurs 'EMA+MACD' et la famille robuste des 'breakout', gère correctement les risques en utilisant un stop-loss à {SL}/un take profit a {TP}. Critères pour l'optimisation: * Le nouveau type d'asymetrie long/short sera testé avec des seuils différents, et une confirmation inverse partielle est demandée. * La mise en place de filtre anti-faux-signaux dans le contexte du regime change (trend vs range) permettra l'ajustement dynamique à la volatilité implicite/realisée. * Des comportements aléatoires sont imposés pour tester une logique inversee puis filtrer par régime, et des paires d'indicateurs rares seront testées pour une approche offbeat. * Un nouveau critère de mode_offbeat sera évalué en alternant entre les deux modes dans la stratégie.
Status: max_iterations
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -28.87 | -0.184 | -9.54% | -35.39% | 0.94 | 132 | continue | needs_work |