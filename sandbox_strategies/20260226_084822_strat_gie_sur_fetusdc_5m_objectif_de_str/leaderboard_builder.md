# Leaderboard Builder - session 20260226_084822_strat_gie_sur_fetusdc_5m_objectif_de_str

Objective: Stratégie sur FETUSDC 5m. Objectif de stratégie crypto: Sur le marché (token) arbitré par la fibonacci_levels sur un timeframe horaire, utiliser l'adx pour identifier les tendances fortes et la donchian pour les tendances faibles. Incorporer une combinaison d'EMA + OBV pour le breakout à des niveaux de volatilité élevés (1h), ajustant la régularité de l'adaptation du régime en fonction de la liquidité horaire, et utilisant un ROC avec une combinaison moyennant le vote majoritaire pour les angles de changements rapides. Utiliser un filtre horaire de liquidité pour éviter les situations à faible volatilité qui peuvent induire des erreurs d'apprentissage, et imposer des conditions additionnelles telles que la MACD en croisement avec le ROC ou l'oscillation du supertrend. Mettre en place un SL/TP adéquats pour éviter les pertes excessives tout en garantissant une entrée raisonnablement favorable, et tester une logique inversee sur certains indicateurs avant de les inclure dans la stratégie. Entrées : fibonacci_levels + donchian ; ROC avec vote majoritaire + MACD
Status: max_iterations
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -965.00% | -100.00% | 0.61 | 3778 | continue | ruined |