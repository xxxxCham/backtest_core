# Leaderboard Builder - session 20260224_213508_objectif_d_velopper_une_strat_gie_de_tra

Objective: Objectif: Développer une stratégie de trading robuste basé sur le breakout et la mean-reversion pour le marché 0GUSDC en 1h à l'aide d'indicateurs peu couramment utilisés, ainsi qu'une approche anti-faux-signaux spécifique.

Les filtres habituels seront inclus dans la stratégie : familles robustes de tendances et combinaisons positives d'indicateurs EMA + OBV, BOLLINGER + STOCHASTIC, EMA pour l'analyse technique à court terme, avec des entrées/sorties ajustées en fonction du contexte.

L'approche de diversification non linéaire sera utilisée : gestion du risque sans seuil fixe (SL/TP adaptatifs basés sur la volatilité), filtre anti-faux-signaux partiellement inversé pour une confirmation contrarianne, et comportements aléatoires imposés tels que mode_inverse avec un seuil supplémentaire.

Les formulations génériques comme "RSI<30/RSI>70" ne seront pas utilisées, mais une hypothèse testable sera proposée pour l'analyse technique à court
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 3 | 4 | -100.00 | -20.000 | -117.82% | -100.00% | 0.80 | 377 | continue | ruined |