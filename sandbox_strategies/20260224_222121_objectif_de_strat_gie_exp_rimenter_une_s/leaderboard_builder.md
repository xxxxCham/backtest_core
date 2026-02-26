# Leaderboard Builder - session 20260224_222121_objectif_de_strat_gie_exp_rimenter_une_s

Objective: Objectif de stratégie: Expérimenter une stratégie basée sur l'utilisation d'indicateurs hors sentiers battus dans le marché des cryptoactifs, à l'heure actuelle (1h timeframe). Les indicateurs choisis sont RSI(3), ATR et ROC.

Les conditions de sortie seront un signal long pour une combinaison d'indicateurs (RSI>80 + AROON UP > 75) et un signal court pour une autre combinaison (RSI<20 + RSI<3). Le stop loss sera fixé à la valeur historique maximale de l'ouverture du jour, et le take profit sera atteint en cas d'augmentation significative des prix. La diversité est garantie par l'utilisation de filtres non conventionnels pour éliminer les comportements aléatoires dans le marché (mode_counter_consensus: demander une confirmation partielle contraire, mode_microstructure ajouter un filtre d'heure/horaire et liquidité). La stratégie doit être soumise à des tests rigoureux pour vérifier ses hypothèses.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | 0.500 | -89.45% | -99.89% | 0.82 | 121 | continue | ruined |
| 2 | 4 | -100.00 | -20.000 | -151.51% | -100.00% | 0.86 | 448 | continue | ruined |