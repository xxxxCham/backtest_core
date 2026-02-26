# Leaderboard Builder - session 20260224_222857_objectif_de_strat_gie_de_trading_g_n_rer

Objective: Objectif de stratégie de trading : Générer un modèle d'apprentissage profond pour la prediction de paires crypto en utilisant l'indicateur AROON, avec une attention particulière aux variations d'asymetrie long/court à partir des filtres 'mode_risk_rotation'. Étudier le comportement de la stratégie dans différents timeframe (1h, 1h et daily) pour identifier les tendances cycliques. Identifier une combinaison significative d'indicateurs non lineaires pour la gestion du risque, incluant EMA + OBV, BOLLINGER + STOCHASTIC et l'indicateur AROON. Utiliser des conditions de signal additionnelles basées sur les données historiques (taux d'utilisation moyen) pour tester la robustesse du modèle dans différents market conditions.

Risk management : Établir un SL et TP approprié à chaque paires crypto en fonction de l'amplitude des fluctuations, ainsi qu'un stoploss unique pour les positions ouvertes sur plusieurs timeframe pour éviter les pertes gênantes dû aux variations diffuses.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -100.00 | -20.000 | -135.54% | -100.00% | 0.69 | 175 | continue | ruined |
| 3 | 5 | -100.00 | -20.000 | -367.36% | -100.00% | 0.70 | 795 | continue | ruined |