# Leaderboard Builder - session 20260226_081321_objectif_de_strat_gie_de_trading_sur_le

Objective: Objectif de stratégie de trading: Sur le token WALUSDC dans le timeframe 4h, utiliser l'indicateur ROC et l'indicateur Aroon pour une detection des points d'ouverture/fermetures. Ensuite, appliquer la combinaison EMA + OBV via un filtre de volatilité (ATR > 20). Pour le contrôle dynamique du risque, ajuster les SL en fonction de l'amplitude des moves, et TP sur une base Keltner. Utiliser également ROC pour détecter les angles prometteurs avec un angle >= 45. L'ensemble doit être validé par la gestion non-linéaire (SL/TP en fonction de volatilité), la gating par volatilité implicite/réalisée, une approche asymétrique à court et long terme, et un filtre horaire pour liquidité. Pour vérifier l'hypothèse, mesurer le rendement net après une période d'exploitation de 7 jours.
Status: max_iterations
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -0.889 | -20.79% | -40.52% | 0.66 | 15 | continue | needs_work |