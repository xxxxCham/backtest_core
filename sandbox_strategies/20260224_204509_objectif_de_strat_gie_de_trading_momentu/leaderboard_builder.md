# Leaderboard Builder - session 20260224_204509_objectif_de_strat_gie_de_trading_momentu

Objective: Objectif de stratégie de trading:
[Momentum] sur [0GUSDC] en 1h. Indicateurs : EMA + OBV + Keltner, Angles de nouveauté deja prometteurs. Entrées : mode_risk_rotation; filtre horaire de liquidite. Sorties : SL/TP = TP - 0,2% ; TP = RSI<30

Sois créatif: Utilise l'EMA pour déterminer les tendances dans le temps (breakout) et la combinaison entre EMA + OBV pour repérer des signaux de momentum. L'anglais du nouveauté est également utilisé pour trouver les signes prometteurs, mais en plus il apporte une perspective alternative sur les angles d'analyse.
Utilisez le mode_risk_rotation afin de varier entre des stratégies à risque serre (utilisation du filtre horaire) et large (ne pas utiliser ce filtre). Cela permet d'adapter la gestion de risque en fonction de l'environnement volatil.
L'objectif présente une hypothèse testable, car il est possible d'analyser les résultats et de vérifier si le trading sur 0GUS
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -20.000 | -223.97% | -100.00% | 0.67 | 765 | continue | ruined |
| 3 | 6 | -100.00 | -20.000 | -1160.98% | -100.00% | 0.49 | 3503 | stop | ruined |