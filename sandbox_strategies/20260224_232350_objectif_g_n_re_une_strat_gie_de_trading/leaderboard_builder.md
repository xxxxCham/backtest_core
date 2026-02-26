# Leaderboard Builder - session 20260224_232350_objectif_g_n_re_une_strat_gie_de_trading

Objective: OBJECTIF: Génère une stratégie de trading crypto à l'aide d'indicateurs non standard (mode_offbeat), comportements aleatoires pour diversifier les entrées et sorties, et des filtres originaux comme le filtrage horaire de liquidité pour la gestion du risque. La stratégie doit tester une hypothèse testable et falsifiable en utilisant un ensemble d'indicateurs rares combinés (mode_risk_rotation: alterner entre profile risque serre/large selon volatilite).

Stratégie : Sur le marché 0GUSDC dans l'horaire de liquidité, les indicateurs suivants seront utilisés : EMA + PIVOT_POINTS (mode_offbeat), BOLLINGER + STOCHASTIC (mode_risk_rotation) et EMA. Les entrées pour cette stratégie sont la volatilité implicite/réelle, le filtre horaire de liquidité (30 minutes - 4 heures) et l'adaptation de régime (trend vs range). La gestion du risque sera gérée par des SL à un niveau de volatilité donné.

L'objectif est d'explorer des approches
Status: max_iterations
Best Sharpe: 0.801
Best Continuous Score: 38.15

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 2 | 2 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 3 | 3 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 4 | 4 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 5 | 5 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 6 | 6 | 38.15 | 0.685 | +15.99% | -27.47% | 1.16 | 19 | continue | approaching_target |
| 7 | 7 | -83.68 | 0.801 | -57.84% | -79.89% | 0.84 | 31 | continue | high_drawdown |