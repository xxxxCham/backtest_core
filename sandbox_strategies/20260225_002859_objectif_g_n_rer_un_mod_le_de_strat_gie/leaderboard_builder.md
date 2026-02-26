# Leaderboard Builder - session 20260225_002859_objectif_g_n_rer_un_mod_le_de_strat_gie

Objective: Objectif: Générer un modèle de stratégie robuste pour le marché en cours (0GUSDC) sur une période d'1h à l'aide d'indicateurs non conventionnels et des filtres horaires. Le but est d'explorer les comportements aléatoires imposés par la gating par volatilité implicite/réelle, le comportement asymétrique long/court (seuils différents) avec un vote majoritaire contraignant et un filtre horaire de liquidité. Cette stratégie devrait éviter les formulations courantes telles que "RSI<30" sans ajout d'autres filtres supplémentaires. Elle doit être testable et falsifiable, avec des sorties spécifiques pour l'entrée et la gestion du risque.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 37.03 | 0.488 | +46.72% | -51.91% | 1.21 | 46 | continue | high_drawdown |
| 2 | 1 | 22.05 | 0.305 | +14.12% | -28.93% | 1.19 | 10 | continue | needs_work |
| 3 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |