# Leaderboard Builder - session 20260225_001040_voici_un_exemple_de_strat_gie_de_trading

Objective: Voici un exemple de stratégie de trading crypto avec le marché 0GUSDC en 1h et les indicateurs disponibles :

[Momentum] sur 0GUSDC dans 1h. Indicateurs : EMA(3), EMA(8) + RSI(6). Entrées : Crossover entre EMA(8) > EMA(3) ou RSI(50) > RSI(70). Sorties : Long sur crossover, short sur undercrossing. Risk management : Stop Loss à 1% sous le niveau d'ouverture de paiement et TP au sommet du trou de fond dans le temps T pris en compte.

Il est recommandé d'inclure des filtres originaux pour éviter les comportements génériques ou les mises à jour obsolètes, tels que la combinaison de signaux contradictoires avec un vote majoritaire et l'utilisation du mode_inverse afin d'étudier le comportement inverse. De plus, il est également recommandé de fournir une hypothese testable et falsifiable pour permettre une analyse approfondie des stratégies proposées.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -3.94 | 0.577 | +8.49% | -55.80% | 1.06 | 63 | continue | high_drawdown |
| 2 | 7 | -43.78 | 0.221 | -5.82% | -56.05% | 0.97 | 105 | continue | high_drawdown |
| 3 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 4 | 5 | -100.00 | -2.028 | -80.34% | -85.08% | 0.77 | 230 | continue | high_drawdown |