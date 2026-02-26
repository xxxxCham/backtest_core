# Leaderboard Builder - session 20260224_214853_objectif_de_strat_gie_de_trading_0gusdc

Objective: Objectif de stratégie de trading #0GUSDC en 1h
- Familles robustes detectees : breakout + momentum + mean-reversion
- Combinaisons indicateurs deja positives : EMA + OBV, BOLLINGER + STOCHASTIC, EMA
- Angles de nouveaute deja prometteurs : curated
- Reutilise au moins un ancrage positif (breakout et momentum), mais MUTER au moins deux dimensions pour eviter la copie (filtre horaire de liquidite, adaptation de regime)

Contraintes de diversification:
- Intègre au moins un axe 'hors sentiers battus' parmi filtre horaire de liquidite, adaptation de regime (trend vs range), gestion du risque non lineaire (SL/TP adaptes a la volatilite), asymetrie long/short (seuils differents).
- Comportements aleatoires imposes pour cette generation: mode_counter_consensus : exiger une confirmation contrarienne partielle, mode_risk_rotation: alterner profile risque serre/large selon volatilite.
- Evite les formulations generiques de type 'RSI<30/RSI>70' sans filtre additionnel.
- Propose une hypothese testable et falsifiable (par
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -20.000 | -1127.92% | -100.00% | 0.48 | 4059 | continue | ruined |
| 3 | 4 | -100.00 | -20.000 | -4720.88% | -100.00% | 0.36 | 14871 | continue | ruined |
| 4 | 6 | -100.00 | -20.000 | -2771.72% | -100.00% | 0.42 | 8408 | stop | ruined |