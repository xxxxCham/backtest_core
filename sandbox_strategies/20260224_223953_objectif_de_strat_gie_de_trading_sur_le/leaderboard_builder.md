# Leaderboard Builder - session 20260224_223953_objectif_de_strat_gie_de_trading_sur_le

Objective: Objectif de stratégie de trading sur le marché [0GUSDC] en 1 heure :

[Style] Breakout - Utilise EMA, BOLLINGER + STOCHASTIC et EMA pour la detection des breakouts. Indice adx détecte les robustes familles de marche. Combinaison EMA+OBV avec un filtre anti-faux-signaux pour augmenter l'accuracy. Utilise ROC, RSI comme indicateurs additionnels pour une confirmation du trend. Mode_offbeat: Priorise des paires d'indicateurs rares combinées. Mode_risk_rotation: Alternant profiles risque serre/large selon volatilité [1h].

Entrées : Conditions de breakouts (adx > 30, AROON UP>80), conditions de robustes familles detectees (amplitude_hunter > 0.5), conditions d'obv et ROC en hausse, RSI entre 20-70.
Sorties : Entrée des SL/TP adaptes a la volatilité (ATR +1% ou ATR*3). Risk management: Aide à limiter les pertes lorsque le marché change de direction (SL = ATR * 1%) et à maximiser les gains en surve
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | -100.00 | -20.000 | -921.79% | -100.00% | 0.61 | 2578 | continue | ruined |
| 2 | 6 | -100.00 | -20.000 | -294.15% | -100.00% | 0.79 | 1239 | stop | ruined |