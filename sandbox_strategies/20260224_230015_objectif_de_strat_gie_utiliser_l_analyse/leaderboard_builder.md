# Leaderboard Builder - session 20260224_230015_objectif_de_strat_gie_utiliser_l_analyse

Objective: Objectif de stratégie : "Utiliser l'analyse multifactorielle en composant EMA avec ROC sur le timeframe 1h pour détecter des breakouts et des tendances, combinant un filtre anti-faux-signaux (confirmation inverse partielle) pour minimiser les faux signal. La stratégie va inclure une gestion du risque non lineaire en utilisant RSI<30 avec SL/TP adaptes a la volatilite sur le timeframe 1h, et alternera entre profils de risque serre et large selon l'amplitude des fluctuations."

Entrée : [market = 0GUSDC], [timeframe = 1h]
Sortie: [trend detection with EMA + ROC].
Risk management: [SL/TP adapted to volatility in the timeframe 1h, RSI<30 with SL/TP adjusted for volatiltiy].

Hypothèse testable : Les breakouts avec la combinaison d'indicateurs (EMA+ROC) sur le timeframe 1h sur market = 0GUSDC peuvent anticiper des tendances en matière de trading crypto.
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 5 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 2 | -100.00 | -20.000 | -117.40% | -100.00% | 0.78 | 297 | continue | ruined |
| 4 | 6 | -100.00 | -20.000 | -426.38% | -100.00% | 0.74 | 1328 | stop | ruined |