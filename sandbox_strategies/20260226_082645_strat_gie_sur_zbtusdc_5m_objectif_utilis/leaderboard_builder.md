# Leaderboard Builder - session 20260226_082645_strat_gie_sur_zbtusdc_5m_objectif_utilis

Objective: Stratégie sur ZBTUSDC 5m. Objectif : Utiliser les indicateurs `bollinger + stoch_rsi` pour un breakout sur le signal de la courbe Bollinger et l'intersection avec une zone d'oscillation basse du RSI, dans le token `DOGEUSDT` pendant une période de. Le stop loss est fixé à 20% sous le coin inférieur de la zone des oscillations (support Bollinger), tandis que le take profit est d'un peu plus de 30%. Les conditions sont les suivantes : un signal bollinger rouge, et un intersection avec une zone basse du RSI. Pour l'approche de diversification, nous allons utiliser des filtres anti-faux signaux (confirmation inverse partielle) pour limiter le risque d'erreur. Nous choisirons également d'explorer un comportement aléatoire avec la mise en place d'une stratégie de rotation de risques alternant entre les profils de gros et petits plaisir, selon l'oscillation des prix (cette fonctionnalité n'est pas disponible pour le moment). Enfin, nous voulons tester une hypothèse testable en écartant la proposition générique 'RSI<30/RSI>70', en
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -201.20% | -100.00% | 0.60 | 675 | stop | ruined |