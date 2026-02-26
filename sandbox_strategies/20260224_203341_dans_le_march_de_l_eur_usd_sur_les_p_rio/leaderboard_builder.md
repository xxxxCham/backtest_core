# Leaderboard Builder - session 20260224_203341_dans_le_march_de_l_eur_usd_sur_les_p_rio

Objective: Dans le marché de l'EUR_USD sur les périodes d'1h à court terme (30 min), il est proposé une stratégie composant plusieurs indicateurs, dont la MACD et la ROC en conjonction avec un filtre par volatilité implicite/réelle. Les entrées sont lorsque le ADX indique que les marchandises se déplacent de manière robuste, lorsqu'il y a une confirmation du mouvement (AROON + STOCHASTIC), et quand la Bollinger est à son maximum en bas. La sortie consisterait à ouvrir un position longue avec stop loss justifié par le point ATR et TP ajustables en fonction de l'ampleur des fluctuations du marché.

Pour atteindre cet objectif, il faut être créatif dans les combinaisons inhabituelles d'indicateurs, utiliser un filtre original pour la volatilité (comme par exemple une moyenne mobile suivie de l'indice de volatility implicite ou réelle), et étudier des périodes à court terme afin de profiter pleinement du marché. L'objectif doit être testable, pouvant être falsifié par la suite
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -100.00 | -20.000 | -450.01% | -100.00% | 0.59 | 1213 | continue | ruined |
| 2 | 4 | -100.00 | -20.000 | -1556.65% | -100.00% | 0.55 | 5090 | continue | ruined |