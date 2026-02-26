# Leaderboard Builder - session 20260224_225542_objectif_de_strat_gie_sur_le_march_btc_u

Objective: Objectif de Stratégie: Sur le marché BTC/USD en 1h. Utiliser l'EMA5 et la MACD avec un seuil d'entraînement de 0,9 pour une combinaison positive, ainsi que les données sur les volumes (volume_oscillator) et l'angle de direction (roc).

Les entrées seront: EMA5 = 12, MACD(12,26,9) = +0.87, volume_oscillator (> 3 std dev) > -3 sigmas, roc (> 0.04) avec un seuil d'entraînement de 0.9 pour les entrées.
Les sorties seront: entropie(donchian)>15 (pour montrer une tendance forte), Aroon(-75)<-25 (pour indiquer que le marché avance) et atr<1%.
Pour la gestion du risque, on utilisera un SL à 0.9% pour les positions longues et TP à 0.8% pour les positions courtes.
La hypothese testable est qu'une combinaison de deux indicateurs positifs peut améliorer le rendement en minimisant la volatilité des gains/pertes.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 2 | 1 | -100.00 | -20.000 | -135.54% | -100.00% | 0.69 | 175 | continue | ruined |