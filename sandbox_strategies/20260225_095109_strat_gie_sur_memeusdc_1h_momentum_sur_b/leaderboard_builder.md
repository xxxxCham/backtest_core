# Leaderboard Builder - session 20260225_095109_strat_gie_sur_memeusdc_1h_momentum_sur_b

Objective: Stratégie sur MEMEUSDC 1h. **[Momentum] sur BTCUSDT. Indicateurs : EMA + RSI + ATR. Entrées : Vitesse de croissance > 0, Momentum > 50. Sorties : S&R. Risk management : SL=Entry - 2% * Entry Price, TP = StrikePrice + 1%.** Le proposition utilise le filtre "vitesse de croissance" pour choisir les moments où l'action est en phase avec la tendance et ajoute un second filtre supplémentaire (la valeur RSI(9)) qui indique si l'actif se trouve dans une situation dite "overbought" ou non. En consommant ces deux informations, le trading system peut avoir une meilleure perception des zones de support et resistance ainsi que la force de la tendance à un moment donné pour potentiellement améliorer les taux de succès d'une stratégie de trading. Il est important de noter qu'il n'est pas recommandé d'utiliser le même filtre de base pour toutes les entrées, ainsi l'algorithme devrait être flexible à divers conditions du marché ou des évènements spécifiques. Pour minimiser la perte
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -64.23 | 0.086 | -15.95% | -59.33% | 0.95 | 169 | continue | high_drawdown |
| 3 | 9 | -100.00 | -20.000 | -132.75% | -100.00% | 0.83 | 382 | stop | ruined |