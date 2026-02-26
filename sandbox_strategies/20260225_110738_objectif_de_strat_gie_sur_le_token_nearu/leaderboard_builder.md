# Leaderboard Builder - session 20260225_110738_objectif_de_strat_gie_sur_le_token_nearu

Objective: Objectif de stratégie : Sur le token NEARUSDC dans la timeframe 4h, utiliser l'indicateur ASYMETRIE_LONG/SHORT avec seuils differents pour une combinaison de signaux contradictoires, un filtre anti-faux-signaux (confirmation inverse partielle) et des entrées basées sur le comportement aleatoire. Intègre l'indicateur EMA en plus d'un autre indicateur pour une analyse de tendance avec confirmation par le STOCHASTIC, ainsi que la combinaison BOLLINGER + STOCHASTIC pour un contraste entre les seuils. Augmentez la diversité à l'aide du mode_inverse en testant une logique inversee puis filtrage par régime et le mode_counter_consensus pour exiger une confirmation partielle. Evitez les formulations génériques comme "RSI<30/RSI>70" sans filtre additionnel, de sorte que votre objectif soit testable et falsifiable. Pour le risk management, utilise un stop loss (SL) à {stop_loss} et une take profit (TP) à {take_profit}.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | -94.35 | 0.171 | -43.88% | -84.30% | 0.95 | 333 | stop | overtrading |
| 2 | 1 | -100.00 | -20.000 | -99.91% | -100.00% | 0.78 | 108 | continue | ruined |