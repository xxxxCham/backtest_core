# Leaderboard Builder - session 20260226_081931_objectif_sur_le_token_synusdc_dans_le_ti

Objective: Objectif: Sur le token SYNUSDC dans le timeframe 3m, lancer une stratégie composée d'indicateurs multiples tels que EMA, STOCHASTIC et RSI avec conditions de filtration additionnelles (mode_consensus ou mode_inverse), un stop loss adapté à la volatilité sur les prix, ainsi qu'un take profit basé sur le pivot point. Indicateurs : EMA + STOCHASTIC + RSI + ATR. Entrées : filtre horaire de liquidité + confirmation partielle mode_consensus / filtre anti-faux-signais + conflit majoritaires mode_inverse. Sorties : vwap pivot point take profit SL/TP. Diversification: Pour éviter les formulations génériques, incluez des approches non standard comme le filtre horaire de liquidité et le filtrage anti-faux-signais. Même si cela requiert un peu plus d'effort, il est essentiel pour la créativité du design stratégique. Propose une hypothèse testable et falsifiable: Cette stratégie doit être conçue de manière à être examinée clairement sous un certain nombre d'angles, permettant ainsi l
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -1216.16% | -100.00% | 0.42 | 4778 | accept | ruined |