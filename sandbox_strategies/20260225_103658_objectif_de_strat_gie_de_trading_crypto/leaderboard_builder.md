# Leaderboard Builder - session 20260225_103658_objectif_de_strat_gie_de_trading_crypto

Objective: Objectif de stratégie de trading crypto : Sur le token LSKUSDC (crypto) avec le timeframe 30m, dans une configuration dite "mode_inverse", utilisons la famille robuste de filtres horaires de liquidité pour trouver les zones d'inversion. Dès que l'indice RSI est <30 sur un timescale, nous entrons en short avec SL=ATR*2 et TP = ATR*7 (ou autre configuration non-lineaire). Sur le même timescale, dans une configuration "mode_risk_rotation", nous entreprenons la manoeuvre inverse à partirque l'indice RSI est >70. A chaque entrée de signal, nous enregistrons également l'angle du vortex pour un contrôle supplémentaire sur le contexte. Utilisons aussi l'indicateur standard_deviation pour ajuster les stop losses (SL = 2*Standard Deviation), tout en utilisant la MACD + OBV comme indication de mean-reversion avec une SL=MACD - 100 et TP=MACD+100. En termes d'entrées, nous avons un signal actif lorsque les deux indicateurs RSI sont <30 ou >70 (
Status: success
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 87.49 | 1.108 | +44.03% | -36.67% | 1.26 | 131 | accept | target_reached |