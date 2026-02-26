# Leaderboard Builder - session 20260225_004557_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : EMA + OBV + Keltner. Entrées : Momentum >= x et Bollinger Lower Band <= Oanda Price <= Bollinger Upper Band, Sorties : Volume Oscillator > Average True Range. Risk management : Stop Loss = Entry Price - Volatility * 1.5, Take Profit = Entry Price + Volatility * 2.0. Hypothèse testable: "Le mécanisme de décision basé sur le volume oscilleur est plus efficace que la combinaison classique des EMA, OBV et Keltner pour anticiper les changements de tendance dans l'offre de monnaie étrangère.
Status: success
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 89.58 | 1.174 | +34.17% | -22.35% | 1.30 | 45 | accept | target_reached |