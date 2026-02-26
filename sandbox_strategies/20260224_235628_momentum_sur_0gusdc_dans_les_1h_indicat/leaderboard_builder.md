# Leaderboard Builder - session 20260224_235628_momentum_sur_0gusdc_dans_les_1h_indicat

Objective: [Momentum] sur 0GUSDC dans les 1h. Indicateurs : EMA + BOLLINGER + VORTICE. Entrées : EMA(FAST) = 5, BollingerBandLower=90, Vortex(SPEED_OF_VOLUME)>1. Sorties : LONG sur RSI<30 et SHORT sur RSI>70. Risk management : Stop Loss à 2% sous le prix d'entrée pour l'entraînement long, Stop Loss à 2% au-dessus du prix de sortie pour l'entraînement court. Hypothèse testable et falsifiable: "Lorsque les données VORTICE ont une valeur supérieure ou égale à celle d'aujourd'hui, le RSI est sous 30, la stratégie long sera rentable" vs "Lorsque les données VORTICE ont une valeur inférieure ou égale à celle d'aujourd'hui, le RSI est au dessus de 70, la stratégie court sera rentable".
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -108.18% | -100.00% | 0.68 | 246 | continue | ruined |
| 2 | 5 | -100.00 | -20.000 | -225.88% | -100.00% | 0.78 | 1127 | continue | ruined |
| 3 | 6 | -100.00 | -20.000 | -84.69% | -100.00% | 0.87 | 476 | stop | ruined |