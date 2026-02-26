# Leaderboard Builder - session 20260224_224955_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : EMA + OBV + ST, Aroon, Donchian. Entrées : EMA(length=5) cross ABOVE DONCHIAN upper line, OBV > Bollinger Band(20). Sorties : Momentum(EMA+OBV+ST) above 0, aroon up > bollinger band down. Risk management: SL = Entry Price - 1%, TP = Entry Price + 3%. Hypothese testable et falsifiable: Combination of EMA/OBV/Stochastic and Aroon produces higher momentum signals on the Bitcoin-USD pair in one hour timeframe, with risk management to limit losses.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -100.00 | -0.639 | -92.75% | -96.19% | 0.71 | 136 | continue | ruined |
| 2 | 5 | -100.00 | -0.263 | -52.98% | -82.69% | 0.91 | 494 | continue | overtrading |
| 3 | 6 | -100.00 | -0.017 | -58.07% | -79.35% | 0.86 | 207 | stop | high_drawdown |