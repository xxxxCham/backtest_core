# Leaderboard Builder - session 20260225_001630_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + MACD + VAPOR_TREND + PSAR. Entrées : EMA(6) > BOLLINGER(95), MACD différentiel positif, BOLLINGER(80) et STOCHASTIC intersection à 20%, VAPOR_TREND en mode offbeat avec seuil d'entrée haut. Sorties : TP = entrées, SL= EMA(6). Risk management: Stop Loss = -5% tp Entry, Take Profit = +10%. Hypothèse testable et falsifiable : Lorsque les conditions sont satisfaites, la stratégie devrait générer des profits en exploitant le sentiment de croissance sur le marché américain.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -0.639 | -92.75% | -96.19% | 0.71 | 136 | continue | ruined |
| 3 | 5 | -100.00 | -0.263 | -52.98% | -82.69% | 0.91 | 494 | continue | overtrading |
| 4 | 6 | -100.00 | -0.017 | -58.07% | -79.35% | 0.86 | 207 | stop | high_drawdown |