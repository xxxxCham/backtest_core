# Leaderboard Builder - session 20260225_085627_momentum_invers_filtr_par_microstructur

Objective: [Momentum inversé filtré par microstructure] sur AAVEUSDC 30m. Indicateurs : WILLIAMS_R + ONCHAIN_SMOOTHING + ATR. Entrées : Entrée long si WILLIAMS_R > -20 et ONCHAIN_SMOOTHING haussier en séance européenne uniquement. Sorties : TP à 2xATR, SL trailing basé sur la volatilité récente. Risk management : SL ajusté dynamiquement via ATR, taille de position inversement proportionnelle à la peur/greed du marché.
Status: failed
Best Sharpe: -20.000
Best Continuous Score: -100.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 9.36 | 0.455 | +33.97% | -73.72% | 1.03 | 1067 | continue | high_drawdown |
| 2 | 2 | -100.00 | -20.000 | -372.45% | -100.00% | 0.74 | 1482 | continue | ruined |
| 3 | 4 | -100.00 | -20.000 | -335.84% | -100.00% | 0.76 | 1274 | continue | ruined |
| 4 | 5 | -100.00 | -20.000 | -243.82% | -100.00% | 0.77 | 953 | continue | ruined |