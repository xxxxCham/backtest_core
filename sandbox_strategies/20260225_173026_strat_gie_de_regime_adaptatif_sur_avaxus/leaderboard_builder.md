# Leaderboard Builder - session 20260225_173026_strat_gie_de_regime_adaptatif_sur_avaxus

Objective: Stratégie de Regime-adaptatif sur AVAXUSDC 30m. Indicateurs : ADX + ATR + KELTNER. Signal adaptatif : si volatilite elevee (ADX), suivre la cassure ; sinon trader le retour a la moyenne via ATR. Sortie lors d'un changement de regime detecte par ADX. Trailing stop à 2.0x ATR, take-profit à 5.5x ATR.
Status: failed
Best Sharpe: 0.403
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -86.70 | 0.403 | -57.59% | -81.49% | 0.95 | 689 | continue | overtrading |
| 2 | 1 | -100.00 | -20.000 | -86.05% | -100.00% | 0.90 | 545 | continue | ruined |
| 3 | 2 | -100.00 | -20.000 | -585.31% | -100.00% | 0.70 | 1728 | continue | ruined |
| 4 | 4 | -100.00 | -20.000 | -95.51% | -100.00% | 0.91 | 627 | continue | ruined |
| 5 | 6 | -100.00 | -20.000 | -299.98% | -100.00% | 0.81 | 1165 | stop | ruined |