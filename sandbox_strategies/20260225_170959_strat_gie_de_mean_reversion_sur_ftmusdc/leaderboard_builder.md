# Leaderboard Builder - session 20260225_170959_strat_gie_de_mean_reversion_sur_ftmusdc

Objective: Stratégie de Mean-reversion sur FTMUSDC 30m. Indicateurs : CCI + MFI + ATR. Entrée quand le prix touche la bande extrême de CCI avec MFI en zone de survente/surachat. Take-profit au retour à la bande médiane, stop si MFI continue dans la tendance. Stop-loss dynamique basé sur ATR (1.5x), ratio risk/reward 2.8:1.
Status: success
Best Sharpe: 1.570
Best Continuous Score: 100.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 100.00 | 1.570 | +125.91% | -15.53% | 1.30 | 322 | accept | target_reached |
| 2 | 1 | 49.71 | 0.721 | +48.24% | -47.24% | 1.04 | 802 | continue | approaching_target |
| 3 | 2 | -17.76 | 0.292 | +5.42% | -57.02% | 1.01 | 750 | continue | high_drawdown |
| 4 | 3 | -100.00 | -0.673 | -70.30% | -80.25% | 0.89 | 449 | continue | high_drawdown |