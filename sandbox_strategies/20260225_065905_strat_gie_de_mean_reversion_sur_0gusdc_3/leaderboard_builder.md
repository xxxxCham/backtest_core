# Leaderboard Builder - session 20260225_065905_strat_gie_de_mean_reversion_sur_0gusdc_3

Objective: Stratégie de Mean-reversion sur 0GUSDC 30m. Indicateurs : OBV + MFI + RSI + ATR. Entrée quand le prix touche la bande extrême de OBV avec MFI en zone de survente/surachat. Sortie quand le prix revient vers la moyenne (OBV neutre). Stop-loss = 1.1x ATR, take-profit = 2.4x ATR.
Status: success
Best Sharpe: 1.568
Best Continuous Score: 100.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 100.00 | 1.568 | +45.21% | -23.67% | 1.18 | 144 | accept | target_reached |
| 2 | 4 | 7.94 | 0.344 | +0.30% | -37.50% | 1.00 | 175 | continue | marginal |
| 3 | 3 | -26.78 | 0.186 | -9.68% | -41.87% | 0.98 | 228 | continue | needs_work |
| 4 | 2 | -42.77 | 0.078 | -16.15% | -45.21% | 0.96 | 240 | continue | needs_work |
| 5 | 1 | -100.00 | -1.882 | -21.91% | -23.03% | 0.41 | 22 | continue | wrong_direction |