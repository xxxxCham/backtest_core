# Cycle Report - trend_supertrend

- Strategy: `trend_supertrend`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.068 | return=+519.79% | drawdown=-40.70% | win_rate=38.64% | trades=176
2. params={'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.024 | return=+259.89% | drawdown=-30.68% | win_rate=38.64% | trades=176
3. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.004 | return=+646.60% | drawdown=-31.74% | win_rate=34.30% | trades=172
4. params={'rsi_period': 28, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.989 | return=+474.04% | drawdown=-40.81% | win_rate=39.04% | trades=187
5. params={'rsi_period': 5, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.956 | return=+530.35% | drawdown=-44.42% | win_rate=24.29% | trades=317
6. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.950 | return=+323.30% | drawdown=-26.51% | win_rate=34.30% | trades=172
7. params={'rsi_period': 28, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.943 | return=+431.04% | drawdown=-49.66% | win_rate=51.36% | trades=294
8. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.937 | return=+470.50% | drawdown=-44.31% | win_rate=48.09% | trades=341
9. params={'rsi_period': 28, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.934 | return=+237.02% | drawdown=-32.17% | win_rate=39.04% | trades=187
10. params={'rsi_period': 28, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.925 | return=+215.52% | drawdown=-28.06% | win_rate=51.36% | trades=294
11. params={'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.919 | return=+397.31% | drawdown=-47.13% | win_rate=50.18% | trades=283
12. params={'rsi_period': 28, 'stop_atr_mult': 2.25, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.911 | return=+347.16% | drawdown=-38.94% | win_rate=37.59% | trades=407
13. params={'rsi_period': 28, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.902 | return=+387.75% | drawdown=-33.76% | win_rate=26.78% | trades=295
14. params={'rsi_period': 50, 'stop_atr_mult': 2.25, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.895 | return=+343.48% | drawdown=-41.21% | win_rate=38.24% | trades=387
15. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.891 | return=+235.25% | drawdown=-33.06% | win_rate=48.09% | trades=341
16. params={'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.890 | return=+198.65% | drawdown=-31.00% | win_rate=50.18% | trades=283
17. params={'rsi_period': 5, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.878 | return=+265.17% | drawdown=-39.05% | win_rate=24.29% | trades=317
18. params={'rsi_period': 50, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.876 | return=+380.85% | drawdown=-39.49% | win_rate=26.37% | trades=273
19. params={'rsi_period': 28, 'stop_atr_mult': 2.25, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.867 | return=+173.58% | drawdown=-22.54% | win_rate=37.59% | trades=407
20. params={'rsi_period': 50, 'stop_atr_mult': 2.25, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.856 | return=+171.74% | drawdown=-29.06% | win_rate=38.24% | trades=387

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
