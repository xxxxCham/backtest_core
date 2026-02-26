# Cycle Report - recap2_trend_supertrend

- Strategy: `trend_supertrend`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `refine`
- Selected params: `{'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.044 | return=+687.51% | drawdown=-28.52% | win_rate=29.06% | trades=234
2. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.991 | return=+343.75% | drawdown=-25.27% | win_rate=29.06% | trades=234
3. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.950 | return=+323.30% | drawdown=-26.51% | win_rate=34.30% | trades=172
4. params={'rsi_period': 35, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.936 | return=+230.70% | drawdown=-26.45% | win_rate=32.13% | trades=249
5. params={'rsi_period': 50, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.930 | return=+227.55% | drawdown=-28.61% | win_rate=31.36% | trades=236
6. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 7.0, 'leverage': 1} | sharpe=0.904 | return=+272.02% | drawdown=-28.37% | win_rate=41.96% | trades=255

## Refinement Local - Configurations Interessantes
1. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.166 | return=+759.03% | drawdown=-25.51% | win_rate=36.63% | trades=202
2. params={'rsi_period': 5, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.131 | return=+724.39% | drawdown=-27.11% | win_rate=31.67% | trades=221
3. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.124 | return=+710.43% | drawdown=-24.94% | win_rate=34.72% | trades=216
4. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 1} | sharpe=1.123 | return=+379.51% | drawdown=-19.83% | win_rate=36.63% | trades=202
5. params={'rsi_period': 5, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.118 | return=+710.10% | drawdown=-30.22% | win_rate=32.61% | trades=230
6. params={'rsi_period': 7, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.112 | return=+703.66% | drawdown=-30.02% | win_rate=32.23% | trades=242
7. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.105 | return=+747.56% | drawdown=-23.90% | win_rate=33.33% | trades=204
8. params={'rsi_period': 9, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.101 | return=+637.87% | drawdown=-29.49% | win_rate=31.86% | trades=226
9. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.100 | return=+759.91% | drawdown=-29.97% | win_rate=34.39% | trades=189
10. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=1.094 | return=+708.60% | drawdown=-25.90% | win_rate=33.80% | trades=213
11. params={'rsi_period': 9, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.091 | return=+686.62% | drawdown=-28.11% | win_rate=36.07% | trades=219
12. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.085 | return=+740.60% | drawdown=-26.15% | win_rate=33.33% | trades=207
13. params={'rsi_period': 33, 'stop_atr_mult': 2.6875, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.084 | return=+499.52% | drawdown=-32.32% | win_rate=31.30% | trades=246
14. params={'rsi_period': 44, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.078 | return=+501.22% | drawdown=-34.33% | win_rate=34.51% | trades=226
15. params={'rsi_period': 7, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.077 | return=+682.75% | drawdown=-28.21% | win_rate=31.49% | trades=235
16. params={'rsi_period': 5, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.076 | return=+362.20% | drawdown=-24.15% | win_rate=31.67% | trades=221
17. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 8.875, 'leverage': 1} | sharpe=1.075 | return=+355.22% | drawdown=-20.74% | win_rate=34.72% | trades=216
18. params={'rsi_period': 5, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.071 | return=+738.13% | drawdown=-32.97% | win_rate=34.81% | trades=181
19. params={'rsi_period': 37, 'stop_atr_mult': 2.6875, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.069 | return=+520.64% | drawdown=-33.88% | win_rate=32.71% | trades=269
20. params={'rsi_period': 44, 'stop_atr_mult': 2.6875, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.065 | return=+507.94% | drawdown=-31.65% | win_rate=32.82% | trades=262

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
