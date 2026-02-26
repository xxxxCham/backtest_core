# Cycle Report - recap_trend_supertrend

- Strategy: `trend_supertrend`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `refine`
- Selected params: `{'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.044 | return=+687.51% | drawdown=-28.52% | win_rate=29.06% | trades=234
2. params={'rsi_period': 50, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.024 | return=+259.89% | drawdown=-30.68% | win_rate=38.64% | trades=176
3. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.004 | return=+646.60% | drawdown=-31.74% | win_rate=34.30% | trades=172
4. params={'rsi_period': 35, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.999 | return=+461.41% | drawdown=-35.14% | win_rate=32.13% | trades=249
5. params={'rsi_period': 50, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.994 | return=+455.10% | drawdown=-35.66% | win_rate=31.36% | trades=236
6. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.991 | return=+343.75% | drawdown=-25.27% | win_rate=29.06% | trades=234
7. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.950 | return=+323.30% | drawdown=-26.51% | win_rate=34.30% | trades=172
8. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 7.0, 'leverage': 2} | sharpe=0.944 | return=+544.03% | drawdown=-35.79% | win_rate=41.96% | trades=255
9. params={'rsi_period': 35, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.944 | return=+238.84% | drawdown=-35.99% | win_rate=38.83% | trades=188
10. params={'rsi_period': 35, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.936 | return=+230.70% | drawdown=-26.45% | win_rate=32.13% | trades=249

## Refinement Local - Configurations Interessantes
1. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.166 | return=+759.03% | drawdown=-25.51% | win_rate=36.63% | trades=202
2. params={'rsi_period': 44, 'stop_atr_mult': 4.0, 'tp_atr_mult': 9.25, 'leverage': 1} | sharpe=1.142 | return=+337.11% | drawdown=-29.87% | win_rate=39.77% | trades=176
3. params={'rsi_period': 44, 'stop_atr_mult': 4.0, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=1.131 | return=+674.21% | drawdown=-44.70% | win_rate=39.77% | trades=176
4. params={'rsi_period': 5, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.131 | return=+724.39% | drawdown=-27.11% | win_rate=31.67% | trades=221
5. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.124 | return=+710.43% | drawdown=-24.94% | win_rate=34.72% | trades=216
6. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 1} | sharpe=1.123 | return=+379.51% | drawdown=-19.83% | win_rate=36.63% | trades=202
7. params={'rsi_period': 5, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.118 | return=+710.10% | drawdown=-30.22% | win_rate=32.61% | trades=230
8. params={'rsi_period': 46, 'stop_atr_mult': 4.0, 'tp_atr_mult': 9.25, 'leverage': 1} | sharpe=1.116 | return=+329.65% | drawdown=-32.16% | win_rate=39.43% | trades=175
9. params={'rsi_period': 44, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 9.25, 'leverage': 1} | sharpe=1.115 | return=+318.21% | drawdown=-28.97% | win_rate=39.01% | trades=182
10. params={'rsi_period': 44, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.112 | return=+544.23% | drawdown=-36.71% | win_rate=38.59% | trades=184
11. params={'rsi_period': 44, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=1.112 | return=+636.42% | drawdown=-44.70% | win_rate=39.01% | trades=182
12. params={'rsi_period': 7, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.112 | return=+703.66% | drawdown=-30.02% | win_rate=32.23% | trades=242
13. params={'rsi_period': 46, 'stop_atr_mult': 4.0, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=1.107 | return=+659.30% | drawdown=-45.42% | win_rate=39.43% | trades=175
14. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.105 | return=+747.56% | drawdown=-23.90% | win_rate=33.33% | trades=204
15. params={'rsi_period': 44, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.101 | return=+539.71% | drawdown=-37.85% | win_rate=39.43% | trades=175
16. params={'rsi_period': 9, 'stop_atr_mult': 2.979166666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.101 | return=+637.87% | drawdown=-29.49% | win_rate=31.86% | trades=226
17. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.100 | return=+759.91% | drawdown=-29.97% | win_rate=34.39% | trades=189
18. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=1.094 | return=+708.60% | drawdown=-25.90% | win_rate=33.80% | trades=213
19. params={'rsi_period': 9, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.091 | return=+686.62% | drawdown=-28.11% | win_rate=36.07% | trades=219
20. params={'rsi_period': 46, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.088 | return=+532.43% | drawdown=-39.43% | win_rate=38.12% | trades=181

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
