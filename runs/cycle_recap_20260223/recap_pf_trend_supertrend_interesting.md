# Cycle Report - recap_pf_trend_supertrend

- Strategy: `trend_supertrend`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `profit_factor`
- Selected source: `refine`
- Selected params: `{'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 1}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.991 | return=+343.75% | drawdown=-25.27% | win_rate=29.06% | trades=234
2. params={'rsi_period': 5, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.044 | return=+687.51% | drawdown=-28.52% | win_rate=29.06% | trades=234
3. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.950 | return=+323.30% | drawdown=-26.51% | win_rate=34.30% | trades=172
4. params={'rsi_period': 50, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.930 | return=+227.55% | drawdown=-28.61% | win_rate=31.36% | trades=236
5. params={'rsi_period': 35, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.936 | return=+230.70% | drawdown=-26.45% | win_rate=32.13% | trades=249
6. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 7.0, 'leverage': 1} | sharpe=0.904 | return=+272.02% | drawdown=-28.37% | win_rate=41.96% | trades=255

## Refinement Local - Configurations Interessantes
1. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.060 | return=+379.96% | drawdown=-23.42% | win_rate=34.39% | trades=189
2. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.100 | return=+759.91% | drawdown=-29.97% | win_rate=34.39% | trades=189
3. params={'rsi_period': 5, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.021 | return=+370.25% | drawdown=-25.85% | win_rate=35.33% | trades=167
4. params={'rsi_period': 5, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.059 | return=+740.50% | drawdown=-33.19% | win_rate=35.33% | trades=167
5. params={'rsi_period': 5, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.029 | return=+369.07% | drawdown=-25.77% | win_rate=34.81% | trades=181
6. params={'rsi_period': 5, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.071 | return=+738.13% | drawdown=-32.97% | win_rate=34.81% | trades=181
7. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.010 | return=+363.69% | drawdown=-24.86% | win_rate=34.78% | trades=184
8. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.044 | return=+727.39% | drawdown=-32.68% | win_rate=34.78% | trades=184
9. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 1} | sharpe=1.123 | return=+379.51% | drawdown=-19.83% | win_rate=36.63% | trades=202
10. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 8.875, 'leverage': 2} | sharpe=1.166 | return=+759.03% | drawdown=-25.51% | win_rate=36.63% | trades=202
11. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 9.625, 'leverage': 1} | sharpe=0.993 | return=+361.47% | drawdown=-25.88% | win_rate=34.95% | trades=186
12. params={'rsi_period': 7, 'stop_atr_mult': 3.5625, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.024 | return=+722.94% | drawdown=-34.55% | win_rate=34.95% | trades=186
13. params={'rsi_period': 5, 'stop_atr_mult': 3.7083333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.007 | return=+361.03% | drawdown=-26.76% | win_rate=35.23% | trades=176
14. params={'rsi_period': 5, 'stop_atr_mult': 3.7083333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.050 | return=+722.05% | drawdown=-34.23% | win_rate=35.23% | trades=176
15. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 9.625, 'leverage': 1} | sharpe=0.996 | return=+357.74% | drawdown=-23.82% | win_rate=34.36% | trades=195
16. params={'rsi_period': 5, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=1.037 | return=+715.49% | drawdown=-29.91% | win_rate=34.36% | trades=195
17. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=1.061 | return=+373.78% | drawdown=-21.24% | win_rate=33.33% | trades=204
18. params={'rsi_period': 7, 'stop_atr_mult': 3.2708333333333335, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.105 | return=+747.56% | drawdown=-23.90% | win_rate=33.33% | trades=204
19. params={'rsi_period': 7, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.998 | return=+348.61% | drawdown=-21.71% | win_rate=36.69% | trades=169
20. params={'rsi_period': 7, 'stop_atr_mult': 3.8541666666666665, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=1.038 | return=+697.23% | drawdown=-28.61% | win_rate=36.69% | trades=169

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
