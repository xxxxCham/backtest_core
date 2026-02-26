# Cycle Report - mean_reversion_bollinger_rsi

- Strategy: `mean_reversion_bollinger_rsi`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'rsi_period': 12, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.5, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 12, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.970 | return=-332.24% | drawdown=-100.00% | win_rate=15.93% | trades=609
2. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.898 | return=-602.99% | drawdown=-100.00% | win_rate=29.62% | trades=780
3. params={'rsi_period': 12, 'stop_atr_mult': 2.25, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.777 | return=-304.38% | drawdown=-100.00% | win_rate=29.44% | trades=428
4. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.770 | return=-301.50% | drawdown=-100.00% | win_rate=29.62% | trades=780
5. params={'rsi_period': 12, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.697 | return=-285.93% | drawdown=-100.00% | win_rate=22.67% | trades=397
6. params={'rsi_period': 12, 'stop_atr_mult': 4.0, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.684 | return=-407.95% | drawdown=-100.00% | win_rate=56.56% | trades=633
7. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.609 | return=-1078.25% | drawdown=-100.00% | win_rate=54.15% | trades=1577
8. params={'rsi_period': 5, 'stop_atr_mult': 0.5, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.577 | return=-519.31% | drawdown=-100.00% | win_rate=27.27% | trades=1925
9. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.473 | return=-539.12% | drawdown=-100.00% | win_rate=54.15% | trades=1577
10. params={'rsi_period': 12, 'stop_atr_mult': 4.0, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.458 | return=-182.77% | drawdown=-100.00% | win_rate=28.81% | trades=361
11. params={'rsi_period': 5, 'stop_atr_mult': 0.5, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.449 | return=-362.88% | drawdown=-100.00% | win_rate=11.87% | trades=1356
12. params={'rsi_period': 5, 'stop_atr_mult': 2.25, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.418 | return=-310.12% | drawdown=-100.00% | win_rate=25.49% | trades=875
13. params={'rsi_period': 20, 'stop_atr_mult': 2.25, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.415 | return=+32.65% | drawdown=-24.82% | win_rate=53.90% | trades=141
14. params={'rsi_period': 12, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 1} | sharpe=0.410 | return=-190.58% | drawdown=-100.00% | win_rate=38.00% | trades=400
15. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.395 | return=+52.67% | drawdown=-25.28% | win_rate=20.57% | trades=141
16. params={'rsi_period': 5, 'stop_atr_mult': 0.5, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.392 | return=-1038.63% | drawdown=-100.00% | win_rate=27.27% | trades=1925
17. params={'rsi_period': 20, 'stop_atr_mult': 2.25, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.382 | return=+16.33% | drawdown=-14.44% | win_rate=53.90% | trades=141
18. params={'rsi_period': 5, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.379 | return=-742.66% | drawdown=-100.00% | win_rate=14.26% | trades=1417
19. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.368 | return=+26.33% | drawdown=-13.73% | win_rate=20.57% | trades=141
20. params={'rsi_period': 5, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.5, 'leverage': 2} | sharpe=0.361 | return=-711.15% | drawdown=-100.00% | win_rate=35.36% | trades=854

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
