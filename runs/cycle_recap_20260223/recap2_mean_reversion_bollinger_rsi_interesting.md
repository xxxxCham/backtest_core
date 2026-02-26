# Cycle Report - recap2_mean_reversion_bollinger_rsi

- Strategy: `mean_reversion_bollinger_rsi`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `refine`
- Selected params: `{'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.713 | return=+58.14% | drawdown=-16.28% | win_rate=51.05% | trades=143
2. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.703 | return=+29.07% | drawdown=-10.64% | win_rate=51.05% | trades=143
3. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 7.0, 'leverage': 2} | sharpe=0.484 | return=+51.76% | drawdown=-20.09% | win_rate=23.61% | trades=144
4. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 4.0, 'leverage': 2} | sharpe=0.452 | return=+38.61% | drawdown=-20.11% | win_rate=27.21% | trades=147
5. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 7.0, 'leverage': 1} | sharpe=0.449 | return=+25.88% | drawdown=-13.00% | win_rate=23.61% | trades=144
6. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 4.0, 'leverage': 1} | sharpe=0.426 | return=+19.30% | drawdown=-12.57% | win_rate=27.21% | trades=147
7. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 4.0, 'leverage': 1} | sharpe=0.392 | return=+26.15% | drawdown=-17.59% | win_rate=39.17% | trades=120
8. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.388 | return=+35.94% | drawdown=-19.31% | win_rate=26.55% | trades=113
9. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.368 | return=+26.33% | drawdown=-13.73% | win_rate=20.57% | trades=141
10. params={'rsi_period': 20, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.345 | return=+14.93% | drawdown=-15.47% | win_rate=55.32% | trades=141

## Refinement Local - Configurations Interessantes
1. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.941 | return=+158.01% | drawdown=-12.16% | win_rate=24.06% | trades=133
2. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 7.375, 'leverage': 2} | sharpe=0.940 | return=+141.91% | drawdown=-12.64% | win_rate=24.81% | trades=133
3. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.930 | return=+161.24% | drawdown=-12.07% | win_rate=24.60% | trades=126
4. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 2.875, 'leverage': 2} | sharpe=0.916 | return=+88.05% | drawdown=-15.97% | win_rate=35.71% | trades=140
5. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 2.875, 'leverage': 2} | sharpe=0.914 | return=+91.70% | drawdown=-16.26% | win_rate=38.35% | trades=133
6. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 7.375, 'leverage': 1} | sharpe=0.910 | return=+70.96% | drawdown=-8.18% | win_rate=24.81% | trades=133
7. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 1} | sharpe=0.908 | return=+79.00% | drawdown=-7.95% | win_rate=24.06% | trades=133
8. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 3.625, 'leverage': 2} | sharpe=0.899 | return=+102.36% | drawdown=-15.15% | win_rate=32.31% | trades=130
9. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 2.875, 'leverage': 1} | sharpe=0.899 | return=+45.85% | drawdown=-11.22% | win_rate=38.35% | trades=133
10. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 8.125, 'leverage': 1} | sharpe=0.895 | return=+80.62% | drawdown=-8.50% | win_rate=24.60% | trades=126
11. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 2.875, 'leverage': 1} | sharpe=0.894 | return=+44.03% | drawdown=-10.94% | win_rate=35.71% | trades=140
12. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 3.625, 'leverage': 1} | sharpe=0.880 | return=+51.18% | drawdown=-10.59% | win_rate=32.31% | trades=130
13. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 6.625, 'leverage': 2} | sharpe=0.863 | return=+120.40% | drawdown=-13.29% | win_rate=24.81% | trades=133
14. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 3.625, 'leverage': 2} | sharpe=0.858 | return=+91.13% | drawdown=-16.01% | win_rate=30.66% | trades=137
15. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 3.625, 'leverage': 1} | sharpe=0.831 | return=+45.57% | drawdown=-11.02% | win_rate=30.66% | trades=137
16. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 6.625, 'leverage': 1} | sharpe=0.831 | return=+60.20% | drawdown=-8.42% | win_rate=24.81% | trades=133
17. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 7.375, 'leverage': 2} | sharpe=0.829 | return=+118.23% | drawdown=-13.43% | win_rate=24.60% | trades=126
18. params={'rsi_period': 19, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.818 | return=+143.07% | drawdown=-14.09% | win_rate=24.31% | trades=144
19. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 5.875, 'leverage': 2} | sharpe=0.813 | return=+103.11% | drawdown=-13.85% | win_rate=24.81% | trades=133
20. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 4.375, 'leverage': 2} | sharpe=0.804 | return=+88.15% | drawdown=-16.66% | win_rate=28.91% | trades=128

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
