# Cycle Report - recap_pf_mean_reversion_bollinger_rsi

- Strategy: `mean_reversion_bollinger_rsi`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `profit_factor`
- Selected source: `refine`
- Selected params: `{'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 1}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.703 | return=+29.07% | drawdown=-10.64% | win_rate=51.05% | trades=143
2. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 1.0, 'leverage': 2} | sharpe=0.713 | return=+58.14% | drawdown=-16.28% | win_rate=51.05% | trades=143
3. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 7.0, 'leverage': 1} | sharpe=0.449 | return=+25.88% | drawdown=-13.00% | win_rate=23.61% | trades=144
4. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 7.0, 'leverage': 2} | sharpe=0.484 | return=+51.76% | drawdown=-20.09% | win_rate=23.61% | trades=144
5. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.368 | return=+26.33% | drawdown=-13.73% | win_rate=20.57% | trades=141
6. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 4.0, 'leverage': 1} | sharpe=0.426 | return=+19.30% | drawdown=-12.57% | win_rate=27.21% | trades=147
7. params={'rsi_period': 20, 'stop_atr_mult': 0.5, 'tp_atr_mult': 4.0, 'leverage': 2} | sharpe=0.452 | return=+38.61% | drawdown=-20.11% | win_rate=27.21% | trades=147
8. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.388 | return=+35.94% | drawdown=-19.31% | win_rate=26.55% | trades=113
9. params={'rsi_period': 20, 'stop_atr_mult': 1.6666666666666667, 'tp_atr_mult': 4.0, 'leverage': 1} | sharpe=0.392 | return=+26.15% | drawdown=-17.59% | win_rate=39.17% | trades=120
10. params={'rsi_period': 20, 'stop_atr_mult': 2.8333333333333335, 'tp_atr_mult': 1.0, 'leverage': 1} | sharpe=0.345 | return=+14.93% | drawdown=-15.47% | win_rate=55.32% | trades=141

## Refinement Local - Configurations Interessantes
1. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 1} | sharpe=0.908 | return=+79.00% | drawdown=-7.95% | win_rate=24.06% | trades=133
2. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.941 | return=+158.01% | drawdown=-12.16% | win_rate=24.06% | trades=133
3. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 8.125, 'leverage': 1} | sharpe=0.895 | return=+80.62% | drawdown=-8.50% | win_rate=24.60% | trades=126
4. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.930 | return=+161.24% | drawdown=-12.07% | win_rate=24.60% | trades=126
5. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 7.375, 'leverage': 1} | sharpe=0.910 | return=+70.96% | drawdown=-8.18% | win_rate=24.81% | trades=133
6. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 7.375, 'leverage': 2} | sharpe=0.940 | return=+141.91% | drawdown=-12.64% | win_rate=24.81% | trades=133
7. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 9.625, 'leverage': 1} | sharpe=0.762 | return=+66.31% | drawdown=-8.77% | win_rate=22.31% | trades=130
8. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=0.794 | return=+132.62% | drawdown=-14.17% | win_rate=22.31% | trades=130
9. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 9.25, 'leverage': 1} | sharpe=0.776 | return=+64.85% | drawdown=-8.76% | win_rate=22.90% | trades=131
10. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=0.809 | return=+129.70% | drawdown=-14.15% | win_rate=22.90% | trades=131
11. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 9.625, 'leverage': 1} | sharpe=0.752 | return=+67.92% | drawdown=-8.68% | win_rate=22.76% | trades=123
12. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 9.625, 'leverage': 2} | sharpe=0.786 | return=+135.84% | drawdown=-14.09% | win_rate=22.76% | trades=123
13. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 9.25, 'leverage': 1} | sharpe=0.766 | return=+66.46% | drawdown=-8.67% | win_rate=23.39% | trades=124
14. params={'rsi_period': 20, 'stop_atr_mult': 0.9375, 'tp_atr_mult': 9.25, 'leverage': 2} | sharpe=0.800 | return=+132.92% | drawdown=-14.07% | win_rate=23.39% | trades=124
15. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 6.625, 'leverage': 1} | sharpe=0.831 | return=+60.20% | drawdown=-8.42% | win_rate=24.81% | trades=133
16. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 6.625, 'leverage': 2} | sharpe=0.863 | return=+120.40% | drawdown=-13.29% | win_rate=24.81% | trades=133
17. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 10.0, 'leverage': 1} | sharpe=0.716 | return=+61.43% | drawdown=-9.20% | win_rate=22.31% | trades=130
18. params={'rsi_period': 20, 'stop_atr_mult': 0.7916666666666667, 'tp_atr_mult': 10.0, 'leverage': 2} | sharpe=0.744 | return=+122.86% | drawdown=-15.33% | win_rate=22.31% | trades=130
19. params={'rsi_period': 20, 'stop_atr_mult': 0.6458333333333334, 'tp_atr_mult': 8.125, 'leverage': 1} | sharpe=0.686 | return=+53.15% | drawdown=-11.28% | win_rate=24.26% | trades=136
20. params={'rsi_period': 20, 'stop_atr_mult': 0.6458333333333334, 'tp_atr_mult': 8.125, 'leverage': 2} | sharpe=0.715 | return=+106.29% | drawdown=-18.73% | win_rate=24.26% | trades=136

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
