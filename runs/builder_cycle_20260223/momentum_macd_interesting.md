# Cycle Report - momentum_macd

- Strategy: `momentum_macd`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 2}`

## Coarse Sweep - Configurations Interessantes
1. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.805 | return=-270.19% | drawdown=-100.00% | win_rate=14.67% | trades=484
2. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.805 | return=-270.19% | drawdown=-100.00% | win_rate=14.67% | trades=484
3. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.805 | return=-270.19% | drawdown=-100.00% | win_rate=14.67% | trades=484
4. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.805 | return=-270.19% | drawdown=-100.00% | win_rate=14.67% | trades=484
5. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 1} | sharpe=0.779 | return=-184.99% | drawdown=-100.00% | win_rate=25.89% | trades=506
6. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 1} | sharpe=0.779 | return=-184.99% | drawdown=-100.00% | win_rate=25.89% | trades=506
7. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 1} | sharpe=0.779 | return=-184.99% | drawdown=-100.00% | win_rate=25.89% | trades=506
8. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 1} | sharpe=0.779 | return=-184.99% | drawdown=-100.00% | win_rate=25.89% | trades=506
9. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 20.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.682 | return=-282.86% | drawdown=-100.00% | win_rate=33.07% | trades=505
10. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 20.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.682 | return=-282.86% | drawdown=-100.00% | win_rate=33.07% | trades=505
11. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 20.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.682 | return=-282.86% | drawdown=-100.00% | win_rate=33.07% | trades=505
12. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 20.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.682 | return=-282.86% | drawdown=-100.00% | win_rate=33.07% | trades=505
13. params={'macd_fast_period': 5.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.666 | return=+29.54% | drawdown=-94.46% | win_rate=45.48% | trades=310
14. params={'macd_fast_period': 5.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.666 | return=+29.54% | drawdown=-94.46% | win_rate=45.48% | trades=310
15. params={'macd_fast_period': 5.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.666 | return=+29.54% | drawdown=-94.46% | win_rate=45.48% | trades=310
16. params={'macd_fast_period': 5.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 4.0, 'tp_atr_mult': 5.0, 'leverage': 2} | sharpe=0.666 | return=+29.54% | drawdown=-94.46% | win_rate=45.48% | trades=310
17. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.662 | return=-376.82% | drawdown=-100.00% | win_rate=26.88% | trades=506
18. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.662 | return=-376.82% | drawdown=-100.00% | win_rate=26.88% | trades=506
19. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.662 | return=-376.82% | drawdown=-100.00% | win_rate=26.88% | trades=506
20. params={'macd_fast_period': 30.0, 'macd_slow_period': 50.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 5.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 0.5, 'leverage': 2} | sharpe=0.662 | return=-376.82% | drawdown=-100.00% | win_rate=26.88% | trades=506

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
