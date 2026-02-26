# Cycle Report - recap_momentum_macd

- Strategy: `momentum_macd`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 1}`

## Coarse Sweep - Configurations Interessantes
1. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 1.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 1} | sharpe=0.206 | return=+11.33% | drawdown=-30.37% | win_rate=21.88% | trades=512
2. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 5.0, 'bollinger_period': 10.0, 'bollinger_std_dev': 3.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 1} | sharpe=0.206 | return=+11.33% | drawdown=-30.37% | win_rate=21.88% | trades=512
3. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 1.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 1} | sharpe=0.206 | return=+11.33% | drawdown=-30.37% | win_rate=21.88% | trades=512
4. params={'macd_fast_period': 5.0, 'macd_slow_period': 10.0, 'macd_signal_period': 5.0, 'bollinger_period': 30.0, 'bollinger_std_dev': 3.0, 'atr_period': 30.0, 'stop_atr_mult': 0.5, 'tp_atr_mult': 5.0, 'leverage': 1} | sharpe=0.206 | return=+11.33% | drawdown=-30.37% | win_rate=21.88% | trades=512

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
