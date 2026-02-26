# Cycle Report - cycle_bollinger_best_longe_3i_avaxusdc_15m

- Strategy: `bollinger_best_longe_3i`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\AVAXUSDC_15m.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'bb_period': 200, 'bb_std': 0.5, 'entry_level': -0.2, 'sl_level': -1.5, 'tp_level': 0.3}`

## Coarse Sweep - Configurations Interessantes
1. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': -0.2, 'sl_level': -1.5, 'tp_level': 0.3} | sharpe=-0.084 | return=-142.59% | drawdown=-100.00% | win_rate=4.08% | trades=98
2. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': -0.2, 'sl_level': -1.5, 'tp_level': 4.0} | sharpe=-0.084 | return=-142.59% | drawdown=-100.00% | win_rate=4.08% | trades=98
3. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': -0.2, 'sl_level': 0.1, 'tp_level': 0.3} | sharpe=-0.084 | return=-142.59% | drawdown=-100.00% | win_rate=4.08% | trades=98
4. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': -0.2, 'sl_level': 0.1, 'tp_level': 4.0} | sharpe=-0.084 | return=-142.59% | drawdown=-100.00% | win_rate=4.08% | trades=98
5. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': 0.7, 'sl_level': -1.5, 'tp_level': 0.3} | sharpe=-0.785 | return=-119.01% | drawdown=-100.00% | win_rate=8.24% | trades=85
6. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': 0.7, 'sl_level': -1.5, 'tp_level': 4.0} | sharpe=-0.785 | return=-119.01% | drawdown=-100.00% | win_rate=8.24% | trades=85
7. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': 0.7, 'sl_level': 0.1, 'tp_level': 0.3} | sharpe=-0.785 | return=-119.01% | drawdown=-100.00% | win_rate=8.24% | trades=85
8. params={'bb_period': 200, 'bb_std': 0.5, 'entry_level': 0.7, 'sl_level': 0.1, 'tp_level': 4.0} | sharpe=-0.785 | return=-119.01% | drawdown=-100.00% | win_rate=8.24% | trades=85
9. params={'bb_period': 200, 'bb_std': 6.0, 'entry_level': 0.7, 'sl_level': -1.5, 'tp_level': 0.3} | sharpe=-4.162 | return=-76.94% | drawdown=-77.17% | win_rate=4.35% | trades=46
10. params={'bb_period': 200, 'bb_std': 6.0, 'entry_level': 0.7, 'sl_level': -1.5, 'tp_level': 4.0} | sharpe=-4.162 | return=-76.94% | drawdown=-77.17% | win_rate=4.35% | trades=46

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
