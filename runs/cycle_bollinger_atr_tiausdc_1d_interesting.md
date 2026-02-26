# Cycle Report - cycle_bollinger_atr_tiausdc_1d

- Strategy: `bollinger_atr`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\TIAUSDC_1d.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 80, 'k_sl': 0.5}`

## Coarse Sweep - Configurations Interessantes
1. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 80, 'k_sl': 0.5} | sharpe=1.201 | return=+124.86% | drawdown=-21.71% | win_rate=59.09% | trades=22
2. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 80, 'k_sl': 4.0} | sharpe=1.201 | return=+124.86% | drawdown=-21.71% | win_rate=59.09% | trades=22
3. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 0, 'k_sl': 0.5} | sharpe=0.846 | return=-8.85% | drawdown=-87.21% | win_rate=52.17% | trades=92
4. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 0, 'k_sl': 4.0} | sharpe=0.846 | return=-8.85% | drawdown=-87.21% | win_rate=52.17% | trades=92
5. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 28, 'atr_percentile': 0, 'k_sl': 0.5} | sharpe=0.846 | return=-8.85% | drawdown=-87.21% | win_rate=52.17% | trades=92
6. params={'bb_period': 10, 'bb_std': 4.0, 'entry_z': 0.5, 'atr_period': 28, 'atr_percentile': 0, 'k_sl': 4.0} | sharpe=0.846 | return=-8.85% | drawdown=-87.21% | win_rate=52.17% | trades=92
7. params={'bb_period': 50, 'bb_std': 1.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 0, 'k_sl': 0.5} | sharpe=0.696 | return=+46.43% | drawdown=-37.52% | win_rate=51.35% | trades=37
8. params={'bb_period': 50, 'bb_std': 1.0, 'entry_z': 0.5, 'atr_period': 7, 'atr_percentile': 0, 'k_sl': 4.0} | sharpe=0.696 | return=+46.43% | drawdown=-37.52% | win_rate=51.35% | trades=37
9. params={'bb_period': 50, 'bb_std': 1.0, 'entry_z': 0.5, 'atr_period': 28, 'atr_percentile': 0, 'k_sl': 0.5} | sharpe=0.696 | return=+46.43% | drawdown=-37.52% | win_rate=51.35% | trades=37
10. params={'bb_period': 50, 'bb_std': 1.0, 'entry_z': 0.5, 'atr_period': 28, 'atr_percentile': 0, 'k_sl': 4.0} | sharpe=0.696 | return=+46.43% | drawdown=-37.52% | win_rate=51.35% | trades=37

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
