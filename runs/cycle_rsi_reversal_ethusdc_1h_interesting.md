# Cycle Report - cycle_rsi_reversal_ethusdc_1h

- Strategy: `rsi_reversal`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\ETHUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'rsi_period': 5, 'oversold_level': 40, 'overbought_level': 60}`

## Coarse Sweep - Configurations Interessantes
1. params={'rsi_period': 5, 'oversold_level': 40, 'overbought_level': 60} | sharpe=0.849 | return=-1739.36% | drawdown=-100.00% | win_rate=48.61% | trades=5194
2. params={'rsi_period': 18, 'oversold_level': 25, 'overbought_level': 60} | sharpe=0.797 | return=-488.85% | drawdown=-100.00% | win_rate=13.81% | trades=717
3. params={'rsi_period': 30, 'oversold_level': 40, 'overbought_level': 75} | sharpe=0.731 | return=+338.45% | drawdown=-40.99% | win_rate=18.59% | trades=355
4. params={'rsi_period': 30, 'oversold_level': 25, 'overbought_level': 75} | sharpe=0.459 | return=-103.03% | drawdown=-100.00% | win_rate=12.64% | trades=87
5. params={'rsi_period': 18, 'oversold_level': 40, 'overbought_level': 90} | sharpe=0.450 | return=+1734.11% | drawdown=-34.44% | win_rate=14.29% | trades=21
6. params={'rsi_period': 18, 'oversold_level': 40, 'overbought_level': 60} | sharpe=0.420 | return=-539.35% | drawdown=-100.00% | win_rate=27.24% | trades=1630
7. params={'rsi_period': 18, 'oversold_level': 10, 'overbought_level': 60} | sharpe=0.417 | return=-313.92% | drawdown=-100.00% | win_rate=3.47% | trades=202
8. params={'rsi_period': 5, 'oversold_level': 25, 'overbought_level': 60} | sharpe=0.415 | return=-1139.49% | drawdown=-100.00% | win_rate=40.98% | trades=3438
9. params={'rsi_period': 5, 'oversold_level': 40, 'overbought_level': 90} | sharpe=0.412 | return=-227.23% | drawdown=-100.00% | win_rate=26.68% | trades=1503
10. params={'rsi_period': 30, 'oversold_level': 25, 'overbought_level': 60} | sharpe=0.334 | return=-305.77% | drawdown=-100.00% | win_rate=10.67% | trades=403

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
