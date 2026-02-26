# Cycle Report - cycle_ema_cross_btcusdc_1h

- Strategy: `ema_cross`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'fast_period': 5, 'slow_period': 50, 'k_sl': 3.0}`

## Coarse Sweep - Configurations Interessantes
1. params={'fast_period': 5, 'slow_period': 50, 'k_sl': 3.0} | sharpe=0.804 | return=-109.12% | drawdown=-100.00% | win_rate=21.37% | trades=1544
2. params={'fast_period': 5, 'slow_period': 35, 'k_sl': 1.0} | sharpe=0.763 | return=-311.70% | drawdown=-100.00% | win_rate=18.48% | trades=1986
3. params={'fast_period': 5, 'slow_period': 20, 'k_sl': 3.0} | sharpe=0.530 | return=-640.62% | drawdown=-100.00% | win_rate=20.64% | trades=2708
4. params={'fast_period': 5, 'slow_period': 50, 'k_sl': 5.0} | sharpe=0.485 | return=-116.11% | drawdown=-100.00% | win_rate=21.31% | trades=1544
5. params={'fast_period': 20, 'slow_period': 50, 'k_sl': 1.0} | sharpe=0.443 | return=-99.26% | drawdown=-100.00% | win_rate=16.16% | trades=792
6. params={'fast_period': 20, 'slow_period': 50, 'k_sl': 3.0} | sharpe=0.429 | return=+67.30% | drawdown=-45.73% | win_rate=23.74% | trades=792
7. params={'fast_period': 20, 'slow_period': 35, 'k_sl': 3.0} | sharpe=0.391 | return=+50.80% | drawdown=-46.45% | win_rate=25.16% | trades=954
8. params={'fast_period': 20, 'slow_period': 35, 'k_sl': 5.0} | sharpe=0.356 | return=+36.58% | drawdown=-50.89% | win_rate=25.47% | trades=954
9. params={'fast_period': 20, 'slow_period': 50, 'k_sl': 5.0} | sharpe=0.323 | return=+26.17% | drawdown=-59.04% | win_rate=23.86% | trades=792
10. params={'fast_period': 12, 'slow_period': 35, 'k_sl': 5.0} | sharpe=0.283 | return=-68.63% | drawdown=-98.24% | win_rate=24.60% | trades=1248

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
