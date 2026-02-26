# Cycle Report - cycle_btc_1h

- Strategy: `ema_cross`
- Data: `D:\.my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet\BTCUSDC_1h.parquet`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'fast_period': 12, 'slow_period': 35, 'k_sl': 3.0}`

## Coarse Sweep - Configurations Interessantes
1. params={'fast_period': 12, 'slow_period': 35, 'k_sl': 3.0} | sharpe=0.162 | return=-36.72% | drawdown=-82.74% | win_rate=24.44% | trades=1248

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
