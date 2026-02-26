# Cycle Report - cycle_smoke

- Strategy: `ema_cross`
- Data: `data\sample_data\ETHUSDT_1m_sample.csv`
- Metric: `sharpe`
- Selected source: `coarse`
- Selected params: `{'fast_period': 12, 'slow_period': 35, 'k_sl': 3.0}`

## Coarse Sweep - Configurations Interessantes
1. params={'fast_period': 12, 'slow_period': 35, 'k_sl': 3.0} | sharpe=-22.702 | return=-43.97% | drawdown=-55.25% | win_rate=25.37% | trades=201

## Refinement Local - Configurations Interessantes
Refinement non active ou aucune configuration refine disponible.

## Notes
- Les configurations sont ordonnées selon la métrique du cycle.
- Validation OOS et walk-forward restent prioritaires avant usage production.
