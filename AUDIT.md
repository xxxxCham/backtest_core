# AUDIT backtest_core

NOTE: Historical audit snapshot. For current operating details, see
`DETAILS_FONCTIONNEMENT.md`.

## Cartographie
- Entrées: `python -m backtest_core` -> `cli.main` -> sous-commandes `cli.commands.*` (list/info/backtest/sweep/optuna/export/visualize/validate); UI via `streamlit run ui/app.py`. Scripts pyproject `backtest-ui`/`backtest-demo` restent cassés (module ou main manquants).
- Flux: CLI/UI/agents -> `backtest.engine.BacktestEngine` -> `_calculate_indicators` (registry) -> `StrategyBase.generate_signals` -> `backtest.simulator[_fast]` -> `backtest.performance.calculate_metrics`.
- Données: `data.loader` cherche `BACKTEST_DATA_DIR` > `TRADX_DATA_ROOT` > `D:/ThreadX_big` > `data/sample_data`, normalise OHLCV (colonnes lowercase) et force UTC si index naïf; scan récursif sans journalisation.

## Cohérence et risques
- Data loader: bascule silencieux vers `D:/ThreadX_big`, timezone imposée à UTC même si les horodatages sont déjà au bon fuseau -> risque de décalage; pas de garde sur doublons/sort ordre d’index.
- Indicateurs: `_calculate_indicators` avale toutes les exceptions et renvoie `None`; pas de vérification ultérieure avant `generate_signals`.
- Moteur/métriques: `_get_periods_per_year` reste calé sur la granularité (ex 525 600 pour 1m) alors que le Sharpe par défaut annualise à 252 après resample quotidien; `calculate_metrics` était lié au nombre de barres pour l’annualisation et estimait la durée de drawdown en minutes fixes (1m) et la volatilité sur les retours bruts incluant des zéros.
- UI/agents: import LLM testé dynamiquement (Ollama/OpenAI) sans option explicite de désactivation; `sys.path` muté au runtime.

## Actions réalisées (cohérence métriques/Sharpe)
- `calculate_metrics` annualise désormais à partir de la durée calendrier quand l’index est datetime (fallback sur `periods_per_year` sinon) et mesure la durée de drawdown sur le delta temporel réel; la volatilité annualisée suit le resample quotidien quand `sharpe_method=daily_resample` pour éviter l’écrasement par des retours nuls intraday.
- Suite Sharpe Ratio: logique `daily_resample` conservée (garde >=3 échantillons et >=3 rendements non nuls, clamp à +/-20, min vol annuelle 0.1%) et couverte par tests.

## Tests exécutés
- `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py` (10/10 OK, warning attendu sur `.pytest_cache` non inscriptible).
