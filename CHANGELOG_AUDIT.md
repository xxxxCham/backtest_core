# CHANGELOG_AUDIT

NOTE: Historical changelog for the audit work. For current operating details,
see `DETAILS_FONCTIONNEMENT.md`.

- Ajusté `backtest/performance.calculate_metrics` pour annualiser sur la durée calendrier quand l’index est datetime, resampler la volatilité en quotidien si `sharpe_method=daily_resample`, et mesurer la durée de drawdown via les timestamps plutôt qu’en minutes fixes.
- Ajouté `tests/test_performance_metrics.py` pour verrouiller l’annualisation calendrier, la volatilité resamplée et la durée de drawdown basée sur l’horloge; fréquences `min`/`h` utilisées pour éviter les dépréciations.
- Tests exécutés: `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py` (10/10 OK, warning `.pytest_cache` non inscriptible attendu sous Windows).
