# FIX_PLAN

NOTE: Plan may be stale. For current operating details, see
`DETAILS_FONCTIONNEMENT.md`.

1) Entrypoints packaging
- Réparer ou retirer les scripts pyproject cassés (`backtest-ui`, `backtest-demo`) en pointant vers une fonction existante ou en basculant sur un wrapper Streamlit explicite.
- Validation: `python -m backtest_core --help` et `streamlit run ui/app.py` (ou nouvelle entrée).

2) Chargement des données
- Journaliser le chemin réellement utilisé (`BACKTEST_DATA_DIR`/`TRADX_DATA_ROOT`/fallback) et rendre le fallback `D:/ThreadX_big` opt-in; option pour conserver la timezone existante au lieu de forcer UTC.
- Ajouter des gardes (tri/déduplication) sur l’index avant normalisation et limiter le scan récursif.
- Validation: `python - <<'PY'\nfrom data.loader import load_ohlcv\nprint(load_ohlcv('BTCUSDT','1h').head())\nPY` avec logs activés.

3) Moteur et indicateurs
- Remonter les erreurs d’indicateurs: `_calculate_indicators` devrait échouer si un calcul renvoie `None` ou lever l’exception d’origine (avec contexte).
- Aligner les paramètres d’annualisation sur la méthode Sharpe choisie (ex: défaut 252 quand `sharpe_method=daily_resample`) et renforcer `_validate_inputs` (index datetime, trié, sans doublons).
- Validation: `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py` puis un run CLI `python -m backtest_core backtest -s ema_cross -d data/sample_data/<file>.parquet -p "{}"`.

4) Observabilité et garde-fous
- Maintenir les tests rapides: `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py` (warning cache possible si `.pytest_cache` non inscriptible).
- Documenter les logs bruyants PowerShell (profil conda/uv) et conserver un check CLI minimal (`python -m backtest_core list strategies`) après chaque modif majeure.
