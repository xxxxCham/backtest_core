# AUDIT backtest_core

## Cartographie
- Points d'entree: `python -m backtest_core` -> `cli.main` -> `cli.commands.*` (orchestration des commandes list/info/backtest/sweep/...).
- Scripts declares: `backtest-demo` (pyproject) pointe vers `demo.quick_test` absent; `backtest-ui` cible `ui.app:main` mais aucune fonction `main` (Streamlit reste accessible via `streamlit run ui/app.py`).
- Flux moteur: CLI/UI/agents -> `backtest.facade.BackendFacade` ou `BacktestEngine` -> `indicators.registry` -> `simulator`/`simulator_fast` -> `performance.calculate_metrics` -> stockage/observabilite.
- Donnees: `data.loader.load_ohlcv` (BACKTEST_DATA_DIR/TRADX_DATA_ROOT/D:/ThreadX_big ou fallback `data/sample_data`) -> normalise OHLCV en float64 avec index UTC.

## Constat de coherence
- Entrypoints: scripts console `backtest-demo` et `backtest-ui` non fonctionnels; aucun wrapper pour lancer Streamlit via l'entree pyproject.
- Data loader: chemins Windows codes en dur + fallback silencieux; timezone forcee en UTC si naif (possible decalage). Pas de controle sur colonnes optionnelles ou volume de donnees.
- Indicateurs/strategies: `indicators.registry` avale les exceptions et retourne `None`, laissant le moteur continuer avec signaux incomplets; validation de parametres tres legere dans `StrategyBase`.
- Moteur/performance:
  - `BacktestEngine` depend du package `performance.device_backend` distinct de `backtest.performance` (risque de collision de nommage et couplage non documente).
  - `max_drawdown_duration_days` convertit les barres en jours en supposant un timeframe minute (approximation pour h/d/w).
  - Annualisation dynamique via `_get_periods_per_year`, mais le Sharpe par defaut resample en quotidien (252) quand l'equity est fournie, ce qui doit etre explicite.
- UI: `ui/app.py` importe directement moteur/loader/strategies sans passer par la facade; aucune fonction `main` pour l'entree `backtest-ui`.
- Observabilite/tests: profils PowerShell bruyants (`StandardOutputEncoding`), creation de `.pytest_cache` parfois refusee (warning Pytest attendu).

## Sharpe Ratio (cas cible)
- Methode par defaut `daily_resample`: resample equity quotidienne puis annualise a 252. Avec peu de trades (1-2) le ratio etait gonfle car aucun seuil d'echantillon minimal.
- Correction appliquee: exigence d'au moins 3 echantillons totaux et 3 rendements non nuls avant calcul, filtre vol minimale conserve, clamp a +/-20 toujours actif, journalisation des retours 0.0 pour donnees insuffisantes.
- Couverture Pytest etendue: cas standard, serie plate, resample court, vol quasi nulle, 1 trade, 3 trades.

## Tests executes
- `python -m pytest tests/test_sharpe_ratio.py` (OK, warning attendu si `.pytest_cache` non ecrivable).
