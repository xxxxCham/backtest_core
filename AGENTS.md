# Notes pour agents

- Tests rapides: `python -m pytest tests/test_sharpe_ratio.py` (warning attendu si `.pytest_cache` non ecrivable) ou `python -m pytest tests/test_sharpe_ratio.py tests/test_performance_metrics.py` pour couvrir les garde-fous metriques.
- PowerShell: messages `StandardOutputEncoding` des profiles conda/uv apparaissent a chaque commande, bruyants mais inoffensifs.
- Donnees: `data.loader` cherche d'abord `BACKTEST_DATA_DIR` puis `TRADX_DATA_ROOT`, sinon `D:/ThreadX_big` ou `data/sample_data`; signaler toute modification de chemin ou de gestion UTC pour garder la portabilite.
- Metriques: `calculate_metrics` expose `total_return_pct`, `max_drawdown`, `win_rate` deja en pourcentage; le CLI les affiche tels quels (pas de *100 supplementaire). Annualisation basee sur la duree calendrier quand l'index est datetime; volatilite annualisee resample en quotidien si `sharpe_method=daily_resample`.
- Sharpe Ratio: methode par defaut `daily_resample` (252 jours) avec garde >=3 echantillons et rendements non nuls; fournir une `DatetimeIndex` et une equity quand c'est possible.
- Entrypoints: scripts pyproject `backtest-ui` et `backtest-demo` restent casses (pas de `ui.app:main`, module demo manquant); utiliser `python -m backtest_core ...` ou `streamlit run ui/app.py`.
- Encodage: rester en ASCII lors des modifications pour eviter le mojibake deja present.
