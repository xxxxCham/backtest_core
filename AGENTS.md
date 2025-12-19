# Notes pour agents

- Tests rapides: `python -m pytest tests/test_sharpe_ratio.py` (warning attendu si `.pytest_cache` non ecrivable).
- PowerShell: messages `StandardOutputEncoding` des profiles conda/uv visibles a chaque commande, ils sont bruyants mais inoffensifs.
- Donnees: `data.loader` cherche d'abord `BACKTEST_DATA_DIR` puis `TRADX_DATA_ROOT`, sinon chemins Windows `D:/ThreadX_big` ou `data/sample_data`. Signaler toute modification de chemin pour garder la portabilite.
- Sharpe Ratio: methode par defaut `daily_resample` (252 jours) avec garde >=3 echantillons et rendements non nuls; fournir une `DatetimeIndex` et une equity quand c'est possible.
- Entrypoints: les scripts pyproject `backtest-ui` et `backtest-demo` sont actuellement cassés (pas de `ui.app:main`, module demo manquant); utiliser `python -m backtest_core ...` ou `streamlit run ui/app.py`.
- Encodage: rester en ASCII lors des modifications pour eviter le mojibake deja present.
