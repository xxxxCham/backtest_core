# FIX_PLAN

1) Reparer les points d'entree
- Ajouter un wrapper `main()` pour `ui/app.py` (appel `streamlit run ui/app.py`) ou retirer l'entree `backtest-ui` si non souhaitee.
- Retirer ou remplacer l'entree `backtest-demo` (module `demo.quick_test` absent) pour eviter une console cassée.
- Validation: `python -m backtest_core --help` et lancement manuel `streamlit run ui/app.py`.

2) Durcir le chargement des donnees
- Rendre le chemin principal configurable (doc claire sur BACKTEST_DATA_DIR) et loguer le fallback quand `D:/ThreadX_big` ou `data/sample_data` est utilise.
- Ajouter une verification explicite de timezone (refuser auto-localisation silencieuse) et des colonnes optionnelles (slippage/fees) si necessaire.
- Validation: chargement d'un fichier de demo `python - <<'PY' ...` et verif des logs.

3) Moteur et metriques
- Documenter le coupling avec `performance.device_backend` et prevoir un alias clair pour eviter la confusion avec `backtest.performance`.
- Revoir `max_drawdown_duration_days` pour utiliser `_get_periods_per_year` ou un mapping timeframe->minutes.
- Garder les nouveaux gardes Sharpe (>=3 echantillons et rendements non nuls) et etendre aux autres ratios si besoin.
- Validation: `python -m pytest tests/test_sharpe_ratio.py`.

4) Boucle de verification continue
- Ajouter un job local (pre-commit ou script) qui lance les tests rapides et signale les warnings attendus (profil PowerShell, cache Pytest).
- Commande proposee: `python -m pytest tests/test_sharpe_ratio.py`.
