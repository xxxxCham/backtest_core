# CHANGELOG_AUDIT

- Ajout d'un garde supplementaire dans `backtest/performance.py` pour le Sharpe Ratio (minimum 3 echantillons et 3 rendements non nuls, logs de fallback conserves).
- Extension de `tests/test_sharpe_ratio.py` avec des cas 0 trade, 1 trade, 3 trades et vol quasi nulle pour verrouiller le comportement du calcul.
- Tests executes: `python -m pytest tests/test_sharpe_ratio.py` (OK, warning attendu sur `.pytest_cache` en environnement restreint).
