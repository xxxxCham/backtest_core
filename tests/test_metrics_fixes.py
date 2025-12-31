"""
Module-ID: tests.test_metrics_fixes

Purpose: Tester correctifs métriques (drawdown clamped -100%, ruine, cas normaux).

Role in pipeline: testing

Key components: test_drawdown_clamped_at_minus_100, test_drawdown_normal_case

Inputs: Equity Series, drawdown_series(), max_drawdown()

Outputs: Drawdown <= 0 et >= -1.0 (clamped -100%)

Dependencies: pytest, numpy, pandas, backtest.performance

Conventions: Drawdown toujours [-1.0, 0]; ruine detectée (equité <0).

Read-if: Modification drawdown calculation.

Skip-if: Tests metrics non critiques.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.performance import (
    calculate_metrics,
    drawdown_series,
    max_drawdown,
)


def test_drawdown_clamped_at_minus_100_percent():
    """Le drawdown ne doit jamais dépasser -100%."""
    # Cas de ruine : équité négative
    equity = pd.Series([10000, 5000, 2000, -500, -1000])

    dd = drawdown_series(equity)
    max_dd = max_drawdown(equity)

    # Aucun drawdown ne doit être < -100%
    assert (dd >= -1.0).all(), f"Drawdown < -100% détecté: {dd.min()}"
    assert max_dd >= -1.0, f"Max drawdown < -100%: {max_dd}"
    assert max_dd == -1.0, f"Max drawdown devrait être -100%, got {max_dd}"


def test_drawdown_normal_case():
    """Le drawdown fonctionne correctement dans les cas normaux."""
    equity = pd.Series([10000, 10500, 10200, 11000, 10800])

    dd = drawdown_series(equity)
    max_dd = max_drawdown(equity)

    # Vérifier les valeurs
    expected_dd = pd.Series([0.0, 0.0, -2.857142857142857e-02, 0.0, -1.818181818181818e-02])
    pd.testing.assert_series_equal(dd, expected_dd, rtol=1e-5)

    assert max_dd < 0, "Max drawdown devrait être négatif"
    assert max_dd > -0.1, "Max drawdown trop grand pour capital stable"


def test_account_ruined_flag_set():
    """Le flag account_ruined doit être activé si équité <= 0."""
    # Cas de ruine
    equity_ruined = pd.Series([10000, 5000, 2000, -500, -1000])
    returns_ruined = equity_ruined.pct_change().fillna(0)
    trades_df = pd.DataFrame({"pnl": [100, -200, -300]})

    metrics = calculate_metrics(
        equity=equity_ruined,
        returns=returns_ruined,
        trades_df=trades_df,
        initial_capital=10000.0,
    )

    assert metrics["account_ruined"] is True, "Flag account_ruined devrait être True"
    assert metrics["min_equity"] == -1000, f"Min equity incorrect: {metrics['min_equity']}"

    # Cas normal (pas de ruine)
    equity_normal = pd.Series([10000, 10500, 10200, 11000, 10800])
    returns_normal = equity_normal.pct_change().fillna(0)

    metrics_normal = calculate_metrics(
        equity=equity_normal,
        returns=returns_normal,
        trades_df=trades_df,
        initial_capital=10000.0,
    )

    assert metrics_normal["account_ruined"] is False, "Flag account_ruined devrait être False"


def test_drawdown_series_empty():
    """drawdown_series gère correctement les séries vides."""
    empty = pd.Series([], dtype=np.float64)

    dd = drawdown_series(empty)

    assert dd.empty, "Drawdown d'une série vide devrait être vide"
    assert len(dd) == 0


def test_max_drawdown_no_loss():
    """Max drawdown devrait être 0 si pas de perte."""
    equity = pd.Series([10000, 10500, 11000, 11500, 12000])  # Seulement gains

    max_dd = max_drawdown(equity)

    assert max_dd == 0.0, f"Max drawdown devrait être 0, got {max_dd}"


def test_drawdown_partial_recovery():
    """Test drawdown avec récupération partielle."""
    equity = pd.Series([10000, 12000, 9000, 10500, 11000])

    max_dd = max_drawdown(equity)

    # Le plus bas est à 9000 depuis le pic de 12000
    expected_max_dd = (9000 - 12000) / 12000  # = -0.25
    assert abs(max_dd - expected_max_dd) < 1e-6, f"Max DD attendu {expected_max_dd}, got {max_dd}"


def test_metrics_with_ruined_account():
    """Calcul complet des métriques avec ruine du compte."""
    # Générer un scénario de ruine
    np.random.seed(42)
    equity = pd.Series([10000, 8000, 5000, 2000, 500, -100])
    returns = equity.pct_change().fillna(0)

    trades_df = pd.DataFrame({
        "pnl": [-2000, -3000, -3000, -1500, -600],
        "entry_ts": pd.date_range("2024-01-01", periods=5, freq="D"),
        "exit_ts": pd.date_range("2024-01-02", periods=5, freq="D"),
    })

    metrics = calculate_metrics(
        equity=equity,
        returns=returns,
        trades_df=trades_df,
        initial_capital=10000.0,
    )

    # Vérifications
    assert metrics["account_ruined"] is True
    assert metrics["min_equity"] == -100
    assert metrics["max_drawdown"] == -100.0  # Clampé
    assert metrics["total_pnl"] == -10100  # 10000 → -100
    assert metrics["total_return_pct"] == -101.0  # -101%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
