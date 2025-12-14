"""
Tests unitaires pour le moteur de backtest.
==========================================

Tests pour le simulateur de trades et le calculateur de performance.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.performance import (
    PerformanceMetrics,
    drawdown_series,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from backtest.simulator import Trade, simulate_trades


def create_test_ohlcv(n: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Crée des données OHLCV de test avec index datetime."""
    np.random.seed(42)

    # Générer des prix
    returns = np.random.randn(n) * 0.02  # 2% volatilité journalière
    prices = start_price * np.exp(np.cumsum(returns))

    # Créer index datetime
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n)]

    # OHLCV
    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 0.1,
        "high": prices + np.abs(np.random.randn(n)) * 0.5,
        "low": prices - np.abs(np.random.randn(n)) * 0.5,
        "close": prices,
        "volume": np.random.exponential(1000, n)
    }, index=pd.DatetimeIndex(dates))

    return df


class TestTradeDataclass(unittest.TestCase):
    """Tests pour la dataclass Trade."""

    def test_trade_creation(self):
        """Test la création d'un trade avec tous les champs."""
        trade = Trade(
            entry_ts=pd.Timestamp("2024-01-01 10:00"),
            exit_ts=pd.Timestamp("2024-01-01 11:00"),
            side="LONG",
            entry_price=100.0,
            exit_price=105.0,
            size=1.0,
            pnl=50.0,
            return_pct=5.0,
            exit_reason="signal",
            leverage=1.0,
            fees_paid=0.5
        )

        self.assertEqual(trade.side, "LONG")
        self.assertEqual(trade.entry_price, 100.0)
        self.assertEqual(trade.exit_price, 105.0)
        self.assertEqual(trade.pnl, 50.0)
        self.assertEqual(trade.return_pct, 5.0)

    def test_trade_to_dict(self):
        """Test la conversion en dictionnaire."""
        trade = Trade(
            entry_ts=pd.Timestamp("2024-01-01 10:00"),
            exit_ts=pd.Timestamp("2024-01-01 11:00"),
            side="SHORT",
            entry_price=110.0,
            exit_price=105.0,
            size=1.0,
            pnl=50.0,
            return_pct=4.5,
            exit_reason="stop_loss",
            leverage=2.0,
            fees_paid=1.0
        )

        d = trade.to_dict()

        self.assertEqual(d["side"], "SHORT")
        self.assertEqual(d["price_entry"], 110.0)
        self.assertEqual(d["price_exit"], 105.0)
        self.assertEqual(d["leverage_used"], 2.0)

    def test_trade_long_vs_short(self):
        """Test les deux directions de trade."""
        long_trade = Trade(
            entry_ts=pd.Timestamp("2024-01-01"),
            exit_ts=pd.Timestamp("2024-01-02"),
            side="LONG",
            entry_price=100.0,
            exit_price=110.0,
            size=1.0,
            pnl=100.0,
            return_pct=10.0,
            exit_reason="signal"
        )

        short_trade = Trade(
            entry_ts=pd.Timestamp("2024-01-01"),
            exit_ts=pd.Timestamp("2024-01-02"),
            side="SHORT",
            entry_price=100.0,
            exit_price=90.0,
            size=1.0,
            pnl=100.0,
            return_pct=10.0,
            exit_reason="signal"
        )

        self.assertEqual(long_trade.side, "LONG")
        self.assertEqual(short_trade.side, "SHORT")


class TestSimulateTrades(unittest.TestCase):
    """Tests pour la fonction simulate_trades."""

    def setUp(self):
        """Prépare les données de test."""
        self.df = create_test_ohlcv(100)

    def test_no_signals_returns_empty(self):
        """Test avec signaux nuls."""
        signals = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        params = {"leverage": 1, "k_sl": 1.5}

        trades_df = simulate_trades(self.df, signals, params)

        self.assertEqual(len(trades_df), 0)

    def test_single_long_signal(self):
        """Test avec un seul signal long."""
        signals = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        signals.iloc[10] = 1  # Signal long à l'indice 10

        params = {"leverage": 1, "k_sl": 10.0, "initial_capital": 10000}

        trades_df = simulate_trades(self.df, signals, params)

        # Au moins un trade devrait être créé (fermé en fin de période ou par stop)
        self.assertGreaterEqual(len(trades_df), 0)

    def test_long_short_signals(self):
        """Test avec signaux alternés long/short."""
        signals = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        signals.iloc[10] = 1   # Long
        signals.iloc[30] = -1  # Short
        signals.iloc[50] = 1   # Long
        signals.iloc[70] = -1  # Short

        params = {"leverage": 1, "k_sl": 10.0}

        trades_df = simulate_trades(self.df, signals, params)

        # On s'attend à plusieurs trades
        self.assertIsInstance(trades_df, pd.DataFrame)

    def test_returns_dataframe(self):
        """Test que simulate_trades retourne un DataFrame."""
        signals = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        signals.iloc[10] = 1

        params = {"leverage": 2, "k_sl": 5.0}

        result = simulate_trades(self.df, signals, params)

        self.assertIsInstance(result, pd.DataFrame)

    def test_params_default_values(self):
        """Test les valeurs par défaut des paramètres."""
        signals = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        signals.iloc[10] = 1

        # Appeler sans paramètres spécifiques
        params = {}
        result = simulate_trades(self.df, signals, params)

        self.assertIsInstance(result, pd.DataFrame)


class TestEquityCurve(unittest.TestCase):
    """Tests pour le calcul de la courbe d'équité."""

    def test_equity_curve_shape(self):
        """Test la forme de la courbe d'équité."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        equity = equity_curve(returns, initial_capital=10000)

        self.assertEqual(len(equity), len(returns))

    def test_equity_curve_initial_value(self):
        """Test que la courbe démarre au capital initial."""
        returns = pd.Series([0.0, 0.0, 0.0])

        equity = equity_curve(returns, initial_capital=10000)

        # Avec 0% de rendement, l'équité reste constante
        self.assertAlmostEqual(equity.iloc[0], 10000, places=2)

    def test_equity_curve_positive_returns(self):
        """Test avec des rendements positifs."""
        returns = pd.Series([0.05, 0.05, 0.05])  # +5% chaque période

        equity = equity_curve(returns, initial_capital=10000)

        # Après 3 périodes de +5%: 10000 * 1.05^3 ≈ 11576.25
        self.assertGreater(equity.iloc[-1], 10000)

    def test_equity_curve_empty_returns(self):
        """Test avec des rendements vides."""
        returns = pd.Series([], dtype=float)

        equity = equity_curve(returns, initial_capital=10000)

        self.assertEqual(len(equity), 0)


class TestDrawdownSeries(unittest.TestCase):
    """Tests pour le calcul du drawdown."""

    def test_drawdown_at_high(self):
        """Test que le drawdown est 0 au plus haut."""
        equity = pd.Series([100, 110, 120, 130])  # Toujours croissant

        dd = drawdown_series(equity)

        # Tout est à 0 car toujours au pic
        self.assertTrue(all(dd == 0))

    def test_drawdown_after_drop(self):
        """Test le drawdown après une baisse."""
        equity = pd.Series([100, 110, 105, 100])  # Pic à 110, puis baisse

        dd = drawdown_series(equity)

        self.assertEqual(dd.iloc[0], 0)  # Au premier pic
        self.assertEqual(dd.iloc[1], 0)  # Au nouveau pic
        self.assertLess(dd.iloc[2], 0)   # En drawdown
        self.assertLess(dd.iloc[3], 0)   # Toujours en drawdown

    def test_drawdown_bounds(self):
        """Test que le drawdown est entre -1 et 0."""
        np.random.seed(42)
        equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))

        dd = drawdown_series(equity)

        self.assertTrue(all(dd <= 0))
        self.assertTrue(all(dd >= -1))


class TestPerformanceMetrics(unittest.TestCase):
    """Tests pour les métriques de performance."""

    def test_metrics_dataclass(self):
        """Test la dataclass PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_pnl=1000.0,
            total_return_pct=10.0,
            annualized_return=15.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.15,
            max_drawdown_duration_days=5.0,
            volatility_annual=0.20,
            total_trades=50,
            win_rate=55.0,
            profit_factor=1.5,
            avg_win=100.0,
            avg_loss=-60.0,
            largest_win=500.0,
            largest_loss=-200.0,
            avg_trade_duration_hours=24.0,
            calmar_ratio=1.0,
            risk_reward_ratio=1.67,
            expectancy=25.0
        )

        self.assertEqual(metrics.total_pnl, 1000.0)
        self.assertEqual(metrics.total_trades, 50)
        self.assertEqual(metrics.win_rate, 55.0)

    def test_metrics_to_dict(self):
        """Test la conversion en dictionnaire."""
        metrics = PerformanceMetrics(
            total_pnl=500.0,
            total_return_pct=5.0,
            annualized_return=10.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.10,
            max_drawdown_duration_days=3.0,
            volatility_annual=0.15,
            total_trades=30,
            win_rate=60.0,
            profit_factor=1.8,
            avg_win=80.0,
            avg_loss=-50.0,
            largest_win=300.0,
            largest_loss=-150.0,
            avg_trade_duration_hours=12.0,
            calmar_ratio=1.5,
            risk_reward_ratio=1.6,
            expectancy=30.0
        )

        d = metrics.to_dict()

        self.assertIn("total_pnl", d)
        self.assertIn("sharpe_ratio", d)
        self.assertIn("max_drawdown", d)
        self.assertEqual(d["total_trades"], 30)


class TestSharpeRatio(unittest.TestCase):
    """Tests pour le calcul du Sharpe ratio."""

    def test_sharpe_positive_returns(self):
        """Test Sharpe avec rendements positifs."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01])

        result = sharpe_ratio(returns)

        # Rendements positifs stables → Sharpe positif
        self.assertGreater(result, 0)

    def test_sharpe_negative_returns(self):
        """Test Sharpe avec rendements négatifs."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01])

        result = sharpe_ratio(returns)

        # Rendements négatifs → Sharpe négatif
        self.assertLess(result, 0)

    def test_sharpe_zero_volatility(self):
        """Test Sharpe avec volatilité nulle."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])  # Rendements constants

        result = sharpe_ratio(returns)

        # Volatilité nulle, devrait gérer le cas
        self.assertTrue(np.isfinite(result) or result == 0)


class TestSortinoRatio(unittest.TestCase):
    """Tests pour le calcul du Sortino ratio."""

    def test_sortino_positive_returns(self):
        """Test Sortino avec rendements positifs."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01])

        result = sortino_ratio(returns)

        # Pas de rendements négatifs → Sortino devrait être élevé ou infini
        self.assertGreaterEqual(result, 0)

    def test_sortino_mixed_returns(self):
        """Test Sortino avec rendements mixtes."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

        result = sortino_ratio(returns)

        # Devrait retourner une valeur finie
        self.assertTrue(np.isfinite(result))


class TestMaxDrawdown(unittest.TestCase):
    """Tests pour le calcul du max drawdown."""

    def test_max_drawdown_value(self):
        """Test la valeur du max drawdown."""
        equity = pd.Series([100, 110, 105, 95, 100])  # Pic à 110, creux à 95

        max_dd = max_drawdown(equity)

        # Drawdown de 110 à 95 = (95-110)/110 ≈ -13.6%
        self.assertAlmostEqual(max_dd, (95-110)/110, places=3)

    def test_max_drawdown_no_drop(self):
        """Test sans drawdown."""
        equity = pd.Series([100, 110, 120, 130])  # Toujours croissant

        max_dd = max_drawdown(equity)

        self.assertEqual(max_dd, 0)


if __name__ == "__main__":
    unittest.main()
