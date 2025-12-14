"""
Tests unitaires pour les stratégies de trading.
===============================================
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators.registry import calculate_indicator
from strategies.base import StrategyResult
from strategies.bollinger_atr import BollingerATRStrategy
from strategies.ema_cross import EMACrossStrategy
from strategies.macd_cross import MACDCrossStrategy
from strategies.rsi_reversal import RSIReversalStrategy


def create_test_ohlcv(n: int = 500) -> pd.DataFrame:
    """Crée des données OHLCV de test."""
    np.random.seed(42)
    close = 100 + np.random.randn(n).cumsum() * 0.5

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.exponential(1000, n)
    })


class TestStrategyBase(unittest.TestCase):
    """Tests pour la classe de base StrategyBase."""

    def test_strategy_result_dataclass(self):
        """Test la dataclass StrategyResult."""
        signals = np.array([0, 1, -1, 0, 1])
        result = StrategyResult(
            signals=signals,
            entry_prices=np.array([100, 101, 102, 103, 104]),
            stop_losses=np.array([98, 99, 104, 101, 102])
        )

        self.assertEqual(len(result.signals), 5)
        self.assertIsNotNone(result.entry_prices)
        self.assertIsNotNone(result.stop_losses)


class TestBollingerATRStrategy(unittest.TestCase):
    """Tests pour la stratégie Bollinger + ATR."""

    def setUp(self):
        """Prépare les données et la stratégie."""
        self.df = create_test_ohlcv(500)
        self.strategy = BollingerATRStrategy()

    def test_strategy_name(self):
        """Vérifie le nom de la stratégie."""
        self.assertEqual(self.strategy.name, "BollingerATR")

    def test_required_indicators(self):
        """Vérifie les indicateurs requis."""
        required = self.strategy.required_indicators
        self.assertIn("bollinger", required)
        self.assertIn("atr", required)

    def test_default_params(self):
        """Test les paramètres par défaut."""
        params = self.strategy.default_params

        self.assertIn("bb_period", params)
        self.assertIn("bb_std", params)
        self.assertIn("atr_period", params)
        self.assertIn("entry_z", params)
        self.assertIn("k_sl", params)
        self.assertIn("leverage", params)

    def test_generate_signals_shape(self):
        """Vérifie la forme des signaux générés."""
        # Calculer les indicateurs nécessaires
        indicators = {
            "bollinger": calculate_indicator("bollinger", self.df, {"period": 20, "std_dev": 2.0}),
            "atr": calculate_indicator("atr", self.df, {"period": 14})
        }

        signals = self.strategy.generate_signals(self.df, indicators, {})

        self.assertEqual(len(signals), len(self.df))

    def test_signal_values(self):
        """Vérifie que les signaux sont dans {-1, 0, 1}."""
        indicators = {
            "bollinger": calculate_indicator("bollinger", self.df, {"period": 20, "std_dev": 2.0}),
            "atr": calculate_indicator("atr", self.df, {"period": 14})
        }

        signals = self.strategy.generate_signals(self.df, indicators, {})
        unique_values = set(signals)

        self.assertTrue(unique_values.issubset({-1, 0, 1}))

    def test_full_run(self):
        """Test l'exécution complète de la stratégie."""
        indicators = {
            "bollinger": calculate_indicator("bollinger", self.df, {"period": 20, "std_dev": 2.0}),
            "atr": calculate_indicator("atr", self.df, {"period": 14})
        }

        result = self.strategy.run(self.df, indicators, {})

        self.assertIsInstance(result, StrategyResult)
        self.assertEqual(len(result.signals), len(self.df))
        # entry_prices et stop_losses sont optionnels dans StrategyResult
        # Ils ne sont pas nécessairement fournis par run() de base
        self.assertIn("strategy_name", result.metadata)
        self.assertEqual(result.metadata["strategy_name"], "BollingerATR")


class TestEMACrossStrategy(unittest.TestCase):
    """Tests pour la stratégie EMA Crossover."""

    def setUp(self):
        """Prépare les données et la stratégie."""
        self.df = create_test_ohlcv(500)
        self.strategy = EMACrossStrategy()

    def test_strategy_name(self):
        """Vérifie le nom de la stratégie."""
        self.assertEqual(self.strategy.name, "EMACross")

    def test_required_indicators(self):
        """Vérifie les indicateurs requis (aucun externe)."""
        required = self.strategy.required_indicators
        # EMA Cross calcule ses propres indicateurs
        self.assertEqual(len(required), 0)

    def test_default_params(self):
        """Test les paramètres par défaut."""
        params = self.strategy.default_params

        self.assertIn("fast_period", params)
        self.assertIn("slow_period", params)
        self.assertIn("leverage", params)

    def test_generate_signals_shape(self):
        """Vérifie la forme des signaux générés."""
        signals = self.strategy.generate_signals(self.df, {}, {})

        self.assertEqual(len(signals), len(self.df))

    def test_signal_values(self):
        """Vérifie que les signaux sont dans {-1, 0, 1}."""
        signals = self.strategy.generate_signals(self.df, {}, {})
        unique_values = set(signals)

        self.assertTrue(unique_values.issubset({-1, 0, 1}))

    def test_crossover_detection(self):
        """Test que les croisements sont bien détectés."""
        # Créer des données avec un croisement évident
        n = 100
        # Prix qui monte puis descend
        prices = np.concatenate([
            np.linspace(100, 150, 50),
            np.linspace(150, 100, 50)
        ])

        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.ones(n) * 1000
        })

        signals = self.strategy.generate_signals(df, {}, {"fast_period": 5, "slow_period": 20})

        # Il devrait y avoir au moins quelques signaux
        n_signals = np.sum(signals != 0)
        self.assertGreater(n_signals, 0)


class TestStrategyReproducibility(unittest.TestCase):
    """Tests de reproductibilité des stratégies."""

    def test_bollinger_atr_reproducibility(self):
        """Test que Bollinger ATR donne les mêmes résultats."""
        df = create_test_ohlcv(500)
        strategy = BollingerATRStrategy()

        indicators = {
            "bollinger": calculate_indicator("bollinger", df, {"period": 20, "std_dev": 2.0}),
            "atr": calculate_indicator("atr", df, {"period": 14})
        }

        # Deux exécutions avec les mêmes données
        signals1 = strategy.generate_signals(df, indicators, {})
        signals2 = strategy.generate_signals(df, indicators, {})

        np.testing.assert_array_equal(signals1, signals2)

    def test_ema_cross_reproducibility(self):
        """Test que EMA Cross donne les mêmes résultats."""
        df = create_test_ohlcv(500)
        strategy = EMACrossStrategy()

        signals1 = strategy.generate_signals(df, {}, {})
        signals2 = strategy.generate_signals(df, {}, {})

        np.testing.assert_array_equal(signals1, signals2)


class TestMACDCrossStrategy(unittest.TestCase):
    """Tests pour la stratégie MACD Crossover."""

    def setUp(self):
        """Prépare les données et la stratégie."""
        self.df = create_test_ohlcv(500)
        self.strategy = MACDCrossStrategy()

    def test_strategy_name(self):
        """Vérifie le nom de la stratégie."""
        self.assertEqual(self.strategy.name, "macd_cross")

    def test_required_indicators(self):
        """Vérifie les indicateurs requis."""
        required = self.strategy.required_indicators
        self.assertIn("macd", required)

    def test_default_params(self):
        """Test les paramètres par défaut."""
        params = self.strategy.default_params

        self.assertIn("fast_period", params)
        self.assertIn("slow_period", params)
        self.assertIn("signal_period", params)
        self.assertIn("leverage", params)

    def test_generate_signals_shape(self):
        """Vérifie la forme des signaux générés."""
        indicators = {
            "macd": calculate_indicator("macd", self.df, {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            })
        }

        signals = self.strategy.generate_signals(self.df, indicators, {})
        self.assertEqual(len(signals), len(self.df))

    def test_signal_values(self):
        """Vérifie que les signaux sont dans {-1, 0, 1}."""
        indicators = {
            "macd": calculate_indicator("macd", self.df, {})
        }

        signals = self.strategy.generate_signals(self.df, indicators, {})
        unique_values = set(signals)

        self.assertTrue(unique_values.issubset({-1, 0, 1}))

    def test_macd_crossover_detection(self):
        """Test que les croisements MACD sont détectés."""
        indicators = {
            "macd": calculate_indicator("macd", self.df, {})
        }

        signals = self.strategy.generate_signals(self.df, indicators, {})

        # Il devrait y avoir au moins quelques signaux
        n_signals = np.sum(signals != 0)
        self.assertGreater(n_signals, 0)

    def test_full_run(self):
        """Test l'exécution complète de la stratégie."""
        indicators = {
            "macd": calculate_indicator("macd", self.df, {})
        }

        result = self.strategy.run(self.df, indicators, {})

        self.assertIsInstance(result, StrategyResult)
        self.assertEqual(len(result.signals), len(self.df))


class TestRSIReversalStrategy(unittest.TestCase):
    """Tests pour la stratégie RSI Reversal."""

    def setUp(self):
        """Prépare les données et la stratégie."""
        self.df = create_test_ohlcv(500)
        self.strategy = RSIReversalStrategy()

    def test_strategy_name(self):
        """Vérifie le nom de la stratégie."""
        self.assertEqual(self.strategy.name, "rsi_reversal")

    def test_required_indicators(self):
        """Vérifie les indicateurs requis."""
        required = self.strategy.required_indicators
        self.assertIn("rsi", required)

    def test_default_params(self):
        """Test les paramètres par défaut."""
        params = self.strategy.default_params

        self.assertIn("rsi_period", params)
        self.assertIn("oversold_level", params)
        self.assertIn("overbought_level", params)
        self.assertIn("leverage", params)

    def test_generate_signals_shape(self):
        """Vérifie la forme des signaux générés."""
        indicators = {
            "rsi": calculate_indicator("rsi", self.df, {"period": 14})
        }

        signals = self.strategy.generate_signals(
            self.df, indicators, {"oversold_level": 30, "overbought_level": 70}
        )
        self.assertEqual(len(signals), len(self.df))

    def test_signal_values(self):
        """Vérifie que les signaux sont dans {-1, 0, 1}."""
        indicators = {
            "rsi": calculate_indicator("rsi", self.df, {})
        }

        signals = self.strategy.generate_signals(
            self.df, indicators, {"oversold_level": 30, "overbought_level": 70}
        )
        unique_values = set(signals)

        self.assertTrue(unique_values.issubset({-1, 0, 1}))

    def test_rsi_reversal_detection(self):
        """Test la détection de survente/surachat."""
        # Créer des données avec des mouvements extrêmes
        n = 200
        prices = np.concatenate([
            np.linspace(100, 50, 50),   # Grosse baisse → survente
            np.linspace(50, 100, 50),   # Remontée
            np.linspace(100, 150, 50),  # Grosse hausse → surachat
            np.linspace(150, 100, 50)   # Correction
        ])

        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.ones(n) * 1000
        })

        indicators = {
            "rsi": calculate_indicator("rsi", df, {"period": 14})
        }

        signals = self.strategy.generate_signals(
            df, indicators, {"oversold_level": 30, "overbought_level": 70}
        )

        # Il devrait y avoir des signaux
        n_signals = np.sum(signals != 0)
        self.assertGreater(n_signals, 0)

    def test_full_run(self):
        """Test l'exécution complète de la stratégie."""
        indicators = {
            "rsi": calculate_indicator("rsi", self.df, {})
        }

        result = self.strategy.run(self.df, indicators, {})

        self.assertIsInstance(result, StrategyResult)
        self.assertEqual(len(result.signals), len(self.df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
