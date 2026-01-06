"""
Test pour vérifier que total_trades est toujours un entier.

Convertie en test pytest pour éviter l'exécution au niveau module.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine


@pytest.fixture
def synthetic_data():
    """Génère des données OHLCV synthétiques."""
    np.random.seed(42)
    n_bars = 4320
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    close = 40000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 50,
        'high': close + np.abs(np.random.randn(n_bars) * 100),
        'low': close - np.abs(np.random.randn(n_bars) * 100),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars).astype(float)
    }, index=dates)


class TestGridSearchTrades:
    """Test que total_trades est toujours un entier."""

    def test_total_trades_is_integer(self, synthetic_data):
        """Vérifie que total_trades est bien un entier, pas un float."""
        # Grille de paramètres avec ema_cross (atr_channel non disponible)
        param_grid = [
            {'fast_period': 8, 'slow_period': 21, 'leverage': 1},
            {'fast_period': 10, 'slow_period': 26, 'leverage': 1},
            {'fast_period': 12, 'slow_period': 30, 'leverage': 1},
            {'fast_period': 15, 'slow_period': 40, 'leverage': 1},
            {'fast_period': 20, 'slow_period': 50, 'leverage': 1},
        ]

        for params in param_grid:
            engine = BacktestEngine(initial_capital=10000)
            result = engine.run(df=synthetic_data, strategy='ema_cross', params=params, timeframe='1h')

            metrics = result.metrics
            total_trades = metrics.get("total_trades", 0)

            # Vérifier le type
            assert isinstance(total_trades, (int, np.integer)), \
                f"total_trades devrait être int, got {type(total_trades).__name__} = {total_trades}"

    def test_metrics_keys_consistency(self, synthetic_data):
        """Vérifie que les clés de métriques sont cohérentes."""
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(
            df=synthetic_data,
            strategy='ema_cross',
            params={'fast_period': 12, 'slow_period': 26, 'leverage': 1},
            timeframe='1h'
        )

        metrics = result.metrics

        # Vérifier les clés obligatoires (noms corrects)
        expected_keys = ['total_trades', 'sharpe_ratio', 'profit_factor']
        for key in expected_keys:
            assert key in metrics, f"Clé '{key}' manquante dans metrics"

        # Vérifier clés avec suffixe _pct (pas sans suffixe)
        pct_keys = ['max_drawdown_pct', 'win_rate_pct', 'total_return_pct']
        for key in pct_keys:
            if key in metrics:
                assert isinstance(metrics[key], (int, float)), \
                    f"Clé '{key}' devrait être numérique"

    def test_no_fractional_trades_in_dataframe(self, synthetic_data):
        """Vérifie qu'on n'obtient pas de trades fractionnaires dans un DataFrame."""
        param_grid = [
            {'fast_period': 8, 'slow_period': 21, 'leverage': 1},
            {'fast_period': 12, 'slow_period': 26, 'leverage': 1},
        ]

        results_list = []
        for params in param_grid:
            engine = BacktestEngine(initial_capital=10000)
            result = engine.run(df=synthetic_data, strategy='ema_cross', params=params, timeframe='1h')
            metrics = result.metrics

            results_list.append({
                "params": str(params),
                "trades": metrics.get("total_trades", 0),
                "sharpe": metrics.get("sharpe_ratio", 0),
            })

        df = pd.DataFrame(results_list)

        # Toutes les valeurs de 'trades' doivent être des entiers
        for val in df['trades'].values:
            if isinstance(val, float):
                assert val.is_integer(), f"Valeur fractionnaire détectée: {val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
