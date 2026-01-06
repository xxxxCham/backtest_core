"""
Tests automatiques pour vérifier les corrections du système.

Teste:
1. Imports httpx avec réimport automatique
2. Imports CuPy avec réimport automatique
3. BacktestResult attributes (total_return vs total_pnl)
4. Sharpe Ratio calculation
5. Backtest complet (si données disponibles)
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHttpxReimport:
    """Test httpx.Client avec réimport automatique."""

    def test_httpx_client_available(self):
        """Vérifie que httpx.Client est disponible après reload."""
        import httpx

        # Simuler la perte de l'attribut (comme après reload Streamlit)
        original_client = None
        if hasattr(httpx, 'Client'):
            original_client = httpx.Client
            delattr(httpx, 'Client')

        try:
            # Réimporter httpx
            importlib.reload(httpx)
            assert hasattr(httpx, 'Client'), "httpx.Client devrait être disponible après reload"

            # Tester l'instanciation
            client = httpx.Client()
            client.close()
        finally:
            # Restaurer si nécessaire
            if original_client is not None:
                httpx.Client = original_client


class TestCuPyFallback:
    """Test CuPy avec fallback CPU."""

    def test_device_backend_cpu_fallback(self):
        """Vérifie que DeviceBackend fonctionne en mode CPU."""
        try:
            from performance.device_backend import DeviceBackend

            backend = DeviceBackend()
            backend.select_device('cpu')
            assert backend.device == 'cpu', "DeviceBackend devrait être en mode CPU"
        except ImportError:
            pytest.skip("DeviceBackend non disponible")


class TestBacktestResult:
    """Test BacktestResult attributes."""

    def test_total_return_attribute(self):
        """Vérifie que BacktestResult a total_return et pas total_pnl."""
        from agents.backtest_executor import BacktestRequest, BacktestResult

        request = BacktestRequest(
            strategy_name="test",
            parameters={"test": 1}
        )

        result = BacktestResult(
            request=request,
            success=True,
            total_return=0.15,  # 15% return
            sharpe_ratio=1.5
        )

        assert hasattr(result, 'total_return'), "total_return attribute manquant!"
        assert result.total_return == 0.15, "total_return valeur incorrecte!"
        assert not hasattr(result, 'total_pnl'), "total_pnl ne devrait pas exister!"


class TestSharpeRatio:
    """Test Sharpe Ratio calculation."""

    def test_sharpe_with_sufficient_samples(self):
        """Vérifie le calcul du Sharpe avec suffisamment d'échantillons."""
        from backtest.performance import sharpe_ratio

        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        sharpe = sharpe_ratio(returns, method='standard')

        # Le sharpe doit être un nombre fini (pas NaN, pas Inf)
        assert np.isfinite(sharpe), f"Sharpe devrait être fini, got {sharpe}"

    def test_sharpe_with_few_samples(self):
        """Vérifie le comportement du Sharpe avec peu d'échantillons."""
        from backtest.performance import sharpe_ratio

        np.random.seed(42)
        returns = pd.Series(np.random.randn(10) * 0.01)
        sharpe = sharpe_ratio(returns, method='standard')

        # Doit retourner un nombre (peut être 0 ou calculé selon implémentation)
        assert isinstance(sharpe, (int, float)), f"Sharpe devrait être numérique, got {type(sharpe)}"


class TestBacktestEngine:
    """Test BacktestEngine avec données synthétiques."""

    @pytest.fixture
    def synthetic_data(self):
        """Génère des données OHLCV synthétiques."""
        np.random.seed(42)
        n_bars = 500
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
        close = 40000 + np.cumsum(np.random.randn(n_bars) * 100)
        return pd.DataFrame({
            'open': close + np.random.randn(n_bars) * 50,
            'high': close + np.abs(np.random.randn(n_bars) * 100),
            'low': close - np.abs(np.random.randn(n_bars) * 100),
            'close': close,
            'volume': np.random.randint(1000, 10000, n_bars).astype(float)
        }, index=dates)

    def test_backtest_ema_cross(self, synthetic_data):
        """Test backtest avec stratégie ema_cross."""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(
            df=synthetic_data,
            strategy='ema_cross',
            params={'fast_period': 12, 'slow_period': 26, 'leverage': 1},
            timeframe='1h'
        )

        assert result is not None, "Résultat backtest ne doit pas être None"
        assert hasattr(result, 'metrics'), "Résultat doit avoir des metrics"
        assert 'total_trades' in result.metrics, "Metrics doit avoir total_trades"
        assert isinstance(result.metrics['total_trades'], (int, np.integer)), \
            f"total_trades doit être int, got {type(result.metrics['total_trades'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
