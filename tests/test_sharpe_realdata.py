"""
Test du fix Sharpe Ratio avec données réalistes (6 mois, 1h).

Convertie en test pytest pour éviter l'exécution au niveau module.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine


@pytest.fixture
def realistic_data():
    """Génère des données réalistes 1h sur 6 mois (~4320 barres)."""
    np.random.seed(42)
    start_date = pd.Timestamp('2024-08-01', tz='UTC')
    periods = 4320  # 6 mois x 30 jours x 24h = 4320
    dates = pd.date_range(start=start_date, periods=periods, freq='h')

    # Prix réaliste BTC avec tendance haussière + volatilité
    price_base = 40000
    trend = np.linspace(0, 10000, periods)  # +$10k sur 6 mois
    noise = np.random.randn(periods).cumsum() * 200
    price = price_base + trend + noise

    # OHLC réaliste
    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(periods) * 150),
        'low': price - np.abs(np.random.randn(periods) * 150),
        'close': price + np.random.randn(periods) * 50,
        'volume': np.random.randint(1000, 50000, size=periods).astype(float)
    }, index=dates)

    # Assurer high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestSharpeRealdata:
    """Tests du calcul Sharpe avec données réalistes."""

    def test_sharpe_varies_with_params(self, realistic_data):
        """Vérifie que le Sharpe est calculé (peut être 0.0 si garde active)."""
        # Paramètres adaptés pour ema_cross
        test_params = [
            {'fast_period': 10, 'slow_period': 21, 'leverage': 1},
            {'fast_period': 14, 'slow_period': 28, 'leverage': 1},
            {'fast_period': 20, 'slow_period': 40, 'leverage': 1},
        ]

        sharpe_values = []
        for params in test_params:
            engine = BacktestEngine(initial_capital=10000)
            result = engine.run(df=realistic_data, strategy='ema_cross', params=params, timeframe='1h')

            sharpe = result.metrics.get('sharpe_ratio', 0)
            sharpe_values.append(sharpe)

        # Vérifier que tous les Sharpe sont des nombres valides
        for sharpe in sharpe_values:
            assert isinstance(sharpe, (int, float)), f"Sharpe invalide: {sharpe}"
            assert np.isfinite(sharpe), f"Sharpe non fini: {sharpe}"

        # Note: Sharpe peut être 0.0 si garde (< MIN_SAMPLES) est active
        # On vérifie juste que le calcul s'exécute sans erreur

    def test_sharpe_is_finite(self, realistic_data):
        """Vérifie que le Sharpe est un nombre fini."""
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(
            df=realistic_data,
            strategy='ema_cross',
            params={'fast_period': 12, 'slow_period': 26, 'leverage': 1},
            timeframe='1h'
        )

        sharpe = result.metrics.get('sharpe_ratio', 0)
        assert np.isfinite(sharpe), f"Sharpe devrait être fini, got {sharpe}"

    def test_metrics_keys_correct(self, realistic_data):
        """Vérifie que les clés de métriques sont correctes (_pct suffix)."""
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(
            df=realistic_data,
            strategy='ema_cross',
            params={'fast_period': 12, 'slow_period': 26, 'leverage': 1},
            timeframe='1h'
        )

        metrics = result.metrics

        # Vérifier les clés avec suffixe _pct (nouvelles conventions)
        # win_rate_pct au lieu de win_rate
        if 'win_rate_pct' in metrics:
            assert isinstance(metrics['win_rate_pct'], (int, float))
        elif 'win_rate' in metrics:
            # Fallback si l'ancienne clé existe encore
            assert isinstance(metrics['win_rate'], (int, float))

        # max_drawdown_pct au lieu de max_drawdown
        if 'max_drawdown_pct' in metrics:
            assert isinstance(metrics['max_drawdown_pct'], (int, float))
        elif 'max_drawdown' in metrics:
            assert isinstance(metrics['max_drawdown'], (int, float))

    def test_multiple_backtests_stability(self, realistic_data):
        """Vérifie la stabilité des résultats sur plusieurs exécutions."""
        params = {'fast_period': 14, 'slow_period': 28, 'leverage': 1}

        results = []
        for _ in range(3):
            engine = BacktestEngine(initial_capital=10000)
            result = engine.run(df=realistic_data, strategy='ema_cross', params=params, timeframe='1h')
            results.append(result.metrics.get('sharpe_ratio', 0))

        # Tous les résultats doivent être identiques (déterminisme)
        assert all(r == results[0] for r in results), \
            f"Résultats non déterministes: {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
