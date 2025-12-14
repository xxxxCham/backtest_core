"""
Tests unitaires pour les indicateurs techniques.
================================================

Tests consolidés pour tous les indicateurs :
- Bollinger Bands
- ATR (Average True Range)
- RSI (Relative Strength Index)
- EMA/SMA (Moyennes mobiles)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Registre d'indicateurs

Consolidé depuis test_indicators.py + test_indicators_new.py (13/12/2025)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators.atr import atr, true_range
from indicators.bollinger import bollinger_bands
from indicators.ema import ema, sma
from indicators.registry import calculate_indicator, list_indicators
from indicators.rsi import rsi
from indicators.macd import (
    macd,
    macd_histogram_divergence,
    calculate_macd,
)
from indicators.adx import (
    adx,
    directional_movement,
    adx_trend_strength,
    adx_signal,
    calculate_adx,
)


# =============================================================================
# FIXTURES COMMUNES
# =============================================================================

@pytest.fixture
def sample_prices():
    """Génère une série de prix simple pour les tests."""
    np.random.seed(42)
    return pd.Series(100 + np.random.randn(100).cumsum())


@pytest.fixture
def sample_ohlcv():
    """Génère un DataFrame OHLCV réaliste pour les tests."""
    np.random.seed(42)
    n = 200
    
    # Générer un prix de base avec tendance et bruit
    base_price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # Créer OHLCV réaliste
    high = base_price + np.abs(np.random.randn(n)) * 0.5
    low = base_price - np.abs(np.random.randn(n)) * 0.5
    open_price = low + (high - low) * np.random.rand(n)
    close = low + (high - low) * np.random.rand(n)
    volume = np.random.exponential(1000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def trending_up_data():
    """Génère des données avec tendance haussière claire."""
    np.random.seed(42)
    n = 100
    
    # Tendance linéaire avec petit bruit
    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 0.5
    close = trend + noise
    
    return pd.DataFrame({
        'open': close - np.random.rand(n) * 0.3,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.exponential(1000, n)
    })


@pytest.fixture
def trending_down_data():
    """Génère des données avec tendance baissière claire."""
    np.random.seed(42)
    n = 100
    
    trend = np.linspace(150, 100, n)
    noise = np.random.randn(n) * 0.5
    close = trend + noise
    
    return pd.DataFrame({
        'open': close + np.random.rand(n) * 0.3,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.exponential(1000, n)
    })


# =============================================================================
# TESTS BOLLINGER BANDS
# =============================================================================

class TestBollingerBands:
    """Tests pour Bollinger Bands."""

    def test_bollinger_output_shape(self, sample_prices):
        """Vérifie la forme des sorties."""
        upper, middle, lower = bollinger_bands(sample_prices, period=20, std_dev=2.0)

        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)

    def test_bollinger_band_ordering(self, sample_prices):
        """Vérifie que upper > middle > lower."""
        upper, middle, lower = bollinger_bands(sample_prices, period=20, std_dev=2.0)

        # Exclure les NaN du début
        valid_idx = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))

        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_bollinger_middle_is_sma(self, sample_prices):
        """Vérifie que la bande du milieu est la SMA."""
        upper, middle, lower = bollinger_bands(sample_prices, period=20, std_dev=2.0)
        expected_sma = sample_prices.rolling(20).mean().values

        np.testing.assert_array_almost_equal(middle, expected_sma)


# =============================================================================
# TESTS ATR
# =============================================================================

class TestATR:
    """Tests pour ATR."""

    def test_atr_output_shape(self, sample_ohlcv):
        """Vérifie la forme des sorties."""
        result = atr(sample_ohlcv['high'], sample_ohlcv['low'], sample_ohlcv['close'], period=14)
        assert len(result) == len(sample_ohlcv)

    def test_atr_positive_values(self, sample_ohlcv):
        """Vérifie que l'ATR est toujours positif."""
        result = atr(sample_ohlcv['high'], sample_ohlcv['low'], sample_ohlcv['close'], period=14)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_true_range(self, sample_ohlcv):
        """Test du True Range."""
        tr = true_range(sample_ohlcv['high'], sample_ohlcv['low'], sample_ohlcv['close'])
        assert len(tr) == len(sample_ohlcv)
        assert np.all(tr[~np.isnan(tr)] >= 0)


# =============================================================================
# TESTS RSI
# =============================================================================

class TestRSI:
    """Tests pour RSI."""

    def test_rsi_output_shape(self, sample_prices):
        """Vérifie la forme des sorties."""
        result = rsi(sample_prices, period=14)
        assert len(result) == len(sample_prices)

    def test_rsi_bounds(self, sample_prices):
        """Vérifie que le RSI est entre 0 et 100."""
        result = rsi(sample_prices, period=14)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)
        assert np.all(result[valid] <= 100)

    def test_rsi_uptrend(self):
        """Test RSI sur tendance haussière."""
        prices = pd.Series(np.linspace(100, 150, 50))
        result = rsi(prices, period=14)
        # En tendance haussière, RSI devrait être > 50
        assert result[-1] > 50

    def test_rsi_downtrend(self):
        """Test RSI sur tendance baissière."""
        prices = pd.Series(np.linspace(150, 100, 50))
        result = rsi(prices, period=14)
        # En tendance baissière, RSI devrait être < 50
        assert result[-1] < 50


# =============================================================================
# TESTS EMA / SMA
# =============================================================================

class TestEMA:
    """Tests pour EMA et SMA."""

    def test_ema_output_shape(self, sample_prices):
        """Vérifie la forme des sorties."""
        result = ema(sample_prices, period=20)
        assert len(result) == len(sample_prices)

    def test_sma_output_shape(self, sample_prices):
        """Vérifie la forme des sorties."""
        result = sma(sample_prices, period=20)
        assert len(result) == len(sample_prices)

    def test_ema_responds_faster_than_sma(self):
        """Vérifie que l'EMA réagit plus vite que la SMA après un changement."""
        # Données avec un saut soudain : 100 pendant 30 barres, puis 120 pendant 5 barres
        prices = pd.Series([100.0] * 30 + [120.0] * 5)

        ema_vals = ema(prices, period=10)
        sma_vals = sma(prices, period=10)

        # Juste après le saut (à l'indice 32), l'EMA devrait être au-dessus de SMA
        idx_after_jump = 32
        assert ema_vals[idx_after_jump] > sma_vals[idx_after_jump]


# =============================================================================
# TESTS MACD
# =============================================================================

class TestMACD:
    """Tests pour l'indicateur MACD."""

    def test_macd_output_shape(self, sample_ohlcv):
        """MACD devrait retourner 3 arrays de même taille."""
        macd_line, signal_line, histogram = macd(sample_ohlcv['close'])

        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)

    def test_macd_histogram_calculation(self, sample_ohlcv):
        """L'histogramme devrait être MACD - Signal."""
        macd_line, signal_line, histogram = macd(sample_ohlcv['close'])

        # Vérifier sur les valeurs non-NaN
        valid = ~(np.isnan(macd_line) | np.isnan(signal_line) | np.isnan(histogram))

        expected = macd_line[valid] - signal_line[valid]
        np.testing.assert_array_almost_equal(histogram[valid], expected, decimal=10)

    def test_macd_custom_periods(self, sample_ohlcv):
        """MACD avec périodes personnalisées."""
        macd_line, signal_line, histogram = macd(
            sample_ohlcv['close'],
            fast_period=8,
            slow_period=21,
            signal_period=5
        )

        assert len(macd_line) == len(sample_ohlcv)

    def test_macd_uptrend(self, trending_up_data):
        """En tendance haussière, MACD devrait être positif."""
        macd_line, _, _ = macd(trending_up_data['close'])

        # Les dernières valeurs (tendance établie) devraient être positives
        last_values = macd_line[-20:]
        valid = last_values[~np.isnan(last_values)]

        assert len(valid) > 0
        assert np.mean(valid) > 0


class TestMACDHistogramDivergence:
    """Tests pour la détection de divergence MACD."""

    def test_divergence_output_shape(self, sample_ohlcv):
        """La divergence devrait avoir la même taille que l'entrée."""
        _, _, histogram = macd(sample_ohlcv['close'])
        divergence = macd_histogram_divergence(sample_ohlcv['close'], histogram)

        assert len(divergence) == len(sample_ohlcv)

    def test_divergence_values(self, sample_ohlcv):
        """Les valeurs devraient être des entiers dans [-1, 0, 1]."""
        _, _, histogram = macd(sample_ohlcv['close'])
        divergence = macd_histogram_divergence(sample_ohlcv['close'], histogram)

        unique_values = set(divergence[~np.isnan(divergence)])
        assert unique_values.issubset({-1, 0, 1})


class TestCalculateMACD:
    """Tests pour la fonction wrapper calculate_macd."""

    def test_calculate_macd_returns_dict(self, sample_ohlcv):
        """calculate_macd devrait retourner un dictionnaire."""
        params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        result = calculate_macd(sample_ohlcv, params)

        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result


# =============================================================================
# TESTS ADX
# =============================================================================

class TestADX:
    """Tests pour l'indicateur ADX."""

    def test_adx_output_shape(self, sample_ohlcv):
        """ADX devrait retourner 3 arrays de même taille."""
        adx_val, plus_di, minus_di = adx(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        assert len(adx_val) == len(sample_ohlcv)
        assert len(plus_di) == len(sample_ohlcv)
        assert len(minus_di) == len(sample_ohlcv)

    def test_adx_bounded(self, sample_ohlcv):
        """ADX devrait être entre 0 et 100."""
        adx_val, plus_di, minus_di = adx(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        valid_adx = adx_val[~np.isnan(adx_val)]
        assert np.all(valid_adx >= 0)
        assert np.all(valid_adx <= 100)

    def test_di_bounded(self, sample_ohlcv):
        """+DI et -DI devraient être entre 0 et 100."""
        _, plus_di, minus_di = adx(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        valid_plus = plus_di[~np.isnan(plus_di)]
        valid_minus = minus_di[~np.isnan(minus_di)]

        assert np.all(valid_plus >= 0)
        assert np.all(valid_minus >= 0)


class TestDirectionalMovement:
    """Tests pour le calcul du mouvement directionnel."""

    def test_dm_output(self, sample_ohlcv):
        """directional_movement devrait retourner 3 arrays (DM+, DM-, TR)."""
        plus_dm, minus_dm, tr = directional_movement(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        assert len(plus_dm) == len(sample_ohlcv)
        assert len(minus_dm) == len(sample_ohlcv)
        assert len(tr) == len(sample_ohlcv)

    def test_dm_non_negative(self, sample_ohlcv):
        """Les DM devraient être non-négatifs."""
        plus_dm, minus_dm, tr = directional_movement(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        valid_plus = plus_dm[~np.isnan(plus_dm)]
        valid_minus = minus_dm[~np.isnan(minus_dm)]

        assert np.all(valid_plus >= 0)
        assert np.all(valid_minus >= 0)


class TestADXTrendStrength:
    """Tests pour l'évaluation de la force de tendance."""

    def test_trend_strength_output(self, sample_ohlcv):
        """adx_trend_strength devrait retourner un array."""
        strength = adx_trend_strength(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        assert len(strength) == len(sample_ohlcv)

    def test_trend_strength_bounded(self, sample_ohlcv):
        """ADX devrait être entre 0 et 100."""
        strength = adx_trend_strength(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        valid_strength = strength[~np.isnan(strength)]
        assert np.all(valid_strength >= 0)
        assert np.all(valid_strength <= 100)


class TestADXSignal:
    """Tests pour les signaux ADX."""

    def test_adx_signal_output(self, sample_ohlcv):
        """adx_signal devrait retourner un array."""
        signals = adx_signal(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        assert len(signals) == len(sample_ohlcv)

    def test_adx_signal_values(self, sample_ohlcv):
        """Les signaux devraient être -1, 0, ou 1."""
        signals = adx_signal(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        valid_signals = signals[~np.isnan(signals)]
        unique_values = set(valid_signals.astype(int))

        assert unique_values.issubset({-1, 0, 1})


class TestCalculateADX:
    """Tests pour la fonction wrapper calculate_adx."""

    def test_calculate_adx_returns_dict(self, sample_ohlcv):
        """calculate_adx devrait retourner un dictionnaire."""
        params = {"period": 14}
        result = calculate_adx(sample_ohlcv, params)

        assert isinstance(result, dict)
        assert 'adx' in result
        assert 'plus_di' in result
        assert 'minus_di' in result


# =============================================================================
# TESTS REGISTRE D'INDICATEURS
# =============================================================================

class TestIndicatorRegistry:
    """Tests pour le registre d'indicateurs."""

    def test_list_indicators(self):
        """Test la liste des indicateurs."""
        indicators = list_indicators()
        assert "bollinger" in indicators
        assert "atr" in indicators
        assert "rsi" in indicators
        assert "ema" in indicators
        assert "macd" in indicators
        assert "adx" in indicators

    def test_calculate_bollinger(self, sample_ohlcv):
        """Test le calcul de Bollinger via registre."""
        result = calculate_indicator("bollinger", sample_ohlcv, {"period": 20})

        assert isinstance(result, tuple)
        assert len(result) == 3  # upper, middle, lower
        upper, middle, lower = result
        assert len(upper) == len(sample_ohlcv)

    def test_calculate_atr(self, sample_ohlcv):
        """Test le calcul d'ATR via registre."""
        result = calculate_indicator("atr", sample_ohlcv, {"period": 14})
        assert len(result) == len(sample_ohlcv)

    def test_calculate_rsi(self, sample_ohlcv):
        """Test le calcul de RSI via registre."""
        result = calculate_indicator("rsi", sample_ohlcv, {"period": 14})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_ohlcv)

    def test_calculate_multiple_indicators(self, sample_ohlcv):
        """Test le calcul de plusieurs indicateurs séquentiellement."""
        bollinger_result = calculate_indicator("bollinger", sample_ohlcv, {"period": 20})
        atr_result = calculate_indicator("atr", sample_ohlcv, {"period": 14})
        rsi_result = calculate_indicator("rsi", sample_ohlcv, {"period": 14})

        assert len(bollinger_result) == 3
        assert len(atr_result) == len(sample_ohlcv)
        assert len(rsi_result) == len(sample_ohlcv)


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

class TestIndicatorsIntegration:
    """Tests d'intégration entre indicateurs."""

    def test_all_indicators_same_data(self, sample_ohlcv):
        """Tous les indicateurs devraient fonctionner sur les mêmes données."""
        macd_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        adx_params = {"period": 14}

        macd_result = calculate_macd(sample_ohlcv, macd_params)
        adx_result = calculate_adx(sample_ohlcv, adx_params)

        # Vérifier que les deux ont la bonne longueur
        assert len(macd_result['macd']) == len(adx_result['adx'])

    def test_strong_trend_detection(self, trending_up_data):
        """En forte tendance, ADX devrait être élevé et MACD positif."""
        macd_line, _, _ = macd(trending_up_data['close'])
        adx_val, plus_di, minus_di = adx(
            trending_up_data['high'],
            trending_up_data['low'],
            trending_up_data['close']
        )

        # En fin de données (tendance établie)
        last_20 = slice(-20, None)

        # MACD devrait être majoritairement positif
        valid_macd = macd_line[last_20][~np.isnan(macd_line[last_20])]
        if len(valid_macd) > 0:
            assert np.mean(valid_macd) > 0

        # +DI devrait dominer -DI en tendance haussière
        valid_plus_di = plus_di[last_20][~np.isnan(plus_di[last_20])]
        valid_minus_di = minus_di[last_20][~np.isnan(minus_di[last_20])]

        if len(valid_plus_di) > 0 and len(valid_minus_di) > 0:
            assert np.mean(valid_plus_di) > np.mean(valid_minus_di)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
