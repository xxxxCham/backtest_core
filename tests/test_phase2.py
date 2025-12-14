"""
Tests Phase 2 - Indicateurs avancés et IndicatorBank
====================================================

Tests pour:
- Ichimoku Cloud
- Parabolic SAR
- Stochastic RSI
- Vortex Indicator
- IndicatorBank (cache disque)

Créé le 13/12/2025
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators.ichimoku import (
    ichimoku,
    tenkan_sen,
    kijun_sen,
    senkou_span_a,
    senkou_span_b,
    chikou_span,
    ichimoku_cloud_position,
    ichimoku_signal,
    calculate_ichimoku,
)
from indicators.psar import (
    parabolic_sar,
    psar_signal,
    psar_stop_loss,
    calculate_psar,
)
from indicators.stoch_rsi import (
    stochastic_rsi,
    stoch_rsi_signal,
    stoch_rsi_divergence,
    calculate_stoch_rsi,
)
from indicators.vortex import (
    vortex,
    vortex_movement,
    vortex_signal,
    vortex_trend_strength,
    vortex_oscillator,
    calculate_vortex,
)
from data.indicator_bank import (
    IndicatorBank,
    CacheStats,
    get_indicator_bank,
    cached_indicator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Génère un DataFrame OHLCV pour les tests."""
    np.random.seed(42)
    n = 200
    
    base_price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
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
    """Données avec tendance haussière claire."""
    np.random.seed(42)
    n = 150
    
    trend = np.linspace(100, 160, n)
    noise = np.random.randn(n) * 0.3
    close = trend + noise
    
    return pd.DataFrame({
        'open': close - np.random.rand(n) * 0.2,
        'high': close + np.abs(np.random.randn(n)) * 0.4,
        'low': close - np.abs(np.random.randn(n)) * 0.4,
        'close': close,
        'volume': np.random.exponential(1000, n)
    })


@pytest.fixture
def temp_cache_dir():
    """Crée un répertoire temporaire pour le cache."""
    cache_dir = tempfile.mkdtemp()
    yield Path(cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)


# =============================================================================
# TESTS ICHIMOKU
# =============================================================================

class TestIchimoku:
    """Tests pour l'indicateur Ichimoku Cloud."""
    
    def test_tenkan_sen_output_shape(self, sample_ohlcv):
        """Tenkan-sen devrait avoir la même taille que l'entrée."""
        result = tenkan_sen(sample_ohlcv['high'], sample_ohlcv['low'], 9)
        assert len(result) == len(sample_ohlcv)
    
    def test_kijun_sen_output_shape(self, sample_ohlcv):
        """Kijun-sen devrait avoir la même taille que l'entrée."""
        result = kijun_sen(sample_ohlcv['high'], sample_ohlcv['low'], 26)
        assert len(result) == len(sample_ohlcv)
    
    def test_ichimoku_full_output(self, sample_ohlcv):
        """ichimoku() devrait retourner 5 composants."""
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert len(tenkan) == len(sample_ohlcv)
        assert len(kijun) == len(sample_ohlcv)
        assert len(senkou_a) == len(sample_ohlcv)
        assert len(senkou_b) == len(sample_ohlcv)
        assert len(chikou) == len(sample_ohlcv)
    
    def test_tenkan_faster_than_kijun(self, sample_ohlcv):
        """Tenkan (9) devrait avoir moins de NaN que Kijun (26)."""
        tenkan = tenkan_sen(sample_ohlcv['high'], sample_ohlcv['low'], 9)
        kijun = kijun_sen(sample_ohlcv['high'], sample_ohlcv['low'], 26)
        
        tenkan_nan = np.sum(np.isnan(tenkan))
        kijun_nan = np.sum(np.isnan(kijun))
        
        assert tenkan_nan < kijun_nan
    
    def test_cloud_position_values(self, sample_ohlcv):
        """Cloud position devrait retourner -1, 0, ou 1."""
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        cloud_pos = ichimoku_cloud_position(sample_ohlcv['close'], senkou_a, senkou_b)
        valid = cloud_pos[~np.isnan(cloud_pos)]
        
        assert set(valid).issubset({-1, 0, 1})
    
    def test_calculate_ichimoku_returns_dict(self, sample_ohlcv):
        """calculate_ichimoku devrait retourner un dict."""
        result = calculate_ichimoku(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert 'tenkan' in result
        assert 'kijun' in result
        assert 'senkou_a' in result
        assert 'senkou_b' in result
        assert 'chikou' in result


# =============================================================================
# TESTS PARABOLIC SAR
# =============================================================================

class TestParabolicSAR:
    """Tests pour le Parabolic SAR."""
    
    def test_psar_output_shape(self, sample_ohlcv):
        """PSAR devrait retourner sar et trend de même taille."""
        sar, trend = parabolic_sar(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert len(sar) == len(sample_ohlcv)
        assert len(trend) == len(sample_ohlcv)
    
    def test_psar_trend_values(self, sample_ohlcv):
        """Trend devrait être 1 (up) ou -1 (down)."""
        sar, trend = parabolic_sar(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        valid_trend = trend[trend != 0]
        assert set(valid_trend).issubset({-1, 1})
    
    def test_psar_sar_position(self, sample_ohlcv):
        """SAR devrait être sous le prix en tendance haussière."""
        sar, trend = parabolic_sar(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        close = sample_ohlcv['close'].values
        
        # En tendance up (1), SAR devrait être sous le prix
        up_mask = (trend == 1) & ~np.isnan(sar)
        if np.sum(up_mask) > 0:
            assert np.mean(sar[up_mask] < close[up_mask]) > 0.8
    
    def test_psar_signal_values(self, sample_ohlcv):
        """psar_signal devrait retourner -1, 0, ou 1."""
        signal = psar_signal(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert set(signal).issubset({-1, 0, 1})
    
    def test_calculate_psar_returns_dict(self, sample_ohlcv):
        """calculate_psar devrait retourner un dict."""
        result = calculate_psar(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert 'sar' in result
        assert 'trend' in result
        assert 'signal' in result


# =============================================================================
# TESTS STOCHASTIC RSI
# =============================================================================

class TestStochasticRSI:
    """Tests pour le Stochastic RSI."""
    
    def test_stoch_rsi_output_shape(self, sample_ohlcv):
        """StochRSI devrait retourner %K et %D de même taille."""
        k, d = stochastic_rsi(sample_ohlcv['close'])
        
        assert len(k) == len(sample_ohlcv)
        assert len(d) == len(sample_ohlcv)
    
    def test_stoch_rsi_bounds(self, sample_ohlcv):
        """StochRSI devrait être entre 0 et 100."""
        k, d = stochastic_rsi(sample_ohlcv['close'])
        
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        
        assert np.all(valid_k >= 0)
        assert np.all(valid_k <= 100)
        assert np.all(valid_d >= 0)
        assert np.all(valid_d <= 100)
    
    def test_stoch_rsi_signal_values(self, sample_ohlcv):
        """stoch_rsi_signal devrait retourner -1, 0, ou 1."""
        signal = stoch_rsi_signal(sample_ohlcv['close'])
        
        assert set(signal).issubset({-1, 0, 1})
    
    def test_calculate_stoch_rsi_returns_dict(self, sample_ohlcv):
        """calculate_stoch_rsi devrait retourner un dict."""
        result = calculate_stoch_rsi(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert 'd' in result
        assert 'signal' in result


# =============================================================================
# TESTS VORTEX
# =============================================================================

class TestVortex:
    """Tests pour l'indicateur Vortex."""
    
    def test_vortex_output_shape(self, sample_ohlcv):
        """Vortex devrait retourner VI+ et VI- de même taille."""
        vi_plus, vi_minus = vortex(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert len(vi_plus) == len(sample_ohlcv)
        assert len(vi_minus) == len(sample_ohlcv)
    
    def test_vortex_positive_values(self, sample_ohlcv):
        """VI+ et VI- devraient être positifs."""
        vi_plus, vi_minus = vortex(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        valid_plus = vi_plus[~np.isnan(vi_plus)]
        valid_minus = vi_minus[~np.isnan(vi_minus)]
        
        assert np.all(valid_plus >= 0)
        assert np.all(valid_minus >= 0)
    
    def test_vortex_movement_output(self, sample_ohlcv):
        """vortex_movement devrait retourner +VM, -VM, TR."""
        plus_vm, minus_vm, tr = vortex_movement(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert len(plus_vm) == len(sample_ohlcv)
        assert len(minus_vm) == len(sample_ohlcv)
        assert len(tr) == len(sample_ohlcv)
    
    def test_vortex_signal_values(self, sample_ohlcv):
        """vortex_signal devrait retourner -1, 0, ou 1."""
        signal = vortex_signal(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert set(signal).issubset({-1, 0, 1})
    
    def test_vortex_oscillator_bounds(self, sample_ohlcv):
        """vortex_oscillator devrait être entre -100 et 100."""
        osc = vortex_oscillator(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        valid_osc = osc[~np.isnan(osc)]
        assert np.all(valid_osc >= -100)
        assert np.all(valid_osc <= 100)
    
    def test_calculate_vortex_returns_dict(self, sample_ohlcv):
        """calculate_vortex devrait retourner un dict."""
        result = calculate_vortex(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert 'vi_plus' in result
        assert 'vi_minus' in result
        assert 'signal' in result
        assert 'oscillator' in result


# =============================================================================
# TESTS INDICATOR BANK
# =============================================================================

class TestIndicatorBank:
    """Tests pour le système de cache IndicatorBank."""
    
    def test_bank_creation(self, temp_cache_dir):
        """Le bank devrait se créer correctement."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        assert bank.enabled
        assert bank.cache_dir.exists()
    
    def test_bank_disabled(self, temp_cache_dir):
        """Le bank désactivé ne devrait rien cacher."""
        bank = IndicatorBank(cache_dir=temp_cache_dir, enabled=False)
        
        df = pd.DataFrame({'close': [1, 2, 3]})
        result = bank.get("test", {}, df)
        
        assert result is None
    
    def test_put_and_get(self, temp_cache_dir, sample_ohlcv):
        """put() puis get() devrait retourner les mêmes données."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        # Données à cacher
        test_result = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        params = {"period": 14}
        
        # Mettre en cache
        success = bank.put("test_indicator", params, sample_ohlcv, test_result)
        assert success
        
        # Récupérer
        cached = bank.get("test_indicator", params, sample_ohlcv)
        
        assert cached is not None
        np.testing.assert_array_equal(cached, test_result)
    
    def test_cache_miss(self, temp_cache_dir, sample_ohlcv):
        """get() sur entrée inexistante devrait retourner None."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        result = bank.get("nonexistent", {}, sample_ohlcv)
        
        assert result is None
        assert bank.stats.misses == 1
    
    def test_cache_hit_stats(self, temp_cache_dir, sample_ohlcv):
        """Les stats devraient tracker les hits/misses."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        test_result = [1, 2, 3]
        params = {"a": 1}
        
        # Miss
        bank.get("indicator", params, sample_ohlcv)
        assert bank.stats.misses == 1
        
        # Put
        bank.put("indicator", params, sample_ohlcv, test_result)
        
        # Hit
        bank.get("indicator", params, sample_ohlcv)
        assert bank.stats.hits == 1
        
        # Hit rate
        assert bank.stats.hit_rate == 50.0  # 1 hit, 1 miss
    
    def test_different_params_different_cache(self, temp_cache_dir, sample_ohlcv):
        """Paramètres différents = entrées de cache différentes."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        result1 = [1, 2, 3]
        result2 = [4, 5, 6]
        
        bank.put("indicator", {"period": 14}, sample_ohlcv, result1)
        bank.put("indicator", {"period": 21}, sample_ohlcv, result2)
        
        cached1 = bank.get("indicator", {"period": 14}, sample_ohlcv)
        cached2 = bank.get("indicator", {"period": 21}, sample_ohlcv)
        
        assert cached1 == result1
        assert cached2 == result2
    
    def test_invalidate_all(self, temp_cache_dir, sample_ohlcv):
        """invalidate() sans argument devrait vider le cache."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        bank.put("ind1", {}, sample_ohlcv, [1])
        bank.put("ind2", {}, sample_ohlcv, [2])
        
        count = bank.invalidate()
        
        assert count == 2
        assert bank.get("ind1", {}, sample_ohlcv) is None
        assert bank.get("ind2", {}, sample_ohlcv) is None
    
    def test_invalidate_specific(self, temp_cache_dir, sample_ohlcv):
        """invalidate(name) devrait invalider seulement cet indicateur."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        bank.put("ind1", {}, sample_ohlcv, [1])
        bank.put("ind2", {}, sample_ohlcv, [2])
        
        count = bank.invalidate("ind1")
        
        assert count == 1
        assert bank.get("ind1", {}, sample_ohlcv) is None
        assert bank.get("ind2", {}, sample_ohlcv) == [2]
    
    def test_list_entries(self, temp_cache_dir, sample_ohlcv):
        """list_entries() devrait lister toutes les entrées."""
        bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        bank.put("bollinger", {"period": 20}, sample_ohlcv, [1])
        bank.put("rsi", {"period": 14}, sample_ohlcv, [2])
        
        entries = bank.list_entries()
        
        assert len(entries) == 2
        indicators = [e["indicator"] for e in entries]
        assert "bollinger" in indicators
        assert "rsi" in indicators


class TestCachedIndicatorDecorator:
    """Tests pour le décorateur @cached_indicator."""
    
    def test_decorator_caches_result(self, temp_cache_dir, sample_ohlcv):
        """Le décorateur devrait cacher le résultat."""
        # Réinitialiser le bank global avec notre temp dir
        import data.indicator_bank as ib
        ib._default_bank = IndicatorBank(cache_dir=temp_cache_dir)
        
        call_count = [0]
        
        @cached_indicator
        def test_indicator(df, params):
            call_count[0] += 1
            return np.array([1, 2, 3])
        
        # Premier appel
        result1 = test_indicator(sample_ohlcv, {"a": 1})
        assert call_count[0] == 1
        
        # Deuxième appel (devrait être caché)
        result2 = test_indicator(sample_ohlcv, {"a": 1})
        assert call_count[0] == 1  # Pas d'appel supplémentaire
        
        np.testing.assert_array_equal(result1, result2)


# =============================================================================
# TESTS D'INTÉGRATION PHASE 2
# =============================================================================

class TestPhase2Integration:
    """Tests d'intégration pour la Phase 2."""
    
    def test_all_new_indicators_in_registry(self):
        """Tous les nouveaux indicateurs devraient être dans le registre."""
        from indicators.registry import list_indicators
        
        indicators = list_indicators()
        
        # Phase 2 indicators
        assert "ichimoku" in indicators
        assert "psar" in indicators
        assert "stoch_rsi" in indicators
        assert "vortex" in indicators
    
    def test_indicators_via_registry(self, sample_ohlcv):
        """Les indicateurs devraient fonctionner via calculate_indicator."""
        from indicators.registry import calculate_indicator
        
        # Ichimoku
        result = calculate_indicator("ichimoku", sample_ohlcv, {})
        assert isinstance(result, dict)
        assert 'tenkan' in result
        
        # PSAR
        result = calculate_indicator("psar", sample_ohlcv, {})
        assert isinstance(result, dict)
        assert 'sar' in result
        
        # Stoch RSI
        result = calculate_indicator("stoch_rsi", sample_ohlcv, {})
        assert isinstance(result, dict)
        assert 'k' in result
        
        # Vortex
        result = calculate_indicator("vortex", sample_ohlcv, {})
        assert isinstance(result, dict)
        assert 'vi_plus' in result
    
    def test_trending_market_signals(self, trending_up_data):
        """En tendance haussière, les indicateurs devraient donner des signaux cohérents."""
        # Ichimoku: prix au-dessus du cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(
            trending_up_data['high'],
            trending_up_data['low'],
            trending_up_data['close']
        )
        cloud_pos = ichimoku_cloud_position(trending_up_data['close'], senkou_a, senkou_b)
        valid_pos = cloud_pos[~np.isnan(cloud_pos)]
        if len(valid_pos) > 0:
            assert np.mean(valid_pos) > 0  # Majoritairement au-dessus
        
        # PSAR: tendance up
        sar, trend = parabolic_sar(
            trending_up_data['high'],
            trending_up_data['low'],
            trending_up_data['close']
        )
        valid_trend = trend[trend != 0][-20:]  # Dernières valeurs
        if len(valid_trend) > 0:
            assert np.mean(valid_trend) > 0  # Majoritairement up
        
        # Vortex: VI+ > VI-
        vi_plus, vi_minus = vortex(
            trending_up_data['high'],
            trending_up_data['low'],
            trending_up_data['close']
        )
        valid_mask = ~np.isnan(vi_plus) & ~np.isnan(vi_minus)
        if np.sum(valid_mask) > 0:
            diff = vi_plus[valid_mask][-20:] - vi_minus[valid_mask][-20:]
            assert np.mean(diff) > 0  # VI+ domine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
