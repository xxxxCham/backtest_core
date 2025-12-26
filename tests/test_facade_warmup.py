"""
Tests pour la validation du warmup dans BackendFacade.

Vérifie que:
1. Les fenêtres trop courtes sont détectées et neutralisées
2. InsufficientDataError est levée si les données sont insuffisantes
3. Les données suffisantes passent la validation
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from backtest.facade import BackendFacade, BacktestRequest, WARMUP_MIN_DEFAULT
from backtest.errors import InsufficientDataError, DataError


@pytest.fixture
def facade():
    """Crée une instance de BackendFacade pour les tests."""
    return BackendFacade(debug=True)


@pytest.fixture
def valid_ohlcv_data():
    """Crée un DataFrame OHLCV valide avec 250 barres."""
    dates = pd.date_range("2024-01-01", periods=250, freq="1h")
    return pd.DataFrame({
        "open": [100.0] * 250,
        "high": [101.0] * 250,
        "low": [99.0] * 250,
        "close": [100.5] * 250,
        "volume": [1000.0] * 250,
    }, index=dates)


@pytest.fixture
def insufficient_ohlcv_data():
    """Crée un DataFrame OHLCV avec seulement 50 barres (insuffisant)."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    return pd.DataFrame({
        "open": [100.0] * 50,
        "high": [101.0] * 50,
        "low": [99.0] * 50,
        "close": [100.5] * 50,
        "volume": [1000.0] * 50,
    }, index=dates)


def test_estimate_bars_between_1h_timeframe(facade):
    """Teste l'estimation du nombre de barres pour un timeframe 1h."""
    start = "2024-01-01"
    end = "2024-01-03"  # 2 jours = 48 heures

    estimated = facade._estimate_bars_between(start, end, "1h")

    assert estimated == 48


def test_estimate_bars_between_4h_timeframe(facade):
    """Teste l'estimation du nombre de barres pour un timeframe 4h."""
    start = "2024-01-01"
    end = "2024-01-11"  # 10 jours = 240 heures

    estimated = facade._estimate_bars_between(start, end, "4h")

    assert estimated == 60  # 240 / 4 = 60


def test_estimate_bars_between_1d_timeframe(facade):
    """Teste l'estimation du nombre de barres pour un timeframe 1d."""
    start = "2024-01-01"
    end = "2024-02-01"  # ~31 jours

    estimated = facade._estimate_bars_between(start, end, "1d")

    assert estimated == 31


def test_load_data_short_window_neutralized(facade):
    """Teste que les fenêtres trop courtes sont neutralisées."""
    with patch("data.loader.load_ohlcv") as mock_load:
        # Mock: première fois avec fenêtre courte (ignoré), deuxième fois sans dates
        mock_load.return_value = pd.DataFrame({
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.5] * 250,
            "volume": [1000.0] * 250,
        }, index=pd.date_range("2024-01-01", periods=250, freq="1h"))

        # Fenêtre de 2 jours en 1h = 48 barres < 200 requis
        df = facade._load_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start="2024-01-01",
            end="2024-01-03"
        )

        # Vérifier que load_ohlcv a été appelé avec start=None, end=None
        # (les dates ont été neutralisées)
        mock_load.assert_called_once_with("BTCUSDT", "1h", start=None, end=None)

        # Le DataFrame retourné doit avoir 250 barres (suffisant)
        assert len(df) == 250


def test_load_data_sufficient_window_unchanged(facade):
    """Teste que les fenêtres suffisantes ne sont pas modifiées."""
    with patch("data.loader.load_ohlcv") as mock_load:
        mock_load.return_value = pd.DataFrame({
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.5] * 250,
            "volume": [1000.0] * 250,
        }, index=pd.date_range("2024-01-01", periods=250, freq="1h"))

        # Fenêtre de 20 jours en 1h = 480 barres > 200 requis
        start = "2024-01-01"
        end = "2024-01-21"

        df = facade._load_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start=start,
            end=end
        )

        # Vérifier que les dates n'ont PAS été neutralisées
        mock_load.assert_called_once_with("BTCUSDT", "1h", start=start, end=end)
        assert len(df) == 250


def test_load_data_insufficient_raises_error(facade):
    """Teste qu'InsufficientDataError est levée si données insuffisantes."""
    with patch("data.loader.load_ohlcv") as mock_load:
        # Retourner seulement 50 barres (< 200 requis)
        mock_load.return_value = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.5] * 50,
            "volume": [1000.0] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        with pytest.raises(InsufficientDataError) as exc_info:
            facade._load_data(
                symbol="BTCUSDT",
                timeframe="1h",
                start=None,
                end=None
            )

        # Vérifier les attributs de l'exception
        assert exc_info.value.available_bars == 50
        assert exc_info.value.required_bars == WARMUP_MIN_DEFAULT
        assert "50 barres < 200 requis" in str(exc_info.value)


def test_validate_dataframe_with_warmup_sufficient(facade, valid_ohlcv_data):
    """Teste que la validation passe avec des données suffisantes."""
    # Ne doit pas lever d'exception
    facade._validate_dataframe(
        valid_ohlcv_data,
        warmup_required=200,
        symbol="BTCUSDT",
        timeframe="1h"
    )


def test_validate_dataframe_with_warmup_insufficient(facade, insufficient_ohlcv_data):
    """Teste qu'InsufficientDataError est levée avec données insuffisantes."""
    with pytest.raises(InsufficientDataError) as exc_info:
        facade._validate_dataframe(
            insufficient_ohlcv_data,
            warmup_required=200,
            symbol="BTCUSDT",
            timeframe="1h"
        )

    assert exc_info.value.available_bars == 50
    assert exc_info.value.required_bars == 200


def test_validate_dataframe_without_warmup_check(facade, insufficient_ohlcv_data):
    """Teste que la validation passe sans check warmup (backward compat)."""
    # Ne doit pas lever d'exception si warmup_required=None
    facade._validate_dataframe(
        insufficient_ohlcv_data,
        warmup_required=None,
        symbol="BTCUSDT",
        timeframe="1h"
    )


def test_validate_dataframe_empty_raises_data_error(facade):
    """Teste que DataError est levée pour DataFrame vide."""
    empty_df = pd.DataFrame()

    with pytest.raises(DataError) as exc_info:
        facade._validate_dataframe(empty_df)

    assert "vide" in str(exc_info.value).lower()


def test_validate_dataframe_missing_columns_raises_error(facade):
    """Teste que DataError est levée pour colonnes manquantes."""
    invalid_df = pd.DataFrame({
        "open": [100.0] * 100,
        "close": [100.0] * 100,
        # Manquants: high, low, volume
    }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

    with pytest.raises(DataError) as exc_info:
        facade._validate_dataframe(invalid_df)

    assert "manquantes" in str(exc_info.value).lower()


def test_backtest_request_with_insufficient_data_returns_error(facade):
    """Teste que run_backtest retourne ErrorCode.INSUFFICIENT_DATA."""
    with patch("data.loader.load_ohlcv") as mock_load:
        # Retourner données insuffisantes
        mock_load.return_value = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.5] * 50,
            "volume": [1000.0] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        request = BacktestRequest(
            strategy_name="bollinger_atr",
            params={"period": 20, "atr_period": 14},
            symbol="BTCUSDT",
            timeframe="1h",
        )

        response = facade.run_backtest(request)

        assert response.status.value == "error"
        assert response.error is not None
        assert response.error.code.value == "insufficient_data"
        assert "50 barres < 200 requis" in response.error.message_user


def test_custom_warmup_requirement(facade):
    """Teste qu'on peut spécifier un warmup custom."""
    with patch("data.loader.load_ohlcv") as mock_load:
        # 150 barres: suffisant pour warmup=100, insuffisant pour warmup=200
        mock_load.return_value = pd.DataFrame({
            "open": [100.0] * 150,
            "high": [101.0] * 150,
            "low": [99.0] * 150,
            "close": [100.5] * 150,
            "volume": [1000.0] * 150,
        }, index=pd.date_range("2024-01-01", periods=150, freq="1h"))

        # Avec warmup_required=100 -> devrait passer
        df = facade._load_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start=None,
            end=None,
            warmup_required=100
        )
        assert len(df) == 150

        # Avec warmup_required=200 -> devrait échouer
        with pytest.raises(InsufficientDataError):
            facade._load_data(
                symbol="BTCUSDT",
                timeframe="1h",
                start=None,
                end=None,
                warmup_required=200
            )
