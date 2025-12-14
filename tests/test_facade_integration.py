"""
Tests d'intégration pour la façade Backend.

Ces tests vérifient:
1. Le happy path backtest via façade
2. Les erreurs utilisateur remontent correctement
3. Le comportement quand LLM n'est pas disponible
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from backtest.facade import (
    BackendFacade,
    BacktestRequest,
    GridOptimizationRequest,
    LLMOptimizationRequest,
    BackendResponse,
    GridOptimizationResponse,
    LLMOptimizationResponse,
    ResponseStatus,
    ErrorCode,
    UIMetrics,
    UIPayload,
    get_facade,
    to_ui_payload,
)
from backtest.errors import (
    BacktestError,
    UserInputError,
    DataError,
    BackendInternalError,
    LLMUnavailableError,
    StrategyNotFoundError,
    ParameterValidationError,
)
from backtest.engine import RunResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Crée un DataFrame OHLCV de test."""
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    
    # Générer des prix réalistes
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_ = low + (high - low) * np.random.rand(n)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def facade():
    """Retourne une instance de façade."""
    return BackendFacade(debug=True)


@pytest.fixture
def mock_run_result(sample_ohlcv_df):
    """Crée un RunResult mocké."""
    n = len(sample_ohlcv_df)
    equity = pd.Series(
        10000 + np.cumsum(np.random.randn(n) * 10),
        index=sample_ohlcv_df.index
    )
    returns = equity.pct_change().fillna(0)
    trades = pd.DataFrame({
        "entry_ts": [sample_ohlcv_df.index[10], sample_ohlcv_df.index[100]],
        "exit_ts": [sample_ohlcv_df.index[50], sample_ohlcv_df.index[150]],
        "side": ["long", "short"],
        "pnl": [100, -50],
    })
    
    return RunResult(
        equity=equity,
        returns=returns,
        trades=trades,
        metrics={
            "total_pnl": 50.0,
            "total_return_pct": 0.5,
            "annualized_return": 5.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 2.0,
            "max_drawdown": 5.0,
            "volatility_annual": 10.0,
            "total_trades": 2,
            "win_rate": 50.0,
            "profit_factor": 2.0,
            "expectancy": 25.0,
        },
        meta={"strategy": "test_strategy"}
    )


# =============================================================================
# TESTS ERREURS
# =============================================================================

class TestBacktestErrors:
    """Tests pour le module d'erreurs."""
    
    def test_user_input_error_creation(self):
        """Test création UserInputError."""
        err = UserInputError(
            message="fast_period doit être < slow_period",
            param_name="fast_period",
            expected="< 20",
            got=25
        )
        
        assert "fast_period" in str(err)
        assert err.code == "INVALID_INPUT"
        assert err.param_name == "fast_period"
        assert err.got == 25
    
    def test_data_error_creation(self):
        """Test création DataError."""
        err = DataError(
            message="Fichier non trouvé",
            symbol="BTCUSDT",
            timeframe="1h",
            missing_columns=["close"]
        )
        
        assert err.code == "DATA_ERROR"
        assert err.symbol == "BTCUSDT"
        assert "close" in err.missing_columns
    
    def test_strategy_not_found_error(self):
        """Test création StrategyNotFoundError."""
        err = StrategyNotFoundError(
            strategy_name="fake_strategy",
            available=["ema_cross", "macd_cross"]
        )
        
        assert "fake_strategy" in str(err)
        assert "ema_cross" in err.hint
    
    def test_error_to_dict(self):
        """Test sérialisation en dict."""
        err = BacktestError(
            message="Test error",
            code="TEST",
            hint="Fix it"
        )
        
        d = err.to_dict()
        assert d["code"] == "TEST"
        assert d["message"] == "Test error"
        assert d["hint"] == "Fix it"


# =============================================================================
# TESTS UI TYPES
# =============================================================================

class TestUITypes:
    """Tests pour UIMetrics et UIPayload."""
    
    def test_ui_metrics_from_run_result(self, mock_run_result):
        """Test conversion RunResult -> UIMetrics."""
        metrics = UIMetrics.from_run_result(mock_run_result)
        
        assert metrics.total_pnl == 50.0
        assert metrics.sharpe_ratio == 1.2
        assert metrics.total_trades == 2
        assert metrics.win_rate == 50.0
    
    def test_ui_metrics_to_dict(self, mock_run_result):
        """Test UIMetrics.to_dict()."""
        metrics = UIMetrics.from_run_result(mock_run_result)
        d = metrics.to_dict()
        
        assert "sharpe_ratio" in d
        assert "total_pnl" in d
        assert d["sharpe_ratio"] == 1.2
    
    def test_ui_payload_from_run_result(self, mock_run_result):
        """Test conversion RunResult -> UIPayload."""
        payload = UIPayload.from_run_result(mock_run_result)
        
        assert payload.metrics.sharpe_ratio == 1.2
        assert payload.equity_series is not None
        assert payload.trades_df is not None
        assert len(payload.trades_df) == 2
    
    def test_to_ui_payload_helper(self, mock_run_result):
        """Test fonction helper to_ui_payload()."""
        payload = to_ui_payload(mock_run_result)
        
        assert isinstance(payload, UIPayload)
        assert payload.metrics.total_pnl == 50.0


# =============================================================================
# TESTS FACADE - BACKTEST SIMPLE
# =============================================================================

class TestBackendFacadeBacktest:
    """Tests pour run_backtest()."""
    
    def test_happy_path_backtest(self, facade, sample_ohlcv_df):
        """Test backtest réussi via façade."""
        request = BacktestRequest(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=sample_ohlcv_df,
            initial_capital=10000.0,
        )
        
        response = facade.run_backtest(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert response.payload is not None
        assert response.error is None
        assert response.payload.metrics.total_trades >= 0
        assert response.duration_ms > 0
    
    def test_invalid_strategy_error(self, facade, sample_ohlcv_df):
        """Test erreur stratégie invalide."""
        request = BacktestRequest(
            strategy_name="nonexistent_strategy",
            params={},
            data=sample_ohlcv_df,
        )
        
        response = facade.run_backtest(request)
        
        assert response.status == ResponseStatus.ERROR
        assert response.error is not None
        assert response.error.code == ErrorCode.STRATEGY_NOT_FOUND
        assert "nonexistent" in response.error.message_user.lower()
    
    def test_empty_dataframe_error(self, facade):
        """Test erreur DataFrame vide."""
        empty_df = pd.DataFrame()
        
        request = BacktestRequest(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=empty_df,
        )
        
        response = facade.run_backtest(request)
        
        assert response.status == ResponseStatus.ERROR
        assert response.error.code == ErrorCode.INVALID_DATA
    
    def test_missing_columns_error(self, facade):
        """Test erreur colonnes manquantes."""
        bad_df = pd.DataFrame({
            "close": [100, 101, 102],
            # manque open, high, low, volume
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))
        
        request = BacktestRequest(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=bad_df,
        )
        
        response = facade.run_backtest(request)
        
        assert response.status == ResponseStatus.ERROR
        assert response.error.code == ErrorCode.INVALID_DATA
    
    def test_request_validation_missing_data(self):
        """Test validation requête - ni data ni symbol."""
        with pytest.raises(ValueError, match="requis"):
            BacktestRequest(
                strategy_name="ema_cross",
                params={},
                # ni data, ni symbol/timeframe
            )
    
    def test_response_is_success_property(self, facade, sample_ohlcv_df):
        """Test propriétés is_success/is_error."""
        request = BacktestRequest(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=sample_ohlcv_df,
        )
        
        response = facade.run_backtest(request)
        
        assert response.is_success == True
        assert response.is_error == False


# =============================================================================
# TESTS FACADE - OPTIMISATION GRILLE
# =============================================================================

class TestBackendFacadeGridOptimization:
    """Tests pour run_grid_optimization()."""
    
    def test_grid_optimization_success(self, facade, sample_ohlcv_df):
        """Test optimisation grille réussie."""
        param_grid = [
            {"fast_period": 5, "slow_period": 20},
            {"fast_period": 10, "slow_period": 30},
            {"fast_period": 15, "slow_period": 40},
        ]
        
        request = GridOptimizationRequest(
            strategy_name="ema_cross",
            param_grid=param_grid,
            data=sample_ohlcv_df,
        )
        
        response = facade.run_grid_optimization(request)
        
        assert response.status in [ResponseStatus.SUCCESS, ResponseStatus.PARTIAL]
        assert response.total_tested == 3
        assert response.total_success >= 1
        assert response.best_result is not None
        assert len(response.results) == 3
    
    def test_grid_optimization_with_progress(self, facade, sample_ohlcv_df):
        """Test callback de progression."""
        progress_calls = []
        
        def progress_cb(current, total):
            progress_calls.append((current, total))
        
        param_grid = [
            {"fast_period": 5, "slow_period": 20},
            {"fast_period": 10, "slow_period": 30},
        ]
        
        request = GridOptimizationRequest(
            strategy_name="ema_cross",
            param_grid=param_grid,
            data=sample_ohlcv_df,
        )
        
        facade.run_grid_optimization(request, progress_callback=progress_cb)
        
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)
    
    def test_grid_respects_max_combinations(self, facade, sample_ohlcv_df):
        """Test limite max_combinations."""
        param_grid = [{"fast_period": i, "slow_period": i + 20} for i in range(5, 50)]
        
        request = GridOptimizationRequest(
            strategy_name="ema_cross",
            param_grid=param_grid,
            data=sample_ohlcv_df,
            max_combinations=5,
        )
        
        response = facade.run_grid_optimization(request)
        
        assert response.total_tested == 5


# =============================================================================
# TESTS FACADE - OPTIMISATION LLM
# =============================================================================

class TestBackendFacadeLLMOptimization:
    """Tests pour run_llm_optimization()."""
    
    def test_llm_unavailable_graceful(self, sample_ohlcv_df):
        """Test que l'UI ne crashe pas si LLM indisponible."""
        facade = BackendFacade()
        
        request = LLMOptimizationRequest(
            strategy_name="ema_cross",
            initial_params={"fast_period": 10, "slow_period": 21},
            param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
            data=sample_ohlcv_df,
            llm_provider="ollama",
            llm_model="llama3.2",
        )
        
        # Même si le module agents existe, la connexion peut échouer
        # Le test vérifie que ça retourne une réponse propre, pas un crash
        response = facade.run_llm_optimization(request)
        
        # Soit SUCCESS (LLM disponible) soit ERROR propre
        assert response.status in [ResponseStatus.SUCCESS, ResponseStatus.ERROR]
        
        if response.status == ResponseStatus.ERROR:
            # Erreur doit être structurée
            assert response.error is not None
            assert response.error.code in [
                ErrorCode.LLM_UNAVAILABLE,
                ErrorCode.LLM_CONNECTION_FAILED,
                ErrorCode.OPTIMIZATION_FAILED,
            ]
            assert response.error.message_user != ""
    
    @patch("agents.integration.create_optimizer_from_engine")
    def test_llm_optimization_mocked(self, mock_create, facade, sample_ohlcv_df):
        """Test optimisation LLM avec mock."""
        # Setup mock
        mock_strategist = Mock()
        mock_executor = Mock()
        mock_create.return_value = (mock_strategist, mock_executor)
        
        mock_result = Mock()
        mock_result.sharpe_ratio = 2.5
        mock_result.total_pnl = 500.0
        mock_result.request = Mock()
        mock_result.request.parameters = {"fast_period": 12, "slow_period": 26}
        
        mock_session = Mock()
        mock_session.best_result = mock_result
        mock_session.history = [mock_result]
        mock_session.total_iterations = 5
        mock_session.convergence_reason = "target_reached"
        
        mock_strategist.optimize.return_value = mock_session
        
        request = LLMOptimizationRequest(
            strategy_name="ema_cross",
            initial_params={"fast_period": 10, "slow_period": 21},
            param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
            data=sample_ohlcv_df,
        )
        
        response = facade.run_llm_optimization(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert response.best_params["fast_period"] == 12
        assert response.total_iterations == 5


# =============================================================================
# TESTS GET_FACADE SINGLETON
# =============================================================================

class TestGetFacade:
    """Tests pour le singleton get_facade()."""
    
    def test_returns_same_instance(self):
        """Test que get_facade retourne toujours la même instance."""
        # Reset le singleton
        import backtest.facade as facade_module
        facade_module._facade_instance = None
        
        f1 = get_facade()
        f2 = get_facade()
        
        assert f1 is f2
    
    def test_uses_config_on_first_call(self):
        """Test que la config est utilisée à la première création."""
        import backtest.facade as facade_module
        facade_module._facade_instance = None
        
        from utils.config import Config
        custom_config = Config(fees_bps=20)
        
        f = get_facade(config=custom_config)
        
        assert f.config.fees_bps == 20
