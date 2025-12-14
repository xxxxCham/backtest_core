"""
Tests d'intégration pour le système autonome avec le vrai moteur.

Ces tests vérifient que:
1. run_backtest_for_agent() utilise vraiment BacktestEngine
2. run_walk_forward_for_agent() utilise vraiment WalkForwardValidator
3. Le système complet fonctionne de bout en bout
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import json


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_ohlcv_data():
    """Génère des données OHLCV réalistes pour les tests."""
    np.random.seed(42)
    n = 500
    
    # Prix simulés avec tendance
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # Créer OHLCV
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    
    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 0.3),
        "low": price - np.abs(np.random.randn(n) * 0.3),
        "close": price + np.random.randn(n) * 0.2,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    
    return df


# ============================================================
# Tests run_backtest_for_agent
# ============================================================

class TestRunBacktestForAgent:
    """Tests pour la fonction wrapper du backtest."""
    
    def test_basic_backtest(self, sample_ohlcv_data):
        """Test qu'un backtest basique fonctionne."""
        from agents.integration import run_backtest_for_agent
        
        result = run_backtest_for_agent(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=sample_ohlcv_data,
        )
        
        # Vérifier que les métriques sont présentes
        assert "sharpe_ratio" in result
        assert "total_return" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "total_trades" in result
        
        # Vérifier que ce sont des valeurs numériques
        assert isinstance(result["sharpe_ratio"], (int, float))
        assert isinstance(result["total_return"], (int, float))
    
    def test_different_strategies(self, sample_ohlcv_data):
        """Test avec différentes stratégies."""
        from agents.integration import run_backtest_for_agent
        from strategies.base import list_strategies
        
        # Tester quelques stratégies disponibles
        strategies_to_test = ["ema_cross", "rsi_reversal"]
        
        for strategy in strategies_to_test:
            if strategy in list_strategies():
                result = run_backtest_for_agent(
                    strategy_name=strategy,
                    params={},  # Utiliser les défauts
                    data=sample_ohlcv_data,
                )
                assert "sharpe_ratio" in result
    
    def test_invalid_strategy_raises(self, sample_ohlcv_data):
        """Test qu'une stratégie invalide lève une erreur."""
        from agents.integration import run_backtest_for_agent
        
        with pytest.raises(ValueError):
            run_backtest_for_agent(
                strategy_name="strategie_qui_nexiste_pas",
                params={},
                data=sample_ohlcv_data,
            )
    
    def test_custom_capital(self, sample_ohlcv_data):
        """Test avec un capital personnalisé."""
        from agents.integration import run_backtest_for_agent
        
        result = run_backtest_for_agent(
            strategy_name="ema_cross",
            params={},
            data=sample_ohlcv_data,
            initial_capital=50000.0,
        )
        
        assert "sharpe_ratio" in result


# ============================================================
# Tests run_walk_forward_for_agent
# ============================================================

class TestRunWalkForwardForAgent:
    """Tests pour la validation walk-forward."""
    
    def test_basic_walk_forward(self, sample_ohlcv_data):
        """Test qu'une validation walk-forward basique fonctionne."""
        from agents.integration import run_walk_forward_for_agent
        
        result = run_walk_forward_for_agent(
            strategy_name="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            data=sample_ohlcv_data,
            n_windows=3,
        )
        
        # Vérifier les métriques de validation
        assert "train_sharpe" in result
        assert "test_sharpe" in result
        assert "overfitting_ratio" in result
        
        # Vérifier que ce sont des valeurs numériques
        assert isinstance(result["train_sharpe"], (int, float))
        assert isinstance(result["test_sharpe"], (int, float))
        assert isinstance(result["overfitting_ratio"], (int, float))
    
    def test_multiple_folds(self, sample_ohlcv_data):
        """Test avec plusieurs folds."""
        from agents.integration import run_walk_forward_for_agent
        
        # Créer des données plus grandes pour les folds
        np.random.seed(42)
        n = 2000  # Données plus grandes
        price = 100 + np.cumsum(np.random.randn(n) * 0.5)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        large_data = pd.DataFrame({
            "open": price,
            "high": price + np.abs(np.random.randn(n) * 0.3),
            "low": price - np.abs(np.random.randn(n) * 0.3),
            "close": price + np.random.randn(n) * 0.2,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }, index=dates)
        
        result = run_walk_forward_for_agent(
            strategy_name="ema_cross",
            params={},
            data=large_data,
            n_windows=3,  # Moins de fenêtres pour éviter les données trop petites
            train_ratio=0.7,
        )
        
        # Les métriques doivent être présentes
        assert "train_sharpe" in result
        assert "test_sharpe" in result
        assert "overfitting_ratio" in result


# ============================================================
# Tests get_strategy_param_bounds
# ============================================================

class TestGetStrategyParamBounds:
    """Tests pour la récupération des bornes de paramètres."""
    
    def test_ema_cross_bounds(self):
        """Test les bornes pour ema_cross."""
        from agents.integration import get_strategy_param_bounds
        
        bounds = get_strategy_param_bounds("ema_cross")
        
        # Devrait avoir des bornes pour les paramètres
        assert isinstance(bounds, dict)
        # Au minimum, il devrait y avoir quelque chose
        # (soit des parameter_specs, soit des defaults)
    
    def test_invalid_strategy_raises(self):
        """Test qu'une stratégie invalide lève une erreur."""
        from agents.integration import get_strategy_param_bounds
        
        with pytest.raises(ValueError):
            get_strategy_param_bounds("strategie_inexistante")


# ============================================================
# Tests create_optimizer_from_engine
# ============================================================

class TestCreateOptimizerFromEngine:
    """Tests pour la factory complète."""
    
    def test_create_optimizer(self, sample_ohlcv_data):
        """Test création d'un optimiseur."""
        from agents.integration import create_optimizer_from_engine
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestExecutor
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
        
        # Note: Ceci va créer les objets mais pas appeler Ollama
        strategist, executor = create_optimizer_from_engine(
            llm_config=config,
            strategy_name="ema_cross",
            data=sample_ohlcv_data,
            use_walk_forward=True,
        )
        
        assert isinstance(strategist, AutonomousStrategist)
        assert isinstance(executor, BacktestExecutor)
    
    def test_invalid_strategy_raises(self, sample_ohlcv_data):
        """Test qu'une stratégie invalide lève une erreur."""
        from agents.integration import create_optimizer_from_engine
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test")
        
        with pytest.raises(ValueError, match="inconnue"):
            create_optimizer_from_engine(
                llm_config=config,
                strategy_name="strategie_inexistante",
                data=sample_ohlcv_data,
            )
    
    def test_executor_can_run_backtest(self, sample_ohlcv_data):
        """Test que l'exécuteur peut vraiment lancer des backtests."""
        from agents.integration import create_optimizer_from_engine
        from agents.backtest_executor import BacktestRequest
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test")
        
        _, executor = create_optimizer_from_engine(
            llm_config=config,
            strategy_name="ema_cross",
            data=sample_ohlcv_data,
            use_walk_forward=False,  # Désactiver pour simplifier
        )
        
        # Créer une requête
        request = BacktestRequest(
            hypothesis="Test initial",
            parameters={"fast_period": 10, "slow_period": 21},
        )
        
        # Exécuter le backtest
        result = executor.run(request)
        
        # Vérifier le résultat
        assert result.success is True
        assert result.sharpe_ratio != 0 or result.total_trades == 0
        assert executor.history.total_experiments == 1


# ============================================================
# Tests d'intégration complets
# ============================================================

class TestFullIntegration:
    """Tests d'intégration de bout en bout."""
    
    def test_full_optimization_with_mock_llm(self, sample_ohlcv_data):
        """Test complet avec LLM mocké."""
        from agents.integration import create_optimizer_from_engine
        from agents.llm_client import LLMConfig, LLMProvider, LLMResponse
        from unittest.mock import Mock
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test")
        
        strategist, executor = create_optimizer_from_engine(
            llm_config=config,
            strategy_name="ema_cross",
            data=sample_ohlcv_data,
            use_walk_forward=False,
        )
        
        # Mocker le LLM pour retourner des décisions prédéfinies
        iteration = [0]
        
        def mock_chat(*args, **kwargs):
            iteration[0] += 1
            if iteration[0] < 3:
                return LLMResponse(
                    content=json.dumps({
                        "action": "continue",
                        "confidence": 0.7,
                        "reasoning": f"Iteration {iteration[0]}",
                        "next_hypothesis": "Try different params",
                        "next_parameters": {
                            "fast_period": 8 + iteration[0],
                            "slow_period": 20 + iteration[0],
                        },
                    }),
                    model="test",
                    provider=LLMProvider.OLLAMA,
                    prompt_tokens=50,
                    completion_tokens=50,
                    total_tokens=100,
                )
            else:
                return LLMResponse(
                    content=json.dumps({
                        "action": "accept",
                        "confidence": 0.9,
                        "reasoning": "Good result found",
                    }),
                    model="test",
                    provider=LLMProvider.OLLAMA,
                    prompt_tokens=50,
                    completion_tokens=50,
                    total_tokens=100,
                )
        
        strategist.llm.chat = mock_chat
        
        # Lancer l'optimisation
        session = strategist.optimize(
            executor=executor,
            initial_params={"fast_period": 10, "slow_period": 21},
            param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
            max_iterations=5,
        )
        
        # Vérifications
        assert session.final_status == "success"
        assert session.best_result is not None
        assert len(session.all_results) >= 2
        
        # Vérifier que les backtests ont vraiment été exécutés
        assert executor.history.total_experiments >= 2
    
    def test_executor_tracks_history(self, sample_ohlcv_data):
        """Test que l'historique est bien tracké."""
        from agents.integration import create_optimizer_from_engine
        from agents.backtest_executor import BacktestRequest
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test")
        
        _, executor = create_optimizer_from_engine(
            llm_config=config,
            strategy_name="ema_cross",
            data=sample_ohlcv_data,
        )
        
        # Exécuter plusieurs backtests
        params_list = [
            {"fast_period": 8, "slow_period": 21},
            {"fast_period": 10, "slow_period": 21},
            {"fast_period": 12, "slow_period": 21},
        ]
        
        for params in params_list:
            request = BacktestRequest(
                hypothesis=f"Test fast={params['fast_period']}",
                parameters=params,
            )
            executor.run(request)
        
        # Vérifier l'historique
        assert executor.history.total_experiments == 3
        assert executor.history.best_result is not None
        
        # Vérifier le contexte pour l'agent
        context = executor.get_context_for_agent()
        assert "3 total" in context
        assert "Best Configuration" in context
