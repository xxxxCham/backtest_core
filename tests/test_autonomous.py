"""
Tests pour le système autonome (BacktestExecutor + AutonomousStrategist).

Ces tests vérifient que le LLM peut vraiment lancer des backtests,
analyser les résultats, et itérer intelligemment.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# Tests BacktestExecutor
# ============================================================

class TestBacktestRequest:
    """Tests pour BacktestRequest."""
    
    def test_request_creation(self):
        """Test création d'une requête basique."""
        from agents.backtest_executor import BacktestRequest
        
        request = BacktestRequest(
            requested_by="strategist",
            hypothesis="Test increasing fast period",
            parameters={"fast": 12, "slow": 26},
        )
        
        assert request.requested_by == "strategist"
        assert request.hypothesis == "Test increasing fast period"
        assert request.parameters == {"fast": 12, "slow": 26}
        assert request.request_id  # Auto-généré
    
    def test_request_id_deterministic(self):
        """Test que l'ID est déterministe pour les mêmes params."""
        from agents.backtest_executor import BacktestRequest
        
        request1 = BacktestRequest(
            hypothesis="Test 1",
            parameters={"fast": 10, "slow": 21},
        )
        request2 = BacktestRequest(
            hypothesis="Test 2",  # Différent
            parameters={"fast": 10, "slow": 21},  # Même params
        )
        
        # Même ID car basé sur les params
        assert request1.request_id == request2.request_id
    
    def test_request_id_different_params(self):
        """Test que l'ID change avec des params différents."""
        from agents.backtest_executor import BacktestRequest
        
        request1 = BacktestRequest(parameters={"fast": 10})
        request2 = BacktestRequest(parameters={"fast": 12})
        
        assert request1.request_id != request2.request_id


class TestBacktestResult:
    """Tests pour BacktestResult."""
    
    def test_result_to_summary(self):
        """Test conversion en dict pour LLM."""
        from agents.backtest_executor import BacktestRequest, BacktestResult
        
        request = BacktestRequest(
            hypothesis="Testing",
            parameters={"fast": 10, "slow": 21},
        )
        
        result = BacktestResult(
            request=request,
            success=True,
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=0.15,
            win_rate=0.55,
            total_trades=100,
            execution_time_ms=150.5,
        )
        
        summary = result.to_summary_dict()
        
        assert summary["success"] is True
        assert summary["sharpe_ratio"] == 1.5
        assert "25" in summary["total_return"]  # "25.00%"
        assert summary["total_trades"] == 100
    
    def test_result_to_analysis_prompt(self):
        """Test génération du prompt d'analyse."""
        from agents.backtest_executor import BacktestRequest, BacktestResult
        
        request = BacktestRequest(
            hypothesis="Testing crossover period",
            parameters={"fast": 10},
        )
        
        result = BacktestResult(
            request=request,
            success=True,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            total_return=0.30,
            max_drawdown=0.10,
            win_rate=0.60,
        )
        
        prompt = result.to_analysis_prompt()
        
        assert "Testing crossover period" in prompt
        assert "Sharpe Ratio: 1.200" in prompt
        assert "Sortino Ratio: 1.800" in prompt
        assert "30.00%" in prompt
    
    def test_result_with_overfitting(self):
        """Test affichage overfitting dans le prompt."""
        from agents.backtest_executor import BacktestRequest, BacktestResult
        
        request = BacktestRequest(hypothesis="Test", parameters={})
        
        result = BacktestResult(
            request=request,
            success=True,
            sharpe_ratio=2.0,
            train_sharpe=2.5,
            test_sharpe=1.0,
            overfitting_ratio=2.5,
        )
        
        prompt = result.to_analysis_prompt()
        
        assert "Walk-Forward Analysis" in prompt
        assert "Train Sharpe: 2.500" in prompt
        assert "Test Sharpe: 1.000" in prompt
        assert "OVERFITTING DETECTED" in prompt


class TestExperimentHistory:
    """Tests pour ExperimentHistory."""
    
    def test_empty_history(self):
        """Test historique vide."""
        from agents.backtest_executor import ExperimentHistory
        
        history = ExperimentHistory()
        
        assert history.total_experiments == 0
        assert history.best_result is None
        assert history.get_tried_parameters() == []
    
    def test_add_result(self):
        """Test ajout d'un résultat."""
        from agents.backtest_executor import (
            ExperimentHistory, BacktestRequest, BacktestResult
        )
        
        history = ExperimentHistory()
        
        request = BacktestRequest(
            hypothesis="Test",
            parameters={"fast": 10},
        )
        result = BacktestResult(
            request=request,
            success=True,
            sharpe_ratio=1.5,
            execution_time_ms=100,
        )
        
        history.add_result(result)
        
        assert history.total_experiments == 1
        assert history.best_result == result
        assert history.best_sharpe == 1.5
    
    def test_best_tracking(self):
        """Test que le meilleur résultat est bien tracké."""
        from agents.backtest_executor import (
            ExperimentHistory, BacktestRequest, BacktestResult
        )
        
        history = ExperimentHistory()
        
        # Premier résultat
        r1 = BacktestResult(
            request=BacktestRequest(hypothesis="1", parameters={}),
            success=True,
            sharpe_ratio=1.0,
        )
        history.add_result(r1)
        
        # Meilleur résultat
        r2 = BacktestResult(
            request=BacktestRequest(hypothesis="2", parameters={}),
            success=True,
            sharpe_ratio=2.0,
        )
        history.add_result(r2)
        
        # Résultat moins bon
        r3 = BacktestResult(
            request=BacktestRequest(hypothesis="3", parameters={}),
            success=True,
            sharpe_ratio=1.5,
        )
        history.add_result(r3)
        
        assert history.best_result == r2
        assert history.best_sharpe == 2.0
    
    def test_overfitting_rejection(self):
        """Test que les résultats overfit ne sont pas best."""
        from agents.backtest_executor import (
            ExperimentHistory, BacktestRequest, BacktestResult
        )
        
        history = ExperimentHistory()
        
        # Résultat correct
        r1 = BacktestResult(
            request=BacktestRequest(hypothesis="1", parameters={}),
            success=True,
            sharpe_ratio=1.0,
            overfitting_ratio=1.2,  # Acceptable
        )
        history.add_result(r1)
        
        # Résultat overfit (meilleur Sharpe mais overfit)
        r2 = BacktestResult(
            request=BacktestRequest(hypothesis="2", parameters={}),
            success=True,
            sharpe_ratio=3.0,  # Beaucoup mieux
            overfitting_ratio=2.0,  # Mais overfit
        )
        history.add_result(r2)
        
        # Le premier devrait rester le best
        assert history.best_result == r1
    
    def test_summary_for_llm(self):
        """Test génération du résumé pour LLM."""
        from agents.backtest_executor import (
            ExperimentHistory, BacktestRequest, BacktestResult
        )
        
        history = ExperimentHistory()
        
        for i in range(3):
            result = BacktestResult(
                request=BacktestRequest(
                    hypothesis=f"Test {i}",
                    parameters={"x": i},
                ),
                success=True,
                sharpe_ratio=1.0 + i * 0.1,
            )
            history.add_result(result)
        
        summary = history.get_summary_for_llm()
        
        assert "3 total" in summary
        assert "Best Configuration" in summary
        assert "Last" in summary
    
    def test_parameter_sensitivity(self):
        """Test analyse de sensibilité."""
        from agents.backtest_executor import (
            ExperimentHistory, BacktestRequest, BacktestResult
        )
        
        history = ExperimentHistory()
        
        # Créer des résultats avec corrélation claire
        for x in [10, 12, 14, 16, 18]:
            result = BacktestResult(
                request=BacktestRequest(
                    hypothesis=f"Test x={x}",
                    parameters={"x": x},
                ),
                success=True,
                sharpe_ratio=x * 0.1,  # Corrélation positive parfaite
            )
            history.add_result(result)
        
        sensitivity = history.analyze_parameter_sensitivity()
        
        assert "x" in sensitivity
        assert sensitivity["x"]["direction"] == "positive"
        assert sensitivity["x"]["impact"] > 0.9  # Haute corrélation


class TestBacktestExecutor:
    """Tests pour BacktestExecutor."""
    
    @pytest.fixture
    def mock_backtest_fn(self):
        """Mock de fonction de backtest."""
        def backtest(strategy, params, data):
            # Simule un backtest
            return {
                "sharpe_ratio": 1.5 + params.get("fast", 10) * 0.01,
                "sortino_ratio": 1.8,
                "total_return": 0.25,
                "max_drawdown": 0.15,
                "win_rate": 0.55,
                "profit_factor": 1.8,
                "total_trades": 100,
            }
        return backtest
    
    @pytest.fixture
    def sample_data(self):
        """Données OHLCV de test."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, n),
        })
    
    def test_executor_creation(self, mock_backtest_fn, sample_data):
        """Test création de l'exécuteur."""
        from agents.backtest_executor import BacktestExecutor
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="ema_cross",
            data=sample_data,
        )
        
        assert executor.strategy_name == "ema_cross"
        assert len(executor.data) == 1000
        assert executor.history.total_experiments == 0
    
    def test_run_backtest(self, mock_backtest_fn, sample_data):
        """Test exécution d'un backtest."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="ema_cross",
            data=sample_data,
        )
        
        request = BacktestRequest(
            requested_by="test",
            hypothesis="Testing fast period 12",
            parameters={"fast": 12, "slow": 26},
        )
        
        result = executor.run(request)
        
        assert result.success is True
        assert result.sharpe_ratio > 0
        assert result.execution_time_ms >= 0  # Peut être très rapide
        assert executor.history.total_experiments == 1
    
    def test_run_batch(self, mock_backtest_fn, sample_data):
        """Test exécution batch."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="ema_cross",
            data=sample_data,
        )
        
        requests = [
            BacktestRequest(hypothesis=f"Test {i}", parameters={"fast": 10 + i})
            for i in range(3)
        ]
        
        results = executor.run_batch(requests)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert executor.history.total_experiments == 3
    
    def test_context_for_agent(self, mock_backtest_fn, sample_data):
        """Test génération de contexte pour agent."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="ema_cross",
            data=sample_data,
        )
        
        # Exécuter quelques backtests
        for i in range(3):
            request = BacktestRequest(
                hypothesis=f"Test {i}",
                parameters={"fast": 10 + i},
            )
            executor.run(request)
        
        context = executor.get_context_for_agent()
        
        assert "Experiment History" in context
        assert "3 total" in context
    
    def test_failed_backtest(self, sample_data):
        """Test gestion des erreurs."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest
        
        def failing_backtest(strategy, params, data):
            raise ValueError("Backtest failed")
        
        executor = BacktestExecutor(
            backtest_fn=failing_backtest,
            strategy_name="test",
            data=sample_data,
        )
        
        request = BacktestRequest(
            hypothesis="Will fail",
            parameters={},
        )
        
        result = executor.run(request)
        
        assert result.success is False
        assert "failed" in result.error_message.lower()


# ============================================================
# Tests AutonomousStrategist
# ============================================================

class TestIterationDecision:
    """Tests pour IterationDecision."""
    
    def test_decision_creation(self):
        """Test création d'une décision."""
        from agents.autonomous_strategist import IterationDecision
        
        decision = IterationDecision(
            action="continue",
            confidence=0.8,
            reasoning="Sharpe can be improved",
            next_hypothesis="Increase fast period",
            next_parameters={"fast": 12},
            insights=["Parameter x is sensitive"],
        )
        
        assert decision.action == "continue"
        assert decision.confidence == 0.8
        assert decision.next_parameters == {"fast": 12}


class TestOptimizationSession:
    """Tests pour OptimizationSession."""
    
    def test_session_creation(self):
        """Test création d'une session."""
        from agents.autonomous_strategist import OptimizationSession
        
        session = OptimizationSession(
            strategy_name="ema_cross",
            initial_params={"fast": 10, "slow": 21},
            max_iterations=10,
        )
        
        assert session.strategy_name == "ema_cross"
        assert session.max_iterations == 10
        assert session.current_iteration == 0
        assert session.best_result is None


class TestAutonomousStrategist:
    """Tests pour AutonomousStrategist."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock du client LLM."""
        from agents.llm_client import LLMResponse, LLMProvider
        
        client = Mock()
        
        # Stocker les réponses pour les retourner en séquence
        responses = [
            LLMResponse(
                content=json.dumps({
                    "action": "continue",
                    "confidence": 0.7,
                    "reasoning": "Testing higher fast period",
                    "next_hypothesis": "Increase fast to 12",
                    "next_parameters": {"fast": 12, "slow": 21},
                    "insights": [],
                }),
                model="test",
                provider=LLMProvider.OLLAMA,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
            LLMResponse(
                content=json.dumps({
                    "action": "accept",
                    "confidence": 0.9,
                    "reasoning": "Good improvement found",
                    "next_hypothesis": "",
                    "next_parameters": {},
                    "insights": ["Fast period 12 works well"],
                }),
                model="test",
                provider=LLMProvider.OLLAMA,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        ]
        
        response_iter = iter(responses)
        
        def chat_side_effect(*args, **kwargs):
            try:
                return next(response_iter)
            except StopIteration:
                # Retourner la dernière réponse par défaut
                return responses[-1]
        
        client.chat.side_effect = chat_side_effect
        return client
    
    @pytest.fixture
    def mock_backtest_fn(self):
        """Mock de fonction de backtest."""
        call_count = [0]
        
        def backtest(strategy, params, data):
            call_count[0] += 1
            # Simule amélioration à chaque itération
            return {
                "sharpe_ratio": 1.0 + call_count[0] * 0.2,
                "sortino_ratio": 1.5,
                "total_return": 0.20 + call_count[0] * 0.05,
                "max_drawdown": 0.15,
                "win_rate": 0.55,
                "profit_factor": 1.5,
                "total_trades": 100,
            }
        return backtest
    
    @pytest.fixture
    def sample_data(self):
        """Données de test."""
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "close": np.random.randn(n).cumsum() + 100,
        })
    
    def test_strategist_creation(self, mock_llm_client):
        """Test création du strategist autonome."""
        from agents.autonomous_strategist import AutonomousStrategist
        
        strategist = AutonomousStrategist(mock_llm_client, verbose=True)
        
        assert strategist.llm == mock_llm_client
        assert strategist.verbose is True
    
    def test_optimize_basic(
        self, mock_llm_client, mock_backtest_fn, sample_data
    ):
        """Test optimisation basique."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestExecutor
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="ema_cross",
            data=sample_data,
        )
        
        strategist = AutonomousStrategist(mock_llm_client, verbose=True)
        
        session = strategist.optimize(
            executor=executor,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=5,
        )
        
        assert session.final_status == "success"
        assert session.best_result is not None
        assert len(session.all_results) >= 2  # Baseline + au moins 1 itération
    
    def test_optimize_max_iterations(self, mock_backtest_fn, sample_data):
        """Test arrêt par max iterations."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestExecutor
        from agents.llm_client import LLMResponse, LLMProvider
        
        # LLM qui ne s'arrête jamais
        client = Mock()
        continue_response = LLMResponse(
            content=json.dumps({
                "action": "continue",
                "confidence": 0.5,
                "reasoning": "Still exploring",
                "next_hypothesis": "Try more",
                "next_parameters": {"fast": 10, "slow": 21},
            }),
            model="test",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
        )
        client.chat.return_value = continue_response
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="test",
            data=sample_data,
        )
        
        strategist = AutonomousStrategist(client, verbose=False)
        
        session = strategist.optimize(
            executor=executor,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=3,
        )
        
        assert session.final_status == "max_iterations"
        assert len(session.all_results) == 4  # Baseline + 3 itérations
    
    def test_validate_parameters(self, mock_llm_client):
        """Test validation des paramètres."""
        from agents.autonomous_strategist import AutonomousStrategist
        
        strategist = AutonomousStrategist(mock_llm_client)
        
        # Paramètres hors bornes
        params = {"fast": 100, "slow": -5}  # Hors bornes
        bounds = {"fast": (5, 20), "slow": (15, 50)}
        defaults = {"fast": 10, "slow": 21}
        
        validated = strategist._validate_parameters(params, bounds, defaults)
        
        assert validated["fast"] == 20  # Clampé au max
        assert validated["slow"] == 15  # Clampé au min
    
    def test_is_better_maximization(self, mock_llm_client):
        """Test comparaison de résultats (maximisation)."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestRequest, BacktestResult
        
        strategist = AutonomousStrategist(mock_llm_client)
        
        req = BacktestRequest(hypothesis="", parameters={})
        
        current = BacktestResult(
            request=req, success=True, sharpe_ratio=1.0
        )
        better = BacktestResult(
            request=req, success=True, sharpe_ratio=1.5
        )
        worse = BacktestResult(
            request=req, success=True, sharpe_ratio=0.8
        )
        
        assert strategist._is_better(better, current, "sharpe_ratio") is True
        assert strategist._is_better(worse, current, "sharpe_ratio") is False
    
    def test_is_better_minimization(self, mock_llm_client):
        """Test comparaison de résultats (minimisation)."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestRequest, BacktestResult
        
        strategist = AutonomousStrategist(mock_llm_client)
        
        req = BacktestRequest(hypothesis="", parameters={})
        
        current = BacktestResult(
            request=req, success=True, max_drawdown=0.20
        )
        better = BacktestResult(
            request=req, success=True, max_drawdown=0.10
        )
        worse = BacktestResult(
            request=req, success=True, max_drawdown=0.30
        )
        
        assert strategist._is_better(better, current, "max_drawdown") is True
        assert strategist._is_better(worse, current, "max_drawdown") is False
    
    def test_progress_callback(
        self, mock_llm_client, mock_backtest_fn, sample_data
    ):
        """Test callback de progression."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestExecutor
        
        progress_calls = []
        
        def on_progress(iteration, result):
            progress_calls.append((iteration, result.sharpe_ratio))
        
        executor = BacktestExecutor(
            backtest_fn=mock_backtest_fn,
            strategy_name="test",
            data=sample_data,
        )
        
        strategist = AutonomousStrategist(
            mock_llm_client, 
            on_progress=on_progress
        )
        
        session = strategist.optimize(
            executor=executor,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=5,
        )
        
        # Au moins le baseline (iteration 0)
        assert len(progress_calls) >= 1
        assert progress_calls[0][0] == 0  # Première itération


# ============================================================
# Tests d'intégration
# ============================================================

class TestIntegration:
    """Tests d'intégration du système complet."""
    
    def test_full_optimization_flow(self):
        """Test du flux complet d'optimisation."""
        from agents.autonomous_strategist import AutonomousStrategist
        from agents.backtest_executor import BacktestExecutor
        from agents.llm_client import LLMResponse, LLMProvider
        
        # Simuler un LLM qui explore puis accepte
        iteration = [0]
        
        def mock_chat(*args, **kwargs):
            iteration[0] += 1
            
            if iteration[0] < 3:
                return LLMResponse(
                    content=json.dumps({
                        "action": "continue",
                        "confidence": 0.6,
                        "reasoning": f"Exploring iteration {iteration[0]}",
                        "next_hypothesis": f"Try fast={10 + iteration[0]}",
                        "next_parameters": {"fast": 10 + iteration[0], "slow": 21},
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
                        "reasoning": "Found good configuration",
                    }),
                    model="test",
                    provider=LLMProvider.OLLAMA,
                    prompt_tokens=50,
                    completion_tokens=50,
                    total_tokens=100,
                )
        
        # Client mock
        client = Mock()
        client.chat = mock_chat
        
        # Fonction de backtest qui améliore avec fast plus élevé
        def backtest(strategy, params, data):
            fast = params.get("fast", 10)
            return {
                "sharpe_ratio": 0.5 + fast * 0.1,
                "total_return": 0.10 + fast * 0.01,
                "max_drawdown": 0.20 - fast * 0.005,
                "win_rate": 0.50,
                "profit_factor": 1.2,
                "total_trades": 50,
                "sortino_ratio": 1.0,
            }
        
        # Données
        np.random.seed(42)
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})
        
        # Exécution
        executor = BacktestExecutor(
            backtest_fn=backtest,
            strategy_name="test_strategy",
            data=data,
        )
        
        strategist = AutonomousStrategist(client, verbose=False)
        
        session = strategist.optimize(
            executor=executor,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 25), "slow": (15, 50)},
            max_iterations=10,
        )
        
        # Vérifications
        assert session.final_status == "success"
        assert session.best_result is not None
        assert session.best_result.sharpe_ratio > 1.0  # Amélioré
        assert len(session.all_results) == 3  # Baseline + 2 itérations
        assert len(session.decisions) == 3  # 3 décisions (continue, continue, accept)
    
    def test_factory_function(self):
        """Test de la factory create_autonomous_optimizer."""
        from agents import create_autonomous_optimizer
        from agents.llm_client import LLMConfig, LLMProvider
        
        # Note: ce test ne peut pas vraiment appeler Ollama,
        # mais on vérifie que la factory fonctionne
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        
        def mock_backtest(strategy, params, data):
            return {"sharpe_ratio": 1.0}
        
        data = pd.DataFrame({"close": [100, 101, 102]})
        
        # La factory devrait lever une erreur car Ollama n'est pas disponible
        # dans les tests, mais on peut au moins vérifier que les imports
        # fonctionnent
        try:
            strategist, executor = create_autonomous_optimizer(
                llm_config=config,
                backtest_fn=mock_backtest,
                strategy_name="test",
                data=data,
            )
            # Si Ollama est disponible, on vérifie les types
            from agents import AutonomousStrategist, BacktestExecutor
            assert isinstance(strategist, AutonomousStrategist)
            assert isinstance(executor, BacktestExecutor)
        except Exception:
            # Normal si Ollama n'est pas disponible
            pass
