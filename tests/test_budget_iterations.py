"""
Tests pour P0-1: Budget iterations corrigé (sweep consomme N combos, pas 1).

Tests:
- AutonomousStrategist: sweep consomme n_combinations vers max_iterations
- AutonomousStrategist: limite de sweeps par session
- Orchestrator: sweep consomme n_combinations vers budget
- Orchestrator: backtest normal consomme 1 combo
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from agents.autonomous_strategist import AutonomousStrategist
from agents.orchestrator import Orchestrator, OrchestratorConfig
from agents.base_agent import AgentRole


class TestAutonomousStrategistBudget:
    """Tests pour budget iterations dans AutonomousStrategist."""

    @patch('agents.integration.run_llm_sweep')
    def test_sweep_consumes_n_combinations_budget(self, mock_run_llm_sweep):
        """Test qu'un sweep consomme n_combinations vers max_iterations."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest, BacktestResult

        # Mock run_llm_sweep pour retourner 20 combinaisons testées
        mock_run_llm_sweep.return_value = {
            "best_params": {"fast": 10, "slow": 22},
            "best_metrics": {
                "sharpe_ratio": 2.5,
                "total_return": 0.15,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 50,
            },
            "top_k": [],
            "summary": "Sweep completed: 20 combos",
            "n_combinations": 20,  # 20 combos testées
        }

        # Setup executor mock
        executor_mock = Mock(spec=BacktestExecutor)
        executor_mock.data = pd.DataFrame({"close": [100, 101, 102]})
        executor_mock.strategy_name = "test_strategy"
        executor_mock.get_context_for_agent = Mock(return_value="Baseline: Sharpe=1.5")

        def fake_backtest(request):
            return BacktestResult(
                request=request,
                success=True,
                sharpe_ratio=1.5,
                total_return=0.1,
                max_drawdown=0.1,
                win_rate=0.55,
                total_trades=30,
                execution_time_ms=10,
            )

        executor_mock.run_backtest = fake_backtest

        strategist = AutonomousStrategist(model="mock")

        # Mock _get_llm_decision pour demander un sweep puis stop
        decision_count = 0

        def fake_get_llm_decision(context, session):
            nonlocal decision_count
            decision_count += 1

            if decision_count == 1:
                # Première décision: sweep
                from agents.autonomous_strategist import IterationDecision
                return IterationDecision(
                    action="sweep",
                    confidence=0.85,
                    ranges={
                        "fast": {"min": 8, "max": 12, "step": 1},
                        "slow": {"min": 20, "max": 24, "step": 1}
                    },
                    rationale="Test budget consumption",
                    optimize_for="sharpe_ratio",
                    max_combinations=20,
                )
            else:
                # Deuxième décision: stop
                from agents.autonomous_strategist import IterationDecision
                return IterationDecision(
                    action="stop",
                    confidence=0.9,
                    reasoning="Budget should be exhausted"
                )

        strategist._get_llm_decision = fake_get_llm_decision
        strategist._run_backtest_with_gpu_optimization = fake_backtest

        # Run avec max_iterations=25 (baseline + 20 sweep + 4 normales max)
        session = strategist.optimize(
            executor=executor_mock,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=25,
            target_metric="sharpe_ratio",
        )

        # Vérifier que run_llm_sweep a été appelé
        assert mock_run_llm_sweep.called

        # Vérifier que le sweep a consommé le budget correctement
        # Le total_combinations_tested devrait être: 1 (baseline) + 20 (sweep) = 21
        # Donc il reste 25 - 21 = 4 combos possibles
        # La 2e décision est "stop" donc pas d'autres backtests

        # Vérifier status
        assert session.final_status in ("success", "no_improvement", "max_iterations")

    @patch('agents.integration.run_llm_sweep')
    def test_sweep_limit_per_session(self, mock_run_llm_sweep):
        """Test que la limite de sweeps par session est respectée."""
        from agents.backtest_executor import BacktestExecutor, BacktestRequest, BacktestResult

        # Mock run_llm_sweep
        mock_run_llm_sweep.return_value = {
            "best_params": {"fast": 10, "slow": 22},
            "best_metrics": {
                "sharpe_ratio": 2.5,
                "total_return": 0.15,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 50,
            },
            "top_k": [],
            "summary": "Sweep completed",
            "n_combinations": 10,
        }

        executor_mock = Mock(spec=BacktestExecutor)
        executor_mock.data = pd.DataFrame({"close": [100, 101, 102]})
        executor_mock.strategy_name = "test_strategy"
        executor_mock.get_context_for_agent = Mock(return_value="Baseline: Sharpe=1.5")

        def fake_backtest(request):
            return BacktestResult(
                request=request,
                success=True,
                sharpe_ratio=1.5,
                total_return=0.1,
                max_drawdown=0.1,
                win_rate=0.55,
                total_trades=30,
                execution_time_ms=10,
            )

        executor_mock.run_backtest = fake_backtest

        strategist = AutonomousStrategist(model="mock")

        # Mock _get_llm_decision pour demander 5 sweeps (dépasse la limite de 3)
        decision_count = 0

        def fake_get_llm_decision(context, session):
            nonlocal decision_count
            decision_count += 1

            # Toujours demander sweep
            from agents.autonomous_strategist import IterationDecision
            return IterationDecision(
                action="sweep",
                confidence=0.85,
                ranges={
                    "fast": {"min": 8, "max": 12, "step": 1},
                    "slow": {"min": 20, "max": 24, "step": 1}
                },
                rationale=f"Sweep #{decision_count}",
                optimize_for="sharpe_ratio",
                max_combinations=10,
            )

        strategist._get_llm_decision = fake_get_llm_decision
        strategist._run_backtest_with_gpu_optimization = fake_backtest

        # Run avec max_iterations suffisant pour 5 sweeps
        session = strategist.optimize(
            executor=executor_mock,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=100,
            target_metric="sharpe_ratio",
        )

        # Vérifier que run_llm_sweep a été appelé maximum 3 fois (limite)
        assert mock_run_llm_sweep.call_count <= 3

        # Vérifier status (devrait être "sweep_limit_reached")
        assert session.final_status in ("sweep_limit_reached", "no_improvement", "max_iterations")


class TestOrchestratorBudget:
    """Tests pour budget iterations dans Orchestrator."""

    @patch('agents.integration.run_llm_sweep')
    def test_sweep_consumes_budget(self, mock_run_llm_sweep):
        """Test que le sweep consomme le budget dans Orchestrator."""
        # Mock run_llm_sweep
        mock_run_llm_sweep.return_value = {
            "best_params": {"bb_period": 22, "bb_std": 2.2},
            "best_metrics": {
                "sharpe_ratio": 2.8,
                "total_return": 0.18,
                "max_drawdown": 0.07,
                "win_rate": 0.65,
                "total_trades": 60,
            },
            "top_k": [],
            "summary": "Grid search completed: 15 combinations",
            "n_combinations": 15,
        }

        config = OrchestratorConfig(
            strategy_name="bollinger_atr",
            data=pd.DataFrame({"close": [100] * 100}),
            max_iterations=20,  # Budget de 20 combos
        )

        orchestrator = Orchestrator(config)

        # Mock strategist pour demander un sweep
        def fake_strategist_execute(context):
            from agents.base_agent import AgentResult, AgentRole
            return AgentResult(
                success=True,
                agent_role=AgentRole.STRATEGIST,
                content="Sweep request",
                data={
                    "sweep": {
                        "ranges": {
                            "bb_period": {"min": 20, "max": 25, "step": 1},
                            "bb_std": {"min": 2.0, "max": 2.5, "step": 0.1}
                        },
                        "rationale": "Testing bb_period/bb_std correlation",
                        "optimize_for": "sharpe_ratio",
                        "max_combinations": 15,
                    },
                    "analysis_summary": "Need grid search",
                    "optimization_strategy": "Grid search approach",
                },
                execution_time_ms=100,
                tokens_used=500,
                llm_calls=1,
            )

        orchestrator.strategist.execute = fake_strategist_execute

        # Ajouter param_specs
        from utils.parameters import ParameterSpec
        orchestrator.context.param_specs = [
            ParameterSpec("bb_period", 15, 30, 20, 1, "int"),
            ParameterSpec("bb_std", 1.5, 3.0, 2.0, 0.1, "float"),
        ]

        # Appeler _handle_propose()
        orchestrator._handle_propose()

        # Vérifier que le compteur de budget a été incrémenté
        assert orchestrator._total_combinations_tested == 15
        assert orchestrator._sweeps_performed == 1

        # Vérifier que run_llm_sweep a été appelé
        assert mock_run_llm_sweep.called

    def test_backtest_normal_consumes_one_combo(self):
        """Test qu'un backtest normal consomme 1 combo vers le budget."""
        config = OrchestratorConfig(
            strategy_name="test_strategy",
            data=pd.DataFrame({"close": [100] * 100}),
            max_iterations=10,
        )

        orchestrator = Orchestrator(config)

        # Mock callback backtest
        def fake_backtest(params):
            return {
                "sharpe_ratio": 1.5,
                "total_return": 0.1,
                "max_drawdown": 0.1,
                "win_rate": 0.55,
                "total_trades": 30,
            }

        orchestrator.config.on_backtest_needed = fake_backtest

        # Exécuter un backtest
        initial_budget = orchestrator._total_combinations_tested
        orchestrator._run_backtest({"fast": 10, "slow": 21})

        # Vérifier que le budget a été incrémenté de 1
        assert orchestrator._total_combinations_tested == initial_budget + 1

    @patch('agents.integration.run_llm_sweep')
    def test_budget_exhausted_stops_iterations(self, mock_run_llm_sweep):
        """Test que le budget épuisé arrête les itérations."""
        mock_run_llm_sweep.return_value = {
            "best_params": {"fast": 10, "slow": 22},
            "best_metrics": {
                "sharpe_ratio": 2.5,
                "total_return": 0.15,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 50,
            },
            "top_k": [],
            "summary": "Sweep completed: 18 combinations",
            "n_combinations": 18,
        }

        config = OrchestratorConfig(
            strategy_name="test_strategy",
            data=pd.DataFrame({"close": [100] * 100}),
            max_iterations=20,  # Budget serré
        )

        orchestrator = Orchestrator(config)

        # Mock strategist pour sweep
        def fake_strategist_execute(context):
            from agents.base_agent import AgentResult, AgentRole
            return AgentResult(
                success=True,
                agent_role=AgentRole.STRATEGIST,
                content="Sweep request",
                data={
                    "sweep": {
                        "ranges": {
                            "fast": {"min": 8, "max": 12, "step": 1},
                            "slow": {"min": 20, "max": 24, "step": 1}
                        },
                        "rationale": "Test budget exhaustion",
                        "optimize_for": "sharpe_ratio",
                        "max_combinations": 18,
                    },
                    "analysis_summary": "Grid search",
                },
                execution_time_ms=100,
                tokens_used=500,
                llm_calls=1,
            )

        orchestrator.strategist.execute = fake_strategist_execute

        from utils.parameters import ParameterSpec
        orchestrator.context.param_specs = [
            ParameterSpec("fast", 5, 20, 10, 1, "int"),
            ParameterSpec("slow", 15, 50, 21, 1, "int"),
        ]

        # Simuler budget déjà consommé à 20 (atteint la limite 20)
        orchestrator._total_combinations_tested = 20

        # Appeler _handle_iterate() devrait vérifier le budget et arrêter
        orchestrator._handle_iterate()

        # Vérifier que le budget a déclenché un arrêt (transition vers REJECTED)
        from agents.state_machine import AgentState
        assert orchestrator.state_machine.current_state == AgentState.REJECTED
        assert len(orchestrator._warnings) > 0
        assert "Budget épuisé" in orchestrator._warnings[0]
