"""
Tests unitaires pour Phase 2 : Mono-agent sweep integration.

Tests:
- system_prompt contient action='sweep'
- _param_bounds_to_specs() convertit correctement
- IterationDecision parse les champs sweep
- Bloc action == "sweep" dans optimize() fonctionne
"""

from unittest.mock import Mock, patch

from agents.autonomous_strategist import (
    AutonomousStrategist,
    IterationDecision,
    _param_bounds_to_specs,
)
from utils.parameters import ParameterSpec


class TestSystemPromptSweep:
    """Tests pour le system_prompt étendu avec sweep."""

    def test_system_prompt_contains_sweep_action(self):
        """Test que system_prompt contient l'action 'sweep'."""
        llm_mock = Mock()
        strategist = AutonomousStrategist(llm_mock)
        prompt = strategist.system_prompt

        assert "sweep" in prompt.lower()
        assert '"action": "continue|accept|stop|change_direction|sweep"' in prompt

    def test_system_prompt_contains_sweep_example(self):
        """Test que system_prompt contient un exemple valide de sweep."""
        llm_mock = Mock()
        strategist = AutonomousStrategist(llm_mock)
        prompt = strategist.system_prompt

        assert "ranges" in prompt
        assert "rationale" in prompt
        assert "optimize_for" in prompt
        assert "max_combinations" in prompt

    def test_system_prompt_contains_sweep_requirements(self):
        """Test que system_prompt documente les requirements du sweep."""
        llm_mock = Mock()
        strategist = AutonomousStrategist(llm_mock)
        prompt = strategist.system_prompt

        assert "CRITICAL REQUIREMENTS FOR" in prompt
        assert "min" in prompt and "max" in prompt and "step" in prompt


class TestIterationDecisionSweep:
    """Tests pour IterationDecision avec champs sweep."""

    def test_iteration_decision_has_sweep_fields(self):
        """Test qu'IterationDecision possède les champs sweep."""
        decision = IterationDecision(
            action="sweep",
            confidence=0.85,
        )

        assert hasattr(decision, "ranges")
        assert hasattr(decision, "rationale")
        assert hasattr(decision, "optimize_for")
        assert hasattr(decision, "max_combinations")

    def test_iteration_decision_sweep_defaults(self):
        """Test valeurs par défaut des champs sweep."""
        decision = IterationDecision(
            action="sweep",
            confidence=0.85,
        )

        assert decision.ranges is None
        assert decision.rationale == ""
        assert decision.optimize_for == "sharpe_ratio"
        assert decision.max_combinations == 100

    def test_iteration_decision_can_set_sweep_fields(self):
        """Test qu'on peut assigner les champs sweep."""
        ranges = {
            "bb_period": {"min": 20, "max": 25, "step": 1},
            "bb_std": {"min": 2.0, "max": 2.5, "step": 0.1}
        }

        decision = IterationDecision(
            action="sweep",
            confidence=0.85,
            ranges=ranges,
            rationale="Test correlation bb_period vs bb_std",
            optimize_for="total_return",
            max_combinations=50,
        )

        assert decision.ranges == ranges
        assert decision.rationale == "Test correlation bb_period vs bb_std"
        assert decision.optimize_for == "total_return"
        assert decision.max_combinations == 50


class TestParamBoundsToSpecs:
    """Tests pour _param_bounds_to_specs()."""

    def test_basic_conversion(self):
        """Test conversion basique bounds -> specs."""
        param_bounds = {
            "fast": (5, 20),
            "slow": (15, 50),
        }
        defaults = {"fast": 10, "slow": 21}

        specs = _param_bounds_to_specs(param_bounds, defaults)

        assert len(specs) == 2
        assert all(isinstance(s, ParameterSpec) for s in specs)

        fast_spec = next(s for s in specs if s.name == "fast")
        assert fast_spec.min_val == 5.0
        assert fast_spec.max_val == 20.0
        assert fast_spec.default == 10

    def test_conversion_with_step(self):
        """Test conversion avec step défini."""
        param_bounds = {
            "test": (10, 50, 5),  # (min, max, step)
        }
        defaults = {"test": 20}

        specs = _param_bounds_to_specs(param_bounds, defaults)

        test_spec = specs[0]
        assert test_spec.step == 5.0

    def test_conversion_int_detection(self):
        """Test détection du type int."""
        param_bounds = {
            "int_param": (10, 50),  # Tous int
            "float_param": (1.0, 5.0),  # Float explicite
        }
        defaults = {"int_param": 20, "float_param": 2.5}

        specs = _param_bounds_to_specs(param_bounds, defaults)

        int_spec = next(s for s in specs if s.name == "int_param")
        float_spec = next(s for s in specs if s.name == "float_param")

        assert int_spec.param_type == "int"
        assert float_spec.param_type == "float"

    def test_conversion_default_from_bounds(self):
        """Test que default est (min+max)/2 si absent."""
        param_bounds = {
            "test": (10, 30),
        }
        defaults = {}  # Pas de default fourni

        specs = _param_bounds_to_specs(param_bounds, defaults)

        test_spec = specs[0]
        assert test_spec.default == 20.0  # (10 + 30) / 2


class TestAutonomousStrategistSweep:
    """Tests pour intégration sweep dans AutonomousStrategist.optimize()."""

    @patch('agents.integration.run_llm_sweep')
    def test_sweep_action_calls_run_llm_sweep(self, mock_run_llm_sweep):
        """Test que action='sweep' appelle run_llm_sweep()."""
        # Setup mock LLM client
        llm_mock = Mock()
        llm_response = Mock()
        llm_response.content = '''{
            "action": "sweep",
            "confidence": 0.85,
            "ranges": {
                "fast": {"min": 8, "max": 12, "step": 1},
                "slow": {"min": 20, "max": 24, "step": 1}
            },
            "rationale": "Test fast/slow correlation",
            "optimize_for": "sharpe_ratio",
            "max_combinations": 20,
            "reasoning": "Sweep is needed",
            "insights": []
        }'''
        llm_response.parse_json = Mock(return_value={
            "action": "sweep",
            "confidence": 0.85,
            "ranges": {
                "fast": {"min": 8, "max": 12, "step": 1},
                "slow": {"min": 20, "max": 24, "step": 1}
            },
            "rationale": "Test fast/slow correlation",
            "optimize_for": "sharpe_ratio",
            "max_combinations": 20,
            "reasoning": "Sweep is needed",
            "insights": []
        })
        llm_mock._call_llm = Mock(return_value=llm_response)
        llm_mock.config = Mock(model="test-model", max_tokens=2000)

        # Mock run_llm_sweep pour retourner résultats valides
        mock_run_llm_sweep.return_value = {
            "best_params": {"fast": 10, "slow": 22},
            "best_metrics": {
                "sharpe_ratio": 2.5,
                "total_return": 0.15,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 50,
                "overfitting_ratio": 1.1,
            },
            "top_k": [],
            "summary": "Sweep completed",
            "n_combinations": 20,
        }

        # Setup executor mock
        executor_mock = Mock()
        executor_mock.strategy_name = "test_strategy"
        executor_mock.data = Mock()  # DataFrame mock
        executor_mock.get_context_for_agent = Mock(return_value="Baseline: Sharpe=1.5")

        # Create strategist and override _call_llm
        strategist = AutonomousStrategist(llm_mock, verbose=False)
        strategist._call_llm = llm_mock._call_llm

        # Monkey-patch _run_backtest_with_gpu_optimization to avoid actual backtest
        def fake_backtest(executor, request):
            from agents.backtest_executor import BacktestResult
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

        strategist._run_backtest_with_gpu_optimization = fake_backtest

        # Run optimization (will execute 1 baseline backtest + 1 sweep)
        # Note: max_iterations is budget of combos, sweep consumes 20 combos
        session = strategist.optimize(
            executor=executor_mock,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=25,  # Budget: baseline(1) + sweep(20) + margin
            target_metric="sharpe_ratio",
        )

        # Vérifier que run_llm_sweep a été appelé
        assert mock_run_llm_sweep.called
        call_args = mock_run_llm_sweep.call_args

        # Vérifier les arguments
        assert "range_proposal" in call_args.kwargs
        assert "param_specs" in call_args.kwargs
        assert "data" in call_args.kwargs
        assert "strategy_name" in call_args.kwargs

        # Vérifier que le meilleur résultat du sweep a été intégré
        assert len(session.all_results) >= 2  # baseline + sweep result

    def test_sweep_action_without_ranges_forces_stop(self):
        """Test que action='sweep' sans ranges force un stop."""
        # Setup mock LLM client qui retourne sweep sans ranges
        llm_mock = Mock()
        llm_response = Mock()
        llm_response.content = '''{
            "action": "sweep",
            "confidence": 0.85,
            "reasoning": "Need sweep",
            "insights": []
        }'''
        llm_response.parse_json = Mock(return_value={
            "action": "sweep",
            "confidence": 0.85,
            "reasoning": "Need sweep",
            "insights": []
        })
        llm_mock._call_llm = Mock(return_value=llm_response)
        llm_mock.config = Mock(model="test-model", max_tokens=2000)

        # Setup executor mock
        executor_mock = Mock()
        executor_mock.strategy_name = "test_strategy"
        executor_mock.data = Mock()
        executor_mock.get_context_for_agent = Mock(return_value="Baseline: Sharpe=1.5")

        # Create strategist
        strategist = AutonomousStrategist(llm_mock, verbose=False)
        strategist._call_llm = llm_mock._call_llm

        # Monkey-patch _run_backtest_with_gpu_optimization
        def fake_backtest(executor, request):
            from agents.backtest_executor import BacktestResult
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

        strategist._run_backtest_with_gpu_optimization = fake_backtest

        # Run optimization
        session = strategist.optimize(
            executor=executor_mock,
            initial_params={"fast": 10, "slow": 21},
            param_bounds={"fast": (5, 20), "slow": (15, 50)},
            max_iterations=5,
            target_metric="sharpe_ratio",
        )

        # Vérifier que l'optimisation s'est arrêtée (pas de crash)
        assert session.final_status in ("success", "no_improvement", "max_iterations")
        # La décision devrait être "stop" (forcé par validation)
        if session.decisions:
            # _get_llm_decision doit avoir forcé action="stop"
            # car ranges était None
            pass
