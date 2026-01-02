"""
Tests unitaires pour Phase 3 : Multi-agents sweep integration.

Tests:
- Templates Jinja2 contiennent documentation sweep
- Orchestrator._handle_propose() détecte sweep request
- Orchestrator._handle_sweep_proposal() exécute sweep
- AgentContext conserve sweep_results
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestTemplatesSweepDocumentation:
    """Tests pour les templates Jinja2 étendus avec sweep."""

    def test_strategist_template_contains_sweep_option(self):
        """Test que strategist.jinja2 contient la documentation sweep."""
        template_path = Path("d:/backtest_core/templates/strategist.jinja2")
        assert template_path.exists()

        content = template_path.read_text(encoding="utf-8")
        assert "GRID SEARCH OPTION" in content
        assert "sweep" in content.lower()
        assert "ranges" in content
        assert "rationale" in content

    def test_strategist_template_has_sweep_format(self):
        """Test que strategist.jinja2 documente le format JSON sweep."""
        template_path = Path("d:/backtest_core/templates/strategist.jinja2")
        content = template_path.read_text(encoding="utf-8")

        assert '"sweep"' in content
        assert "ALTERNATIVE: Grid Search" in content or "Grid Search (Sweep) Format" in content
        assert "optimize_for" in content
        assert "max_combinations" in content

    def test_analyst_template_contains_sweep_consideration(self):
        """Test que analyst.jinja2 suggère sweep dans recommendations."""
        template_path = Path("d:/backtest_core/templates/analyst.jinja2")
        assert template_path.exists()

        content = template_path.read_text(encoding="utf-8")
        assert "GRID SEARCH CONSIDERATION" in content or "grid search" in content.lower()
        assert "parameter correlations" in content.lower() or "correlations" in content.lower()


class TestOrchestratorSweepDetection:
    """Tests pour détection sweep dans Orchestrator."""

    @patch('agents.integration.run_llm_sweep')
    def test_handle_propose_detects_sweep_request(self, mock_run_llm_sweep):
        """Test que _handle_propose() détecte un sweep request."""
        from agents.orchestrator import Orchestrator, OrchestratorConfig
        from agents.base_agent import AgentResult, AgentRole
        from utils.parameters import ParameterSpec
        import pandas as pd

        # Setup config minimal
        config = OrchestratorConfig(
            strategy_name="test_strategy",
            data=pd.DataFrame({"close": [100, 101, 102]}),  # Données minimales
        )

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
            "n_combinations": 20,
        }

        # Create orchestrator
        orchestrator = Orchestrator(config)

        # Mock strategist.execute() pour retourner un sweep request
        def fake_strategist_execute(context):
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
                        "rationale": "Testing fast/slow correlation",
                        "optimize_for": "sharpe_ratio",
                        "max_combinations": 20,
                    },
                    "analysis_summary": "Need grid search",
                    "optimization_strategy": "Grid search approach",
                },
                execution_time_ms=100,
                tokens_used=500,
                llm_calls=1,
            )

        orchestrator.strategist.execute = fake_strategist_execute

        # Ajouter param_specs au contexte
        orchestrator.context.param_specs = [
            ParameterSpec("fast", 5, 20, 10, 1, "int"),
            ParameterSpec("slow", 15, 50, 21, 1, "int"),
        ]

        # Appeler _handle_propose()
        orchestrator._handle_propose()

        # Vérifier que run_llm_sweep a été appelé
        assert mock_run_llm_sweep.called

        # Vérifier que context contient sweep_results
        assert hasattr(orchestrator.context, 'sweep_results')
        assert hasattr(orchestrator.context, 'sweep_summary')

        # Vérifier qu'une proposition artificielle a été créée
        assert len(orchestrator.context.strategist_proposals) == 1
        assert orchestrator.context.strategist_proposals[0]["parameters"] == {"fast": 10, "slow": 22}

    def test_handle_propose_with_normal_proposals(self):
        """Test que _handle_propose() gère normalement les proposals (non-sweep)."""
        from agents.orchestrator import Orchestrator, OrchestratorConfig
        from agents.base_agent import AgentResult, AgentRole
        import pandas as pd

        config = OrchestratorConfig(
            strategy_name="test_strategy",
            data=pd.DataFrame({"close": [100, 101, 102]}),
        )

        orchestrator = Orchestrator(config)

        # Mock strategist.execute() pour retourner des proposals normales
        def fake_strategist_execute(context):
            return AgentResult(
                success=True,
                agent_role=AgentRole.STRATEGIST,
                content="Proposals",
                data={
                    "proposals": [
                        {
                            "id": 1,
                            "name": "Test proposal",
                            "parameters": {"fast": 10, "slow": 21},
                            "rationale": "Testing",
                        }
                    ],
                    "analysis_summary": "Analysis",
                },
                execution_time_ms=100,
                tokens_used=500,
                llm_calls=1,
            )

        orchestrator.strategist.execute = fake_strategist_execute

        # Appeler _handle_propose()
        orchestrator._handle_propose()

        # Vérifier que proposals normales sont traitées
        assert len(orchestrator.context.strategist_proposals) == 1
        assert not hasattr(orchestrator.context, 'sweep_results') or orchestrator.context.sweep_results is None


class TestHandleSweepProposal:
    """Tests pour Orchestrator._handle_sweep_proposal()."""

    @patch('agents.integration.run_llm_sweep')
    def test_handle_sweep_proposal_executes_sweep(self, mock_run_llm_sweep):
        """Test que _handle_sweep_proposal() exécute le sweep correctement."""
        from agents.orchestrator import Orchestrator, OrchestratorConfig
        from utils.parameters import ParameterSpec
        import pandas as pd

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
            "summary": "Grid search completed: 30 combinations",
            "n_combinations": 30,
        }

        config = OrchestratorConfig(
            strategy_name="bollinger_atr",
            data=pd.DataFrame({"close": [100] * 100}),
        )

        orchestrator = Orchestrator(config)

        # Ajouter param_specs
        orchestrator.context.param_specs = [
            ParameterSpec("bb_period", 15, 30, 20, 1, "int"),
            ParameterSpec("bb_std", 1.5, 3.0, 2.0, 0.1, "float"),
        ]

        sweep_request = {
            "ranges": {
                "bb_period": {"min": 20, "max": 25, "step": 1},
                "bb_std": {"min": 2.0, "max": 2.5, "step": 0.1}
            },
            "rationale": "Testing bb_period/bb_std correlation",
            "optimize_for": "sharpe_ratio",
            "max_combinations": 30,
        }

        # Appeler _handle_sweep_proposal()
        orchestrator._handle_sweep_proposal(sweep_request)

        # Vérifier run_llm_sweep appelé
        assert mock_run_llm_sweep.called
        call_kwargs = mock_run_llm_sweep.call_args.kwargs

        assert "range_proposal" in call_kwargs
        assert "param_specs" in call_kwargs
        assert "data" in call_kwargs

        # Vérifier context
        assert orchestrator.context.sweep_results == mock_run_llm_sweep.return_value
        assert orchestrator.context.sweep_summary == "Grid search completed: 30 combinations"

        # Vérifier proposition artificielle
        assert len(orchestrator.context.strategist_proposals) == 1
        proposal = orchestrator.context.strategist_proposals[0]
        assert proposal["parameters"] == {"bb_period": 22, "bb_std": 2.2}
        assert "Sweep Best Config" in proposal["name"]

    @patch('agents.integration.run_llm_sweep')
    def test_handle_sweep_proposal_handles_errors(self, mock_run_llm_sweep):
        """Test que _handle_sweep_proposal() gère les erreurs proprement."""
        from agents.orchestrator import Orchestrator, OrchestratorConfig
        from utils.parameters import ParameterSpec
        import pandas as pd

        # Mock run_llm_sweep pour lever une erreur
        mock_run_llm_sweep.side_effect = ValueError("Invalid ranges")

        config = OrchestratorConfig(
            strategy_name="test_strategy",
            data=pd.DataFrame({"close": [100] * 100}),
        )

        orchestrator = Orchestrator(config)
        orchestrator.context.param_specs = [
            ParameterSpec("test", 1, 10, 5, 1, "int"),
        ]

        sweep_request = {
            "ranges": {"test": {"min": 1, "max": 10, "step": 1}},
            "rationale": "Test error handling",
        }

        # Appeler _handle_sweep_proposal() - ne devrait pas crash
        orchestrator._handle_sweep_proposal(sweep_request)

        # Vérifier que erreur est capturée
        assert len(orchestrator._errors) > 0
        assert any("Sweep failed" in err for err in orchestrator._errors)

        # Vérifier transition vers VALIDATE
        # (difficile à tester car state_machine est modifié)
