"""
Tests pour le système de templates Jinja2.

Valide:
- Chargement et rendu des templates
- Variables injectées correctement
- Filtres personnalisés fonctionnels
- Intégration avec les agents
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from utils.template import (
    render_prompt,
    render_prompt_from_string,
    list_available_templates,
    get_jinja_env,
)
from agents.base_agent import AgentContext, MetricsSnapshot


class TestTemplateEngine:
    """Tests du moteur de templates de base."""
    
    def test_jinja_env_initialization(self):
        """Le Jinja2 environment doit être initialisé correctement."""
        env = get_jinja_env()
        assert env is not None
        assert "format_percent" in env.filters
        assert "format_float" in env.filters
    
    def test_render_from_string_simple(self):
        """Rendu d'un template simple."""
        template = "Hello {{ name }}!"
        result = render_prompt_from_string(template, {"name": "Agent"})
        assert result == "Hello Agent!"
    
    def test_render_from_string_with_loop(self):
        """Rendu avec boucle."""
        template = """
        {% for item in items %}
        - {{ item }}
        {% endfor %}
        """
        result = render_prompt_from_string(template, {"items": ["a", "b", "c"]})
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result
    
    def test_custom_filter_format_percent(self):
        """Filtre format_percent fonctionne."""
        template = "Value: {{ value|format_percent }}"
        result = render_prompt_from_string(template, {"value": 0.1523})
        assert "15.23%" in result
    
    def test_custom_filter_format_float(self):
        """Filtre format_float fonctionne."""
        template = "Value: {{ value|format_float(3) }}"
        result = render_prompt_from_string(template, {"value": 1.23456})
        assert "1.235" in result
    
    def test_list_available_templates(self):
        """Liste des templates disponibles."""
        templates = list_available_templates()
        assert isinstance(templates, list)
        assert "analyst.jinja2" in templates
        assert "strategist.jinja2" in templates
        assert "critic.jinja2" in templates
        assert "validator.jinja2" in templates
    
    def test_template_not_found_raises(self):
        """Template inexistant lève TemplateNotFound."""
        from jinja2 import TemplateNotFound
        with pytest.raises(TemplateNotFound):
            render_prompt("nonexistent.jinja2", {})


class TestAnalystTemplate:
    """Tests du template analyst.jinja2."""
    
    @pytest.fixture
    def sample_context(self):
        """Contexte de test pour l'analyst."""
        metrics = MetricsSnapshot(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=-0.15,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=150
        )
        
        return {
            "strategy_name": "ema_cross",
            "strategy_description": "EMA crossover strategy",
            "data_symbol": "BTCUSDT",
            "data_timeframe": "1h",
            "data_date_range": "2023-01-01 to 2023-12-31",
            "data_rows": 8760,
            "iteration": 5,
            "current_params": {"fast_period": 12, "slow_period": 26},
            "current_metrics": metrics,
            "train_metrics": None,
            "test_metrics": None,
            "overfitting_ratio": 0.0,
            "iteration_history": [],
            "optimization_target": "sharpe_ratio",
            "min_sharpe": 1.0,
            "max_drawdown_limit": 0.20,
            "min_trades": 100,
            "max_overfitting_ratio": 1.5,
        }
    
    def test_analyst_template_renders(self, sample_context):
        """Template analyst se rend sans erreur."""
        result = render_prompt("analyst.jinja2", sample_context)
        assert result is not None
        assert len(result) > 0
    
    def test_analyst_contains_strategy_info(self, sample_context):
        """Le prompt contient les infos de stratégie."""
        result = render_prompt("analyst.jinja2", sample_context)
        assert "ema_cross" in result
        assert "BTCUSDT" in result
        assert "1h" in result
    
    def test_analyst_contains_params(self, sample_context):
        """Le prompt contient les paramètres."""
        result = render_prompt("analyst.jinja2", sample_context)
        assert "fast_period" in result
        assert "slow_period" in result
        assert "12" in result
        assert "26" in result
    
    def test_analyst_contains_objectives(self, sample_context):
        """Le prompt contient les objectifs."""
        result = render_prompt("analyst.jinja2", sample_context)
        assert "Optimization Objectives" in result
        assert "sharpe_ratio" in result
        assert "1.0" in result  # min_sharpe
    
    def test_analyst_with_walk_forward(self, sample_context):
        """Le prompt inclut walk-forward si disponible."""
        train_metrics = MetricsSnapshot(
            sharpe_ratio=1.8, total_return=0.30, max_drawdown=-0.10,
            win_rate=0.60, profit_factor=2.0, total_trades=100
        )
        test_metrics = MetricsSnapshot(
            sharpe_ratio=1.2, total_return=0.18, max_drawdown=-0.12,
            win_rate=0.52, profit_factor=1.5, total_trades=50
        )
        
        sample_context["train_metrics"] = train_metrics
        sample_context["test_metrics"] = test_metrics
        sample_context["overfitting_ratio"] = 1.5
        
        result = render_prompt("analyst.jinja2", sample_context)
        assert "Walk-Forward Analysis" in result
        assert "1.800" in result  # train sharpe
        assert "1.200" in result  # test sharpe
        assert "1.50" in result   # overfitting ratio
    
    def test_analyst_with_iteration_history(self, sample_context):
        """Le prompt inclut l'historique si disponible."""
        sample_context["iteration_history"] = [
            {"iteration": 1, "sharpe_ratio": 1.2, "total_return": 0.15},
            {"iteration": 2, "sharpe_ratio": 1.3, "total_return": 0.18},
            {"iteration": 3, "sharpe_ratio": 1.4, "total_return": 0.22},
        ]
        
        result = render_prompt("analyst.jinja2", sample_context)
        assert "Previous Iterations" in result
        assert "Sharpe=1.20" in result or "1.20" in result


class TestStrategistTemplate:
    """Tests du template strategist.jinja2."""
    
    @pytest.fixture
    def sample_context(self):
        """Contexte de test pour le strategist."""
        # Mock ParameterConfig
        param_specs = [
            Mock(name="fast_period", min_value=5, max_value=50, step=1, current_value=12),
            Mock(name="slow_period", min_value=20, max_value=200, step=5, current_value=26),
        ]
        
        metrics = MetricsSnapshot(
            sharpe_ratio=1.5, total_return=0.25, max_drawdown=-0.15,
            win_rate=0.55, profit_factor=1.8, total_trades=150
        )
        
        return {
            "strategy_name": "ema_cross",
            "iteration": 3,
            "param_specs": param_specs,
            "current_params": {"fast_period": 12, "slow_period": 26},
            "current_metrics": metrics,
            "overfitting_ratio": 1.2,
            "analyst_report": "Strategy shows good performance...",
            "best_metrics": None,
            "best_params": None,
            "optimization_target": "sharpe_ratio",
            "min_sharpe": 1.0,
            "max_drawdown_limit": 0.20,
            "min_trades": 100,
        }
    
    def test_strategist_template_renders(self, sample_context):
        """Template strategist se rend sans erreur."""
        result = render_prompt("strategist.jinja2", sample_context)
        assert result is not None
        assert len(result) > 0
    
    def test_strategist_contains_constraints(self, sample_context):
        """Le prompt contient les contraintes de paramètres."""
        result = render_prompt("strategist.jinja2", sample_context)
        assert "fast_period" in result
        assert "min=5" in result
        assert "max=50" in result
        assert "slow_period" in result
        assert "min=20" in result
        assert "max=200" in result
    
    def test_strategist_contains_current_performance(self, sample_context):
        """Le prompt contient les métriques actuelles."""
        result = render_prompt("strategist.jinja2", sample_context)
        assert "Current Performance" in result
        assert "1.500" in result  # sharpe
    
    def test_strategist_with_overfitting_warning(self, sample_context):
        """Le prompt affiche un warning si overfitting élevé."""
        sample_context["overfitting_ratio"] = 1.8
        result = render_prompt("strategist.jinja2", sample_context)
        assert "⚠️" in result or "HIGH overfitting" in result
    
    def test_strategist_with_analyst_report(self, sample_context):
        """Le prompt inclut le rapport de l'analyst."""
        result = render_prompt("strategist.jinja2", sample_context)
        assert "Analyst Report Summary" in result
        assert "Strategy shows good performance" in result


class TestCriticTemplate:
    """Tests du template critic.jinja2."""
    
    @pytest.fixture
    def sample_context(self):
        """Contexte de test pour le critic."""
        param_specs = [
            Mock(name="fast_period", min_value=5, max_value=50),
            Mock(name="slow_period", min_value=20, max_value=200),
        ]
        
        metrics = MetricsSnapshot(
            sharpe_ratio=1.5, total_return=0.25, max_drawdown=-0.15,
            win_rate=0.55, profit_factor=1.8, total_trades=150
        )
        
        proposals = [
            {
                "id": 1,
                "name": "Conservative Adjustment",
                "priority": "HIGH",
                "risk_level": "LOW",
                "parameters": {"fast_period": 10, "slow_period": 30},
                "rationale": "Smoother signals",
                "expected_impact": {"sharpe_ratio": "+5% to +10%"}
            },
            {
                "id": 2,
                "name": "Aggressive Tuning",
                "priority": "MEDIUM",
                "risk_level": "HIGH",
                "parameters": {"fast_period": 8, "slow_period": 50},
                "rationale": "Faster entries",
                "expected_impact": {"sharpe_ratio": "+15% to +20%"}
            }
        ]
        
        return {
            "strategy_name": "ema_cross",
            "iteration": 3,
            "current_metrics": metrics,
            "overfitting_ratio": 1.2,
            "analyst_report": "Performance is solid but could improve...",
            "strategist_proposals": proposals,
            "current_params": {"fast_period": 12, "slow_period": 26},
            "param_specs": param_specs,
            "min_sharpe": 1.0,
            "max_drawdown_limit": 0.20,
            "max_overfitting_ratio": 1.5,
        }
    
    def test_critic_template_renders(self, sample_context):
        """Template critic se rend sans erreur."""
        result = render_prompt("critic.jinja2", sample_context)
        assert result is not None
        assert len(result) > 0
    
    def test_critic_contains_proposals(self, sample_context):
        """Le prompt contient toutes les propositions."""
        result = render_prompt("critic.jinja2", sample_context)
        assert "Proposal 1" in result
        assert "Proposal 2" in result
        assert "Conservative Adjustment" in result
        assert "Aggressive Tuning" in result
    
    def test_critic_shows_parameter_changes(self, sample_context):
        """Le prompt montre les changements de paramètres."""
        result = render_prompt("critic.jinja2", sample_context)
        assert "fast_period:" in result
        assert "12 →" in result or "12 ->" in result  # current -> new
    
    def test_critic_shows_baseline_performance(self, sample_context):
        """Le prompt montre la performance baseline."""
        result = render_prompt("critic.jinja2", sample_context)
        assert "Current Performance Baseline" in result
        assert "1.500" in result  # sharpe


class TestValidatorTemplate:
    """Tests du template validator.jinja2."""
    
    @pytest.fixture
    def sample_context(self):
        """Contexte de test pour le validator."""
        metrics = MetricsSnapshot(
            sharpe_ratio=1.5, total_return=0.25, max_drawdown=-0.15,
            win_rate=0.55, profit_factor=1.8, total_trades=150
        )
        
        proposals = [
            {
                "id": 1,
                "name": "Best Proposal",
                "critic_evaluation": {
                    "recommendation": "APPROVE",
                    "overfitting_score": 25,
                    "robustness_score": 85
                }
            }
        ]
        
        objective_check = {
            "sharpe_meets_minimum": True,
            "drawdown_within_limit": True,
            "overfitting_acceptable": True,
            "sufficient_trades": True,
            "critic_approved": True,
        }
        
        return {
            "strategy_name": "ema_cross",
            "iteration": 5,
            "objective_check": objective_check,
            "current_metrics": metrics,
            "min_sharpe": 1.0,
            "max_drawdown_limit": 0.20,
            "min_trades": 100,
            "overfitting_ratio": 1.2,
            "max_overfitting_ratio": 1.5,
            "train_metrics": None,
            "test_metrics": None,
            "analyst_report": "Solid performance with room for improvement...",
            "strategist_proposals": proposals,
            "critic_concerns": ["Minor concern about market regime"],
            "iteration_history": [{"iteration": i} for i in range(1, 6)],
            "best_metrics": metrics,
        }
    
    def test_validator_template_renders(self, sample_context):
        """Template validator se rend sans erreur."""
        result = render_prompt("validator.jinja2", sample_context)
        assert result is not None
        assert len(result) > 0
    
    def test_validator_shows_criteria_check(self, sample_context):
        """Le prompt montre le check des critères."""
        result = render_prompt("validator.jinja2", sample_context)
        assert "OBJECTIVE CRITERIA CHECK" in result
        assert "✅ PASS" in result or "PASS" in result
    
    def test_validator_shows_all_criteria_met(self, sample_context):
        """Le prompt indique si tous les critères sont remplis."""
        result = render_prompt("validator.jinja2", sample_context)
        assert "All criteria met: YES" in result
    
    def test_validator_shows_failed_criteria(self, sample_context):
        """Le prompt montre les critères échoués."""
        sample_context["objective_check"]["sharpe_meets_minimum"] = False
        result = render_prompt("validator.jinja2", sample_context)
        assert "❌ FAIL" in result or "FAIL" in result
    
    def test_validator_shows_proposal_evaluations(self, sample_context):
        """Le prompt montre les évaluations des propositions."""
        result = render_prompt("validator.jinja2", sample_context)
        assert "PROPOSALS EVALUATED" in result
        assert "Proposal 1" in result
        assert "APPROVE" in result
        assert "25/100" in result  # overfitting score
        assert "85/100" in result  # robustness score
    
    def test_validator_shows_concerns(self, sample_context):
        """Le prompt montre les concerns du critic."""
        result = render_prompt("validator.jinja2", sample_context)
        assert "CRITIC CONCERNS" in result
        assert "Minor concern about market regime" in result
    
    def test_validator_shows_iteration_history(self, sample_context):
        """Le prompt montre l'historique."""
        result = render_prompt("validator.jinja2", sample_context)
        assert "ITERATION HISTORY" in result
        assert "Total iterations: 5" in result


class TestAgentIntegration:
    """Tests d'intégration avec les agents réels."""
    
    def test_analyst_uses_template(self):
        """L'AnalystAgent utilise bien le template."""
        from agents.analyst import AnalystAgent
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="test-model",
            ollama_host="http://localhost:11434"
        )
        
        agent = AnalystAgent(config)
        
        # Mock context minimal
        context = Mock()
        context.strategy_name = "test_strategy"
        context.strategy_description = None
        context.data_symbol = "TEST"
        context.data_timeframe = "1h"
        context.data_date_range = "2023"
        context.data_rows = 100
        context.iteration = 1
        context.current_params = {"param": 10}
        context.current_metrics = Mock(to_summary_str=lambda: "Metrics summary")
        context.train_metrics = None
        context.test_metrics = None
        context.overfitting_ratio = 0.0
        context.iteration_history = []
        context.optimization_target = "sharpe"
        context.min_sharpe = 1.0
        context.max_drawdown_limit = 0.2
        context.min_trades = 50
        context.max_overfitting_ratio = 1.5
        
        # Appeler _build_analysis_prompt
        prompt = agent._build_analysis_prompt(context)
        
        # Vérifier que le prompt est généré
        assert prompt is not None
        assert len(prompt) > 0
        assert "test_strategy" in prompt
        assert "TEST" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
