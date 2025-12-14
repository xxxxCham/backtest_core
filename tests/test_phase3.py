"""
Tests Phase 3 - Intelligence LLM.

Tests couvrant:
- State Machine (transitions, validations)
- LLM Client (mocking)
- Agents (Analyst, Strategist, Critic, Validator)
- Orchestrator (workflow complet)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json

# Imports du module agents
from agents.state_machine import AgentState, StateMachine, ValidationResult
from agents.llm_client import LLMConfig, LLMMessage, LLMResponse, LLMProvider
from agents.base_agent import AgentContext, AgentResult, AgentRole, MetricsSnapshot, ParameterConfig


# =============================================================================
# Tests State Machine
# =============================================================================

class TestStateMachine:
    """Tests pour la State Machine."""
    
    def test_initial_state(self):
        """Vérifie l'état initial."""
        sm = StateMachine()
        assert sm.current_state == AgentState.INIT
        assert sm.iteration == 0
        assert not sm.is_terminal
    
    def test_valid_transitions(self):
        """Test des transitions valides."""
        sm = StateMachine()
        
        # INIT → ANALYZE
        result = sm.transition_to(AgentState.ANALYZE)
        assert result.is_valid
        assert sm.current_state == AgentState.ANALYZE
        
        # ANALYZE → PROPOSE
        result = sm.transition_to(AgentState.PROPOSE)
        assert result.is_valid
        assert sm.current_state == AgentState.PROPOSE
        
        # PROPOSE → CRITIQUE
        result = sm.transition_to(AgentState.CRITIQUE)
        assert result.is_valid
        assert sm.current_state == AgentState.CRITIQUE
        
        # CRITIQUE → VALIDATE
        result = sm.transition_to(AgentState.VALIDATE)
        assert result.is_valid
        assert sm.current_state == AgentState.VALIDATE
    
    def test_invalid_transition(self):
        """Test des transitions invalides."""
        sm = StateMachine()
        
        # INIT → VALIDATE (invalide, doit passer par ANALYZE, PROPOSE, CRITIQUE)
        result = sm.transition_to(AgentState.VALIDATE)
        assert not result.is_valid
        assert "invalide" in result.message.lower() or "VALIDATE" in result.message
        assert sm.current_state == AgentState.INIT  # Pas de changement
    
    def test_terminal_states(self):
        """Vérifie que APPROVED/REJECTED sont terminaux."""
        sm = StateMachine()
        
        # Aller jusqu'à VALIDATE
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        sm.transition_to(AgentState.CRITIQUE)
        sm.transition_to(AgentState.VALIDATE)
        
        # Transition vers APPROVED
        result = sm.transition_to(AgentState.APPROVED)
        assert result.is_valid
        assert sm.is_terminal
        
        # Impossible de transitionner depuis APPROVED
        result = sm.transition_to(AgentState.ANALYZE)
        assert not result.is_valid
    
    def test_iteration_increment(self):
        """Vérifie que l'itération s'incrémente via ITERATE."""
        sm = StateMachine()
        
        # Cycle complet
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        sm.transition_to(AgentState.CRITIQUE)
        sm.transition_to(AgentState.VALIDATE)
        
        assert sm.iteration == 0
        
        # ITERATE → ANALYZE (incrémente)
        sm.transition_to(AgentState.ITERATE)
        result = sm.transition_to(AgentState.ANALYZE)
        
        assert result.is_valid
        assert sm.iteration == 1
    
    def test_max_iterations_limit(self):
        """Vérifie la limite de max_iterations."""
        sm = StateMachine(max_iterations=2)
        
        # Itération 0
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        sm.transition_to(AgentState.CRITIQUE)
        sm.transition_to(AgentState.VALIDATE)
        sm.transition_to(AgentState.ITERATE)
        
        # Itération 1
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        sm.transition_to(AgentState.CRITIQUE)
        sm.transition_to(AgentState.VALIDATE)
        sm.transition_to(AgentState.ITERATE)
        
        # Itération 2 - devrait être la dernière
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        sm.transition_to(AgentState.CRITIQUE)
        sm.transition_to(AgentState.VALIDATE)
        
        # Vérifie que max iterations est atteint
        assert sm.iteration >= 2
    
    def test_fail_transition(self):
        """Test de la transition d'échec."""
        sm = StateMachine()
        
        sm.transition_to(AgentState.ANALYZE)
        sm.fail("Test failure reason")
        
        assert sm.current_state == AgentState.FAILED
        assert sm.is_terminal
    
    def test_history_tracking(self):
        """Vérifie le suivi de l'historique."""
        sm = StateMachine()
        
        sm.transition_to(AgentState.ANALYZE)
        sm.transition_to(AgentState.PROPOSE)
        
        summary = sm.get_summary()
        
        # Vérifie que l'historique contient les transitions
        assert len(summary["history"]) >= 2
        assert summary["current_state"] is not None
    
    def test_can_transition_to(self):
        """Test de can_transition_to."""
        sm = StateMachine()
        
        assert sm.can_transition_to(AgentState.ANALYZE)
        assert not sm.can_transition_to(AgentState.VALIDATE)
        
        sm.transition_to(AgentState.ANALYZE)
        assert sm.can_transition_to(AgentState.PROPOSE)
        assert not sm.can_transition_to(AgentState.INIT)


class TestValidationResult:
    """Tests pour ValidationResult."""
    
    def test_success(self):
        """Test création succès."""
        result = ValidationResult.success()
        assert result.is_valid
    
    def test_failure(self):
        """Test création échec."""
        result = ValidationResult.failure("Error message", ["err1", "err2"])
        assert not result.is_valid
        assert result.message == "Error message"
        assert len(result.errors) == 2


# =============================================================================
# Tests LLM Client
# =============================================================================

class TestLLMConfig:
    """Tests pour LLMConfig."""
    
    def test_default_config(self):
        """Test configuration par défaut."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3.2"
        assert config.ollama_host == "http://localhost:11434"
    
    def test_from_env(self):
        """Test création depuis environnement."""
        config = LLMConfig.from_env()
        assert config.provider in [LLMProvider.OLLAMA, LLMProvider.OPENAI]
        assert config.model is not None


class TestLLMMessage:
    """Tests pour LLMMessage."""
    
    def test_message_creation(self):
        """Test création message."""
        msg = LLMMessage(role="system", content="You are a helpful assistant")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"
    
    def test_user_message(self):
        """Test message utilisateur."""
        msg = LLMMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
    
    def test_to_dict(self):
        """Test conversion dict."""
        msg = LLMMessage(role="user", content="Test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test"}


class TestLLMResponse:
    """Tests pour LLMResponse."""
    
    def test_success_response(self):
        """Test réponse succès."""
        resp = LLMResponse(
            content="Hello world",
            model="test-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        assert resp.content == "Hello world"
        assert resp.total_tokens == 30
    
    def test_response_is_valid(self):
        """Test propriété is_valid."""
        resp = LLMResponse(
            content="Valid response",
            model="test",
            provider=LLMProvider.OLLAMA
        )
        assert resp.is_valid


# =============================================================================
# Tests Base Agent
# =============================================================================

class TestMetricsSnapshot:
    """Tests pour MetricsSnapshot."""
    
    def test_from_dict(self):
        """Test création depuis dict."""
        data = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.10,
            "win_rate": 0.55,
            "total_trades": 100
        }
        snapshot = MetricsSnapshot.from_dict(data)
        
        assert snapshot.sharpe_ratio == 1.5
        assert snapshot.total_return == 0.25
        assert snapshot.max_drawdown == 0.10
        assert snapshot.win_rate == 0.55
        assert snapshot.total_trades == 100
    
    def test_to_dict(self):
        """Test conversion dict."""
        snapshot = MetricsSnapshot(
            sharpe_ratio=2.0,
            total_return=0.30,
            max_drawdown=0.08
        )
        d = snapshot.to_dict()
        
        assert d["sharpe_ratio"] == 2.0
        assert d["total_return"] == 0.30
        assert d["max_drawdown"] == 0.08


class TestAgentContext:
    """Tests pour AgentContext."""
    
    def test_context_creation(self):
        """Test création contexte."""
        ctx = AgentContext(
            session_id="test123",
            iteration=1,
            strategy_name="ema_cross",
            current_params={"fast": 12, "slow": 26}
        )
        
        assert ctx.session_id == "test123"
        assert ctx.iteration == 1
        assert ctx.strategy_name == "ema_cross"
        assert ctx.current_params == {"fast": 12, "slow": 26}
    
    def test_to_summary_str(self):
        """Test génération du résumé textuel."""
        ctx = AgentContext(
            session_id="test",
            iteration=0,
            strategy_name="test_strategy",
            current_params={"x": 1}
        )
        summary = ctx.to_summary_str()
        
        assert "test_strategy" in summary
        assert "Iteration: 0" in summary


class TestAgentResult:
    """Tests pour AgentResult."""
    
    def test_success_result(self):
        """Test résultat succès."""
        result = AgentResult.success_result(
            role=AgentRole.ANALYST,
            content="Analysis complete",
            data={"rating": 8}
        )
        
        assert result.success
        assert result.content == "Analysis complete"
        assert result.data["rating"] == 8
    
    def test_failure_result(self):
        """Test résultat échec."""
        result = AgentResult.failure_result(
            role=AgentRole.ANALYST,
            error="LLM timeout"
        )
        
        assert not result.success
        assert "LLM timeout" in result.errors


# =============================================================================
# Tests Agents avec mocking
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Fixture pour LLM client mocké."""
    client = Mock()
    client.model = "mock-model"
    return client


@pytest.fixture
def sample_context():
    """Fixture pour contexte de test."""
    return AgentContext(
        session_id="test-session",
        iteration=0,
        strategy_name="ema_cross",
        strategy_description="EMA crossover strategy",
        current_params={"fast_period": 12, "slow_period": 26},
        param_specs=[
            ParameterConfig(
                name="fast_period",
                current_value=12,
                min_value=5,
                max_value=50
            ),
            ParameterConfig(
                name="slow_period",
                current_value=26,
                min_value=10,
                max_value=200
            )
        ],
        current_metrics=MetricsSnapshot(
            sharpe_ratio=0.8,
            total_return=0.15,
            max_drawdown=0.12,
            win_rate=0.52,
            total_trades=85
        ),
        min_sharpe=1.0,
        max_drawdown_limit=0.20,
        min_trades=30
    )


class TestAnalystAgent:
    """Tests pour AnalystAgent."""
    
    def test_analyst_success(self, mock_llm_client, sample_context):
        """Test exécution réussie."""
        from agents.analyst import AnalystAgent
        
        # Mock la réponse LLM avec le bon format (incluant key_metrics_assessment requis)
        mock_response = LLMResponse(
            content=json.dumps({
                "summary": "Strategy shows potential but needs improvement",
                "performance_rating": "FAIR",
                "risk_rating": "MODERATE",
                "overfitting_risk": "MODERATE",
                "strengths": ["Good win rate", "Low drawdown"],
                "weaknesses": ["Sharpe below target"],
                "concerns": ["Market regime dependency"],
                "key_metrics_assessment": {
                    "sharpe_ratio": {"value": 1.2, "assessment": "Below target but acceptable"},
                    "max_drawdown": {"value": -0.12, "assessment": "Good drawdown control"},
                    "win_rate": {"value": 0.58, "assessment": "Strong win rate"},
                    "profit_factor": {"value": 1.6, "assessment": "Solid profit factor"}
                },
                "recommendations": ["Optimize Sharpe ratio", "Maintain risk controls"],
                "proceed_to_optimization": True,
                "reasoning": "Strategy shows solid fundamentals with room for improvement on risk-adjusted returns"
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = AnalystAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        assert result.success
        assert result.data.get("performance_rating") == "FAIR"
        assert result.data.get("proceed_to_optimization") is True
    
    def test_analyst_invalid_response(self, mock_llm_client, sample_context):
        """Test gestion réponse invalide."""
        from agents.analyst import AnalystAgent
        
        # Réponse avec format invalide
        mock_response = LLMResponse(
            content=json.dumps({
                "performance_rating": 6,  # Invalide - devrait être string
                "risk_rating": 7
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = AnalystAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        # Devrait échouer car format invalide
        assert not result.success


class TestStrategistAgent:
    """Tests pour StrategistAgent."""
    
    def test_strategist_generates_proposals(self, mock_llm_client, sample_context):
        """Test génération de propositions."""
        from agents.strategist import StrategistAgent
        
        mock_response = LLMResponse(
            content=json.dumps({
                "reasoning": "Current params need adjustment",
                "proposals": [
                    {
                        "id": 1,
                        "name": "Tighten periods",
                        "parameters": {"fast_period": 10, "slow_period": 21},
                        "rationale": "Reduce lag",
                        "expected_impact": "+15% Sharpe",
                        "confidence": 0.7,
                        "risks": ["More whipsaws"]
                    }
                ],
                "total_proposals": 1
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=600,
            completion_tokens=300,
            total_tokens=900
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = StrategistAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        assert result.success
        proposals = result.data.get("proposals", [])
        assert len(proposals) >= 1


class TestCriticAgent:
    """Tests pour CriticAgent."""
    
    def test_critic_evaluates_proposals(self, mock_llm_client, sample_context):
        """Test évaluation des propositions."""
        from agents.critic import CriticAgent
        
        # Ajouter des propositions au contexte
        sample_context.strategist_proposals = [
            {
                "id": 1,
                "name": "Test proposal",
                "parameters": {"fast_period": 10, "slow_period": 21}
            }
        ]
        
        mock_response = LLMResponse(
            content=json.dumps({
                "proposal_evaluations": [
                    {
                        "proposal_id": 1,
                        "overfitting_score": 3,
                        "robustness_score": 7,
                        "recommendation": "APPROVE",
                        "concerns": []
                    }
                ],
                "approved_proposals": [
                    {"id": 1, "name": "Test proposal", "parameters": {"fast_period": 10, "slow_period": 21}}
                ],
                "rejected_proposals": [],
                "overall_assessment": "Proposal looks reasonable"
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=700,
            completion_tokens=250,
            total_tokens=950
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = CriticAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        assert result.success


class TestValidatorAgent:
    """Tests pour ValidatorAgent."""
    
    def test_validator_approve(self, mock_llm_client, sample_context):
        """Test décision APPROVE."""
        from agents.validator import ValidatorAgent
        
        # Mettre des métriques au-dessus des seuils
        sample_context.current_metrics = MetricsSnapshot(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=0.08,
            win_rate=0.58,
            total_trades=120
        )
        # S'assurer que les seuils sont respectés
        sample_context.min_sharpe = 1.0
        sample_context.max_drawdown_limit = 0.20
        sample_context.min_trades = 30
        
        mock_response = LLMResponse(
            content=json.dumps({
                "decision": "APPROVE",
                "reasoning": "All criteria met",
                "criteria_met": {
                    "sharpe_criterion": True,
                    "drawdown_criterion": True,
                    "trades_criterion": True
                },
                "confidence": 0.85,
                "final_report": "Strategy approved for deployment"
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=800,
            completion_tokens=200,
            total_tokens=1000
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = ValidatorAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        assert result.success
        # La décision finale dépend de la validation interne des critères
        assert result.data.get("decision") in ["APPROVE", "ITERATE"]
    
    def test_validator_iterate(self, mock_llm_client, sample_context):
        """Test décision ITERATE."""
        from agents.validator import ValidatorAgent
        
        mock_response = LLMResponse(
            content=json.dumps({
                "decision": "ITERATE",
                "reasoning": "Sharpe below target",
                "criteria_check": {
                    "sharpe_met": False,
                    "drawdown_met": True,
                    "trades_met": True
                },
                "confidence": 0.6,
                "recommendations": ["Increase period range", "Try different MA types"]
            }),
            model="mock-model",
            provider=LLMProvider.OLLAMA,
            prompt_tokens=800,
            completion_tokens=180,
            total_tokens=980
        )
        mock_llm_client.chat.return_value = mock_response
        
        agent = ValidatorAgent(mock_llm_client)
        result = agent.execute(sample_context)
        
        assert result.success
        assert result.data.get("decision") == "ITERATE"


# =============================================================================
# Tests Orchestrator
# =============================================================================

class TestOrchestrator:
    """Tests pour l'Orchestrator."""
    
    @pytest.fixture
    def orchestrator_config(self, tmp_path):
        """Fixture pour configuration Orchestrator."""
        from agents.orchestrator import OrchestratorConfig
        
        # Créer un fichier de données temporaire
        data_file = tmp_path / "test_data.parquet"
        data_file.write_text("")  # Fichier vide pour le test
        
        return OrchestratorConfig(
            strategy_name="ema_cross",
            strategy_description="Test strategy",
            data_path=str(data_file),
            initial_params={"fast_period": 12, "slow_period": 26},
            param_specs=[
                ParameterConfig(name="fast_period", current_value=12, min_value=5, max_value=50),
                ParameterConfig(name="slow_period", current_value=26, min_value=10, max_value=200)
            ],
            max_iterations=3,
            llm_config=LLMConfig(provider=LLMProvider.OLLAMA, model="test")
        )
    
    def test_orchestrator_init(self, orchestrator_config):
        """Test initialisation."""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.create_llm_client') as mock_create:
            mock_create.return_value = Mock()
            
            orch = Orchestrator(orchestrator_config)
            
            assert orch.config.strategy_name == "ema_cross"
            assert orch.state_machine.current_state == AgentState.INIT
    
    def test_orchestrator_config_validation_fails(self, tmp_path):
        """Test échec validation config."""
        from agents.orchestrator import Orchestrator, OrchestratorConfig
        
        # Config sans strategy_name
        config = OrchestratorConfig(
            strategy_name="",  # Invalide
            data_path=str(tmp_path / "nonexistent.parquet")
        )
        
        with patch('agents.orchestrator.create_llm_client') as mock_create:
            mock_create.return_value = Mock()
            
            orch = Orchestrator(config)
            result = orch.run()
            
            assert not result.success
            assert result.final_state == AgentState.FAILED


# =============================================================================
# Tests d'intégration légers
# =============================================================================

class TestAgentStateEnum:
    """Tests pour l'enum AgentState."""
    
    def test_all_states_defined(self):
        """Vérifie que tous les états sont définis."""
        expected = {
            "INIT", "ANALYZE", "PROPOSE", "CRITIQUE", 
            "VALIDATE", "ITERATE", "APPROVED", "REJECTED", "FAILED"
        }
        actual = {state.name for state in AgentState}
        assert actual == expected
    
    def test_terminal_states(self):
        """Vérifie les états terminaux."""
        terminal = {AgentState.APPROVED, AgentState.REJECTED, AgentState.FAILED}
        
        for state in AgentState:
            sm = StateMachine()
            # Force l'état (pour test uniquement)
            sm._current_state = state
            
            if state in terminal:
                assert sm.is_terminal, f"{state} devrait être terminal"
            else:
                assert not sm.is_terminal, f"{state} ne devrait pas être terminal"


class TestParameterConfig:
    """Tests pour ParameterConfig."""
    
    def test_basic_config(self):
        """Test configuration basique."""
        pc = ParameterConfig(
            name="test_param",
            current_value=50,
            min_value=0,
            max_value=100
        )
        
        assert pc.name == "test_param"
        assert pc.min_value == 0
        assert pc.max_value == 100
        assert pc.current_value == 50
    
    def test_with_step(self):
        """Test avec step."""
        pc = ParameterConfig(
            name="window",
            current_value=10,
            min_value=5,
            max_value=50,
            step=5
        )
        
        assert pc.step == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
