"""
Tests pour la variable d'environnement UNLOAD_LLM_DURING_BACKTEST.

Vérifie que le comportement de déchargement du LLM peut être contrôlé
via la variable d'environnement.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ajouter le parent au path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from agents.autonomous_strategist import AutonomousStrategist
from agents.llm_client import LLMConfig, LLMProvider, OllamaClient


class TestUnloadLLMEnvironmentVariable:
    """Tests de la variable UNLOAD_LLM_DURING_BACKTEST."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        # Sauvegarder l'état initial
        self.original_env = os.environ.get('UNLOAD_LLM_DURING_BACKTEST')
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        # Restaurer l'état initial
        if self.original_env is not None:
            os.environ['UNLOAD_LLM_DURING_BACKTEST'] = self.original_env
        elif 'UNLOAD_LLM_DURING_BACKTEST' in os.environ:
            del os.environ['UNLOAD_LLM_DURING_BACKTEST']
    
    def test_default_behavior_false_when_not_set(self):
        """Par défaut, UNLOAD_LLM est False (CPU-only compatible)."""
        # Supprimer la variable si elle existe
        if 'UNLOAD_LLM_DURING_BACKTEST' in os.environ:
            del os.environ['UNLOAD_LLM_DURING_BACKTEST']
        
        # Créer un agent avec un mock client
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None  # Auto-détection depuis env
        )
        
        # Vérifier que c'est False par défaut
        assert agent.unload_llm_during_backtest is False
    
    def test_env_var_true_explicit(self):
        """Variable d'env à 'True' active le déchargement."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'True'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        assert agent.unload_llm_during_backtest is True
    
    def test_env_var_one_numeric(self):
        """Variable d'env à '1' active le déchargement."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = '1'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        assert agent.unload_llm_during_backtest is True
    
    def test_env_var_yes(self):
        """Variable d'env à 'yes' active le déchargement."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'yes'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        assert agent.unload_llm_during_backtest is True
    
    def test_env_var_false_explicit(self):
        """Variable d'env à 'False' désactive le déchargement."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'False'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        assert agent.unload_llm_during_backtest is False
    
    def test_env_var_zero_numeric(self):
        """Variable d'env à '0' désactive le déchargement."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = '0'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        assert agent.unload_llm_during_backtest is False
    
    def test_env_var_case_insensitive(self):
        """Vérifier que la casse n'importe pas."""
        test_cases = [
            ('TRUE', True),
            ('true', True),
            ('True', True),
            ('FALSE', False),
            ('false', False),
            ('False', False),
        ]
        
        for env_value, expected in test_cases:
            os.environ['UNLOAD_LLM_DURING_BACKTEST'] = env_value
            
            config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
            client = OllamaClient(config)
            
            agent = AutonomousStrategist(
                llm_client=client,
                unload_llm_during_backtest=None
            )
            
            assert agent.unload_llm_during_backtest == expected, \
                f"Failed for env_value={env_value}"
    
    def test_explicit_parameter_overrides_env(self):
        """Paramètre explicite override la variable d'env."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'True'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        # Force False explicitement
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=False  # Explicit override
        )
        
        # Doit utiliser la valeur explicite, pas l'env
        assert agent.unload_llm_during_backtest is False
    
    @patch('agents.autonomous_strategist.GPUMemoryManager')
    @patch('agents.llm_client.OllamaClient')
    def test_backtest_with_unload_true_calls_manager(self, mock_client_class, mock_gpu_manager):
        """Quand UNLOAD=True, le GPU manager doit être utilisé."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'True'
        
        # Mock le client LLM
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.model = "test-model"
        mock_client_class.return_value = mock_client
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = mock_client
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        # Mock executor
        mock_executor = MagicMock()
        mock_executor.run.return_value = MagicMock()
        
        # Mock GPU manager instance
        mock_manager_instance = MagicMock()
        mock_manager_instance.unload.return_value = {"test": "state"}
        mock_gpu_manager.return_value = mock_manager_instance
        
        # Exécuter un backtest
        from agents.backtest_executor import BacktestRequest
        request = BacktestRequest(
            strategy_name="test_strategy",
            parameters={"param": 1},
            requested_by="test_agent",
        )
        
        agent._run_backtest_with_gpu_optimization(mock_executor, request)
        
        # Vérifier que le GPU manager a été appelé
        assert mock_manager_instance.unload.called
        assert mock_manager_instance.reload.called
    
    @patch('agents.autonomous_strategist.GPUMemoryManager')
    def test_backtest_with_unload_false_skips_manager(self, mock_gpu_manager):
        """Quand UNLOAD=False, le GPU manager ne doit pas être utilisé."""
        os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'False'
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="test-model")
        client = OllamaClient(config)
        
        agent = AutonomousStrategist(
            llm_client=client,
            unload_llm_during_backtest=None
        )
        
        # Mock executor
        mock_executor = MagicMock()
        mock_executor.run.return_value = MagicMock()
        
        # Exécuter un backtest
        from agents.backtest_executor import BacktestRequest
        request = BacktestRequest(
            strategy_name="test_strategy",
            parameters={"param": 1},
            requested_by="test_agent",
        )
        
        agent._run_backtest_with_gpu_optimization(mock_executor, request)
        
        # Vérifier que le GPU manager n'a PAS été créé
        assert not mock_gpu_manager.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
