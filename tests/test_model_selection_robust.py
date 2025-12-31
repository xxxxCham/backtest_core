"""
Module-ID: tests.test_model_selection_robust

Purpose: Tester robustesse sélection modèles LLM (retry Ollama, fallback lists, timeout).

Role in pipeline: testing

Key components: TestModelSelectionRobustness, test_retry_on_ollama_connection_error

Inputs: Mock httpx.get, patch time.sleep, RoleModelConfig

Outputs: Retry logic validée, fallback OK si Ollama down

Dependencies: pytest, unittest.mock, agents.model_config

Conventions: 3 retries max; timeout 2s; connection error handling.

Read-if: Modification retry logic ou fallback.

Skip-if: Tests model selection non critiques.
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from agents.model_config import RoleModelConfig


class TestModelSelectionRobustness:
    """Tests de robustesse pour la sélection de modèles."""

    def test_retry_on_ollama_connection_error(self):
        """Test: retry automatique si Ollama non accessible initialement."""
        with patch("agents.model_config.httpx.get") as mock_get, \
             patch("agents.model_config.time.sleep"):
            # Simule 2 échecs puis succès
            mock_get.side_effect = [
                Exception("Connection refused"),  # Tentative 1: échec
                Exception("Connection refused"),  # Tentative 2: échec
                MagicMock(  # Tentative 3: succès
                    status_code=200,
                    json=lambda: {
                        "models": [
                            {"name": "deepseek-r1:8b"},
                            {"name": "deepseek-r1:32b"},
                        ]
                    }
                ),
            ]

            config = RoleModelConfig()
            installed = config.get_installed_models()

            assert len(installed) == 2
            assert "deepseek-r1:8b" in installed
            assert "deepseek-r1:32b" in installed
            assert mock_get.call_count == 3  # 3 tentatives

    def test_fallback_to_configured_models_when_ollama_down(self):
        """Test: fallback sur modèles configurés si Ollama inaccessible."""
        with patch("agents.model_config.httpx.get") as mock_get, \
             patch("agents.model_config.time.sleep"):
            # Simule Ollama inaccessible (timeout sur toutes tentatives)
            mock_get.side_effect = Exception("Connection refused")

            config = RoleModelConfig()
            installed = config.get_installed_models()

            # Cache vide car Ollama inaccessible
            assert len(installed) == 0

            # Mais get_model() doit quand même retourner un modèle configuré
            model = config.get_model("analyst", iteration=1)
            assert model is not None
            assert model in ["deepseek-r1:8b", "mistral:7b-instruct", "martain7r/finance-llama-8b:q4_k_m", "gemma3:12b"]

    def test_fallback_levels(self):
        """Test: cascade de fallbacks (niveau 1, 2, 3)."""
        with patch("agents.model_config.httpx.get") as mock_get:
            # Niveau 1: Ollama OK avec 1 modèle installé (pas dans la liste configurée)
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"models": [{"name": "unknown-model:7b"}]}
            )

            config = RoleModelConfig()

            # Niveau 1 échoue (modèle installé pas dans config)
            # Niveau 2: fallback sur modèles configurés
            model = config.get_model("analyst", iteration=1)
            assert model is not None
            assert model in ["deepseek-r1:8b", "mistral:7b-instruct", "martain7r/finance-llama-8b:q4_k_m", "gemma3:12b"]

    def test_no_warning_when_models_available_after_retry(self, caplog):
        """Test: aucun warning si modèles trouvés après retry."""
        with patch("agents.model_config.httpx.get") as mock_get, \
             patch("agents.model_config.time.sleep"):
            # Première tentative échoue, deuxième réussit
            mock_get.side_effect = [
                Exception("Connection refused"),
                MagicMock(
                    status_code=200,
                    json=lambda: {"models": [{"name": "deepseek-r1:8b"}]}
                ),
            ]

            with caplog.at_level(logging.WARNING):
                config = RoleModelConfig()
                model = config.get_model("analyst", iteration=1)

            # Pas de warning "Aucun modèle disponible" car trouvé après retry
            warning_messages = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
            assert not any("Aucun modèle disponible" in msg for msg in warning_messages)
            assert model == "deepseek-r1:8b"

    def test_heavy_models_blocked_early_iterations(self):
        """Test: modèles lourds bloqués en début d'itération."""
        with patch("agents.model_config.httpx.get") as mock_get:
            # Tous les modèles installés
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "models": [
                        {"name": "deepseek-r1:8b"},     # LIGHT
                        {"name": "deepseek-r1:32b"},    # HEAVY
                        {"name": "qwq:32b"},            # HEAVY
                    ]
                }
            )

            config = RoleModelConfig()

            # Iteration 1: pas de heavy (critic.allow_heavy_after_iteration=2)
            model = config.get_model("critic", iteration=1, allow_heavy=False)
            assert model not in ["deepseek-r1:32b", "qwq:32b"]

            # Iteration 3: heavy autorisés
            model = config.get_model("critic", iteration=3, allow_heavy=False)
            assert model in ["deepseek-r1-distill:14b", "mistral:22b", "gemma3:27b", "deepseek-r1:32b", "qwq:32b"]

    def test_retry_delay_timing(self):
        """Test: vérifier le délai entre retries."""
        with patch("agents.model_config.httpx.get") as mock_get, \
             patch("agents.model_config.time.sleep") as mock_sleep:

            # Simule 3 échecs
            mock_get.side_effect = Exception("Connection refused")

            _ = time.time()
            RoleModelConfig()
            _ = time.time()

            # Par défaut: 5 tentatives => 4 sleeps (backoff exponentiel: 1,2,4,8)
            assert mock_sleep.call_count == 4
            sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_args == [1.0, 2.0, 4.0, 8.0]

    def test_fallback_ultimate_any_installed_model(self):
        """Test: fallback ultime sur n'importe quel modèle installé."""
        with patch("agents.model_config.httpx.get") as mock_get:
            # Ollama installé mais avec un modèle inconnu
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"models": [{"name": "completely-unknown:99b"}]}
            )

            config = RoleModelConfig()

            # Aucun modèle configuré n'est installé
            # → Fallback niveau 2: modèles configurés (car installed est vide après filtrage)
            # → Fallback niveau 3: premier installé si aucun configuré ne match
            model = config.get_model("analyst", iteration=1)

            # Le comportement actuel: niveau 2 activé car installed_models vide après filtrage
            # On s'attend à un modèle configuré (martain7r/finance-llama-8b:q4_k_m par exemple)
            assert model in ["deepseek-r1:8b", "mistral:7b-instruct", "martain7r/finance-llama-8b:q4_k_m", "gemma3:12b", "completely-unknown:99b"]

    def test_empty_ollama_response(self):
        """Test: Ollama retourne 0 modèles (installé mais vide)."""
        with patch("agents.model_config.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"models": []}  # Liste vide
            )

            config = RoleModelConfig()
            installed = config.get_installed_models()

            assert len(installed) == 0

            # Fallback sur modèles configurés (niveau 2)
            model = config.get_model("analyst", iteration=1)
            assert model is not None
            assert model in ["deepseek-r1:8b", "mistral:7b-instruct", "martain7r/finance-llama-8b:q4_k_m", "gemma3:12b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
