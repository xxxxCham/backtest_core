"""
Module-ID: agents.model_config

Purpose: Configuration multi-modèles par rôle d'agent avec sélection intelligente (rapide/lourd par itération).

Role in pipeline: orchestration

Key components: RoleModelConfig, ModelCategory, KNOWN_MODELS, get_model

Inputs: role (analyst/strategist/critic/validator), iteration, allow_heavy flag

Outputs: Modèle sélectionné (aléatoire parmi configurés), fallback si non dispo

Dependencies: utils.log, httpx (Ollama discovery)

Conventions: ANALYST=rapide, STRATEGIST=moyen, CRITIC/VALIDATOR=lourd optionnel; early iterations excluent modèles lourds; fallback cascade si modèle absent.

Read-if: Ajout modèles, configuration par rôle, ou règles de sélection itérative.

Skip-if: Vous utilisez la config par défaut.
"""

from __future__ import annotations

# pylint: disable=logging-fstring-interpolation

import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)


def _ollama_base_url() -> str:
    """Retourne l'URL de base Ollama (avec fallback local)."""
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return base.rstrip("/")


def _fetch_ollama_tags_with_retries(
    *,
    max_attempts: int = 5,
    timeout_s: float = 3.0,
    base_backoff_s: float = 1.0,
) -> Optional[dict]:
    """Récupère /api/tags avec retries/backoff (Ollama peut démarrer lentement)."""
    url = f"{_ollama_base_url()}/api/tags"
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = httpx.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        if attempt < max_attempts - 1:
            sleep_s = float(base_backoff_s * (2 ** attempt))
            time.sleep(sleep_s)

    if last_exc is not None:
        logger.warning("Impossible de lister les modeles Ollama apres retries: %s", last_exc)
    return None


class ModelCategory(Enum):
    """Catégories de modèles par taille/vitesse."""

    LIGHT = "light"      # < 10B params, rapide (< 30s)
    MEDIUM = "medium"    # 10-30B params, modéré (30s-2min)
    HEAVY = "heavy"      # > 30B params, lent (> 2min)


@dataclass
class ModelInfo:
    """Information sur un modèle LLM."""

    name: str                      # Nom Ollama (ex: "deepseek-r1:32b")
    category: ModelCategory        # Catégorie de taille
    description: str = ""          # Description courte
    recommended_for: List[str] = field(default_factory=list)  # Rôles recommandés
    avg_response_time_s: float = 30.0  # Temps de réponse moyen estimé

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ModelInfo):
            return self.name == other.name
        return False


# Base de données des modèles connus avec leurs caractéristiques
KNOWN_MODELS: Dict[str, ModelInfo] = {
    # Light models (< 10B) - Rapides
    "deepseek-r1:8b": ModelInfo(
        name="deepseek-r1:8b",
        category=ModelCategory.LIGHT,
        description="DeepSeek R1 8B - Rapide, bon pour analyses simples",
        recommended_for=["analyst", "strategist"],
        avg_response_time_s=15.0,
    ),
    "mistral:7b-instruct": ModelInfo(
        name="mistral:7b-instruct",
        category=ModelCategory.LIGHT,
        description="Mistral 7B Instruct - Très rapide, polyvalent",
        recommended_for=["analyst", "strategist"],
        avg_response_time_s=10.0,
    ),
    "llama3.1:8b-local": ModelInfo(
        name="llama3.1:8b-local",
        category=ModelCategory.LIGHT,
        description="Llama 3.1 8B - Rapide, bonne qualité",
        recommended_for=["analyst", "strategist"],
        avg_response_time_s=20.0,
    ),
    "martain7r/finance-llama-8b:q4_k_m": ModelInfo(
        name="martain7r/finance-llama-8b:q4_k_m",
        category=ModelCategory.LIGHT,
        description="Finance Llama 8B - Spécialisé finance/trading",
        recommended_for=["analyst", "critic"],
        avg_response_time_s=15.0,
    ),

    # Medium models (10-30B) - Équilibrés
    "gemma3:12b": ModelInfo(
        name="gemma3:12b",
        category=ModelCategory.MEDIUM,
        description="Gemma 3 12B - Bon équilibre qualité/vitesse",
        recommended_for=["strategist", "critic"],
        avg_response_time_s=45.0,
    ),
    "deepseek-r1-distill:14b": ModelInfo(
        name="deepseek-r1-distill:14b",
        category=ModelCategory.MEDIUM,
        description="DeepSeek R1 Distill 14B - Raisonnement efficace",
        recommended_for=["strategist", "critic", "validator"],
        avg_response_time_s=60.0,
    ),
    "mistral:22b": ModelInfo(
        name="mistral:22b",
        category=ModelCategory.MEDIUM,
        description="Mistral 22B - Puissant et raisonnablement rapide",
        recommended_for=["critic", "validator"],
        avg_response_time_s=90.0,
    ),
    "gemma3:27b": ModelInfo(
        name="gemma3:27b",
        category=ModelCategory.MEDIUM,
        description="Gemma 3 27B - Très bonne qualité",
        recommended_for=["critic", "validator"],
        avg_response_time_s=120.0,
    ),

    # Heavy models (> 30B) - Puissants mais lents
    "deepseek-r1:32b": ModelInfo(
        name="deepseek-r1:32b",
        category=ModelCategory.HEAVY,
        description="DeepSeek R1 32B - Excellent raisonnement",
        recommended_for=["critic", "validator"],
        avg_response_time_s=180.0,
    ),
    "qwq:32b": ModelInfo(
        name="qwq:32b",
        category=ModelCategory.HEAVY,
        description="QwQ 32B - Raisonnement profond",
        recommended_for=["critic", "validator"],
        avg_response_time_s=200.0,
    ),
    "qwen2.5:32b": ModelInfo(
        name="qwen2.5:32b",
        category=ModelCategory.HEAVY,
        description="Qwen 2.5 32B - Polyvalent haute qualité",
        recommended_for=["strategist", "critic", "validator"],
        avg_response_time_s=150.0,
    ),
    "qwen3-vl:30b": ModelInfo(
        name="qwen3-vl:30b",
        category=ModelCategory.HEAVY,
        description="Qwen 3 VL 30B - Vision + Langage",
        recommended_for=["analyst"],  # Pour analyse de charts
        avg_response_time_s=180.0,
    ),
    "deepseek-r1:70b": ModelInfo(
        name="deepseek-r1:70b",
        category=ModelCategory.HEAVY,
        description="DeepSeek R1 70B - Maximum puissance, TRÈS LENT",
        recommended_for=["validator"],  # Réservé aux décisions critiques
        avg_response_time_s=600.0,  # ~10 minutes
    ),
    "gpt-oss:20b": ModelInfo(
        name="gpt-oss:20b",
        category=ModelCategory.MEDIUM,
        description="GPT OSS 20B",
        recommended_for=["strategist", "critic"],
        avg_response_time_s=100.0,
    ),
    # Llama 3.3 70B - Multi-GPU optimisé
    "llama3.3:70b-instruct-q4_K_M": ModelInfo(
        name="llama3.3:70b-instruct-q4_K_M",
        category=ModelCategory.HEAVY,
        description="Llama 3.3 70B Instruct Q4 - Multi-GPU, raisonnement avancé",
        recommended_for=["critic", "validator"],
        avg_response_time_s=300.0,
    ),
    "llama3.3-70b-optimized": ModelInfo(
        name="llama3.3-70b-optimized",
        category=ModelCategory.HEAVY,
        description="Llama 3.3 70B Optimisé - Config multi-GPU personnalisée",
        recommended_for=["critic", "validator"],
        avg_response_time_s=300.0,
    ),
}


@dataclass
class RoleModelAssignment:
    """Configuration des modèles pour un rôle."""

    role: str
    models: List[str] = field(default_factory=list)
    allow_heavy_after_iteration: int = 3  # N'autoriser les modèles lourds qu'après N itérations
    prefer_specialized: bool = True  # Préférer les modèles spécialisés pour ce rôle

    def get_available_models(
        self,
        iteration: int = 1,
        allow_heavy: bool = False,
        installed_models: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Retourne les modèles disponibles pour cette itération.

        Args:
            iteration: Numéro d'itération actuel
            allow_heavy: Forcer l'autorisation des modèles lourds
            installed_models: Set des modèles installés (pour filtrage)

        Returns:
            Liste des modèles utilisables
        """
        available = []

        for model_name in self.models:
            # Vérifier si installé
            if installed_models and model_name not in installed_models:
                continue

            # Vérifier la catégorie
            model_info = KNOWN_MODELS.get(model_name)
            if model_info and model_info.category == ModelCategory.HEAVY:
                # Modèles lourds : vérifier les conditions
                if not allow_heavy and iteration < self.allow_heavy_after_iteration:
                    continue

            available.append(model_name)

        return available


@dataclass
class RoleModelConfig:
    """
    Configuration complète des modèles par rôle.

    Permet d'assigner plusieurs modèles à chaque rôle avec sélection
    aléatoire ou basée sur des critères.
    """

    # Configuration par rôle
    analyst: RoleModelAssignment = field(default_factory=lambda: RoleModelAssignment(
        role="analyst",
        models=["deepseek-r1:8b", "mistral:7b-instruct", "martain7r/finance-llama-8b:q4_k_m", "gemma3:12b"],
        allow_heavy_after_iteration=5,
    ))

    strategist: RoleModelAssignment = field(default_factory=lambda: RoleModelAssignment(
        role="strategist",
        models=["deepseek-r1:8b", "gemma3:12b", "deepseek-r1-distill:14b", "mistral:22b"],
        allow_heavy_after_iteration=3,
    ))

    critic: RoleModelAssignment = field(default_factory=lambda: RoleModelAssignment(
        role="critic",
        models=["deepseek-r1-distill:14b", "mistral:22b", "gemma3:27b", "deepseek-r1:32b", "qwq:32b", "llama3.3-70b-optimized"],
        allow_heavy_after_iteration=2,
    ))

    validator: RoleModelAssignment = field(default_factory=lambda: RoleModelAssignment(
        role="validator",
        models=["deepseek-r1-distill:14b", "gemma3:27b", "deepseek-r1:32b", "qwq:32b", "llama3.3-70b-optimized"],
        allow_heavy_after_iteration=3,
    ))

    # Cache des modèles installés
    _installed_models: Optional[Set[str]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialise le cache des modèles installés."""
        self._refresh_installed_models()

    def _refresh_installed_models(self) -> Set[str]:
        """Rafraîchit la liste des modèles Ollama installés."""
        data = _fetch_ollama_tags_with_retries()
        if not data:
            self._installed_models = set()
            return self._installed_models

        models = data.get("models", [])
        # Normaliser les noms (enlever tags :latest, :70b, etc.) pour compatibilité avec KNOWN_MODELS
        normalized_names = set()
        for m in models:
            name = m.get("name", "")
            if name:
                # Ajouter nom complet ET nom sans tag pour compatibilité
                normalized_names.add(name)
                if ":" in name:
                    normalized_names.add(name.split(":")[0])

        self._installed_models = normalized_names
        logger.debug("Modeles installes: %s", len(self._installed_models))

        return self._installed_models

    def get_installed_models(self) -> Set[str]:
        """Retourne les modèles installés (avec cache)."""
        if self._installed_models is None:
            self._refresh_installed_models()
        return self._installed_models or set()

    def get_role_assignment(self, role: str) -> RoleModelAssignment:
        """Retourne l'assignment pour un rôle."""
        role = role.lower()
        if role == "analyst":
            return self.analyst
        elif role == "strategist":
            return self.strategist
        elif role == "critic":
            return self.critic
        elif role == "validator":
            return self.validator
        else:
            # Défaut: utiliser analyst
            logger.warning(f"Rôle inconnu: {role}, utilisation de analyst")
            return self.analyst

    def get_model(
        self,
        role: str,
        iteration: int = 1,
        allow_heavy: bool = False,
        random_selection: bool = True,
    ) -> Optional[str]:
        """
        Obtient un modèle pour un rôle donné.

        Args:
            role: Nom du rôle (analyst, strategist, critic, validator)
            iteration: Numéro d'itération actuel
            allow_heavy: Forcer l'autorisation des modèles lourds
            random_selection: Si True, sélection aléatoire parmi les modèles disponibles

        Returns:
            Nom du modèle ou None si aucun disponible
        """
        assignment = self.get_role_assignment(role)
        installed = self.get_installed_models()

        # Niveau 1: intersection config role ∩ installed
        available = assignment.get_available_models(
            iteration=iteration,
            allow_heavy=allow_heavy,
            installed_models=installed,
        )
        if available:
            return random.choice(available) if (random_selection and len(available) > 1) else available[0]

        # Niveau 2: fallback sur la config du role, meme si Ollama est down / liste vide
        fallback_cfg = assignment.get_available_models(
            iteration=iteration,
            allow_heavy=allow_heavy,
            installed_models=None,
        )
        if fallback_cfg:
            return (
                random.choice(fallback_cfg)
                if (random_selection and len(fallback_cfg) > 1)
                else fallback_cfg[0]
            )

        # Niveau 3: n'importe quel modele installe
        if installed:
            return next(iter(installed))

        logger.warning("Aucun modele disponible pour %s (iteration=%s)", role, iteration)
        return None

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Retourne les infos d'un modèle."""
        return KNOWN_MODELS.get(model_name)

    def set_role_models(self, role: str, models: List[str]) -> None:
        """Configure les modèles pour un rôle."""
        assignment = self.get_role_assignment(role)
        assignment.models = models
        logger.info(f"Modèles pour {role}: {models}")

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la configuration."""
        return {
            "analyst": {
                "models": self.analyst.models,
                "allow_heavy_after_iteration": self.analyst.allow_heavy_after_iteration,
            },
            "strategist": {
                "models": self.strategist.models,
                "allow_heavy_after_iteration": self.strategist.allow_heavy_after_iteration,
            },
            "critic": {
                "models": self.critic.models,
                "allow_heavy_after_iteration": self.critic.allow_heavy_after_iteration,
            },
            "validator": {
                "models": self.validator.models,
                "allow_heavy_after_iteration": self.validator.allow_heavy_after_iteration,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoleModelConfig:
        """Désérialise la configuration."""
        config = cls()

        for role in ["analyst", "strategist", "critic", "validator"]:
            if role in data:
                role_data = data[role]
                assignment = config.get_role_assignment(role)
                assignment.models = role_data.get("models", assignment.models)
                assignment.allow_heavy_after_iteration = role_data.get(
                    "allow_heavy_after_iteration",
                    assignment.allow_heavy_after_iteration
                )

        return config


def list_available_models() -> List[ModelInfo]:
    """Liste tous les modèles installés avec leurs infos."""
    data = _fetch_ollama_tags_with_retries()
    if not data:
        # Fallback: retourner les modèles connus (utile pour l'UI quand Ollama est lent)
        return list(KNOWN_MODELS.values())

    models = data.get("models", [])

    result: List[ModelInfo] = []
    for m in models:
        name = m.get("name", "")
        if not name:
            continue
        if name in KNOWN_MODELS:
            result.append(KNOWN_MODELS[name])
            continue

        size_gb = m.get("size", 0) / (1024**3)
        if size_gb < 6:
            category = ModelCategory.LIGHT
        elif size_gb < 15:
            category = ModelCategory.MEDIUM
        else:
            category = ModelCategory.HEAVY

        result.append(
            ModelInfo(
                name=name,
                category=category,
                description=f"Modele {name} ({size_gb:.1f} GB)",
                recommended_for=["analyst", "strategist"],
            )
        )

    return result


def get_models_by_category(category: ModelCategory) -> List[ModelInfo]:
    """Retourne les modèles installés d'une catégorie."""
    all_models = list_available_models()
    return [m for m in all_models if m.category == category]


# Singleton pour la configuration globale
_global_config: Optional[RoleModelConfig] = None


def get_global_model_config() -> RoleModelConfig:
    """Retourne la configuration globale (singleton)."""
    global _global_config
    if _global_config is None:
        _global_config = RoleModelConfig()
    return _global_config


def set_global_model_config(config: RoleModelConfig) -> None:
    """Définit la configuration globale."""
    global _global_config
    _global_config = config


__all__ = [
    "ModelCategory",
    "ModelInfo",
    "RoleModelAssignment",
    "RoleModelConfig",
    "KNOWN_MODELS",
    "list_available_models",
    "get_models_by_category",
    "get_global_model_config",
    "set_global_model_config",
]
