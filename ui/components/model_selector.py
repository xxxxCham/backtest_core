"""
Module-ID: ui.components.model_selector

Purpose: Sélecteur modèles LLM - query Ollama, fallback list, recommendations par rôle.

Role in pipeline: configuration

Key components: get_available_models_for_ui(), FALLBACK_LLM_MODELS, role-based recommendations

Inputs: Ollama endpoint (optionnel), role (Analyst/Strategist/Critic/Validator)

Outputs: Model list [str] ordenée (recommandés en premier), fallback si Ollama down

Dependencies: agents.ollama_manager (optionnel), log

Conventions: Fallback list actualisée (Dec 2025); Ollama préféré; timeout 2s.

Read-if: Modification fallback list ou role mappings.

Skip-if: Vous appelez get_available_models_for_ui().
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    from agents.ollama_manager import list_ollama_models
except ImportError:
    def list_ollama_models() -> List[str]:
        return []
from utils.log import get_logger
from utils.model_loader import get_model_info_for_ui, get_ollama_model_names

logger = get_logger(__name__)

# Liste de modèles recommandés utilisée comme fallback
FALLBACK_LLM_MODELS: List[str] = [
    # Modèles recommandés pour trading (Dec 2025)
    "deepseek-r1:70b",              # 34 GB - Le plus puissant
    "deepseek-r1:32b",              # 19 GB - Excellent rapport qualité/prix
    "qwq:32b",                      # 23 GB - Très bon pour raisonnement
    "qwen2.5:32b",                  # 19 GB - Alternative solide
    "mistral:22b",                  # 13 GB - Équilibré
    "gemma3:27b",                   # 17 GB - Bon pour analyse
    "deepseek-r1-distill:14b",      # 9 GB - Équilibré
    "gemma3:12b",                   # 8 GB - Rapide
    "deepseek-r1:8b",               # 5 GB - Builder rapide
    "mistral:7b-instruct",          # 4 GB - Ultra rapide
    "llama3.2",                     # 2 GB - Léger
    "phi3",                         # 2 GB - Léger alternatif
]

# Modèles recommandés pour différentes tâches
RECOMMENDED_FOR_ANALYSIS = ["deepseek-r1:32b", "qwq:32b", "qwen2.5:32b"]
RECOMMENDED_FOR_STRATEGY = ["deepseek-r1:70b", "deepseek-r1:32b", "qwq:32b"]
RECOMMENDED_FOR_CRITICISM = ["mistral:22b", "gemma3:27b", "qwen2.5:32b"]
RECOMMENDED_FOR_FAST = ["deepseek-r1:8b", "mistral:7b-instruct", "llama3.2"]

# Configuration optimale par rôle (basée sur benchmarks Dec 2025)
OPTIMAL_CONFIG_BY_ROLE = {
    "analyst": ["qwen2.5:14b"],
    "strategist": ["gemma3:27b"],
    "critic": ["llama3.3-70b-optimized"],
    "validator": ["llama3.3-70b-optimized"],
}

# Fallback si le modèle optimal n'est pas installé
OPTIMAL_CONFIG_FALLBACK = {
    "analyst": ["deepseek-r1:8b", "gemma3:12b"],
    "strategist": ["gemma3:27b", "mistral:22b"],
    "critic": ["deepseek-r1:32b", "qwq:32b"],
    "validator": ["deepseek-r1:32b", "qwq:32b"],
}


def _sort_with_preferred(
    models: Iterable[str], preferred_order: Sequence[str]
) -> List[str]:
    """
    Trie les modèles en mettant ceux de preferred_order en premier.

    Args:
        models: Liste des modèles à trier
        preferred_order: Ordre préféré

    Returns:
        Liste triée de modèles
    """
    preferred_index = {name: i for i, name in enumerate(preferred_order)}

    def sort_key(name: str) -> tuple[int, int | str]:
        if name in preferred_index:
            return 0, preferred_index[name]
        return 1, name

    unique = sorted(set(models), key=sort_key)
    return unique


def _get_library_models() -> List[str]:
    try:
        return get_ollama_model_names()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Erreur lecture models.json pour la liste UI: %s", exc)
        return []


def _normalize_model_name(name: str) -> str:
    if name.endswith(":latest"):
        return name.rsplit(":", 1)[0]
    return name


def get_available_models_for_ui(
    preferred_order: Sequence[str] | None = None,
    fallback: Sequence[str] | None = None,
) -> List[str]:
    """
    Retourne la liste des modèles LLM à proposer dans l'UI.

    - Si Ollama répond, on fusionne les modèles installés avec models.json.
    - Si Ollama est inaccessible, on utilise models.json si dispo.
    - Sinon, fallback sur une liste minimale.

    Args:
        preferred_order: Ordre conseillé pour le tri (facultatif).
        fallback: Fallback explicite (sinon FALLBACK_LLM_MODELS).

    Returns:
        list[str]: Noms de modèles Ollama (installés + bibliothèque).

    Example:
        >>> models = get_available_models_for_ui(
        ...     preferred_order=RECOMMENDED_FOR_STRATEGY
        ... )
        >>> st.selectbox("Modèle", models)
    """
    installed = [_normalize_model_name(n) for n in list_ollama_models() if n]
    library_models = _get_library_models()
    available = sorted(set(installed) | set(library_models))

    if available:
        if preferred_order:
            return _sort_with_preferred(available, preferred_order)
        # Par défaut: tri alphabétique des modèles installés + bibliothèque
        return available

    # Ollama inaccessible ou aucun modèle retourné: fallback
    models = list(fallback or FALLBACK_LLM_MODELS)
    logger.warning(
        "⚠️ Ollama ne renvoie aucun modèle, utilisation du fallback UI: %s",
        models[:3]
    )
    return models


def get_model_info(model_name: str) -> dict:
    """
    Retourne des informations sur un modèle.

    Priorité: models.json > hardcoded fallback

    Args:
        model_name: Nom du modèle

    Returns:
        dict: Informations du modèle (size, description, etc.)
    """
    # 1. Chercher dans models.json d'abord
    try:
        info = get_model_info_for_ui(model_name)
        if info and info.get("size_gb") != "?":
            return {
                "name": info.get("name", model_name),
                "size_gb": info["size_gb"],
                "description": info.get("description", ""),
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Erreur lecture models.json pour %s: %s", model_name, exc)

    # 2. Fallback hardcodé pour compatibilité
    model_sizes = {
        "deepseek-r1:70b": 34,
        "deepseek-r1:32b": 19,
        "qwq:32b": 23,
        "qwen2.5:32b": 19,
        "mistral:22b": 13,
        "gemma3:27b": 17,
        "deepseek-r1-distill:14b": 9,
        "gemma3:12b": 8,
        "deepseek-r1:8b": 5,
        "mistral:7b-instruct": 4,
        "llama3.2": 2,
        "phi3": 2,
    }

    model_descriptions = {
        "deepseek-r1:70b": "Le plus puissant - Excellent pour stratégies complexes",
        "deepseek-r1:32b": "Optimal - Meilleur rapport qualité/prix",
        "qwq:32b": "Excellent raisonnement - Idéal pour analyse",
        "qwen2.5:32b": "Alternative solide - Polyvalent",
        "mistral:22b": "Équilibré - Bon pour critique",
        "gemma3:27b": "Bon pour analyse - Rapide",
        "deepseek-r1-distill:14b": "Équilibré - Compact",
        "gemma3:12b": "Rapide - Léger",
        "deepseek-r1:8b": "Ultra rapide - Tests rapides",
        "mistral:7b-instruct": "Ultra rapide - Très léger",
        "llama3.2": "Léger - Pour tests",
        "phi3": "Léger - Pour tests",
    }

    return {
        "name": model_name,
        "size_gb": model_sizes.get(model_name, "?"),
        "description": model_descriptions.get(model_name, "Modèle LLM"),
    }


def get_optimal_config_for_role(
    role: str,
    available_models: List[str],
) -> List[str]:
    """
    Retourne la configuration optimale pour un rôle donné.

    Si le modèle optimal n'est pas installé, utilise le fallback.

    Args:
        role: Rôle de l'agent (analyst, strategist, critic, validator)
        available_models: Liste des modèles disponibles

    Returns:
        Liste des modèles optimaux pour ce rôle (filtrés par disponibilité)

    Example:
        >>> optimal = get_optimal_config_for_role("critic", installed_models)
        >>> # Returns: ["llama3.3-70b-optimized"] if installed, else fallback
    """
    # Modèle optimal primaire
    optimal_primary = OPTIMAL_CONFIG_BY_ROLE.get(role, [])

    # Vérifier si le modèle optimal est disponible
    available_set = set(available_models)
    optimal_available = [m for m in optimal_primary if m in available_set]

    if optimal_available:
        return optimal_available

    # Fallback si le modèle optimal n'est pas installé
    fallback_options = OPTIMAL_CONFIG_FALLBACK.get(role, [])
    fallback_available = [m for m in fallback_options if m in available_set]

    if fallback_available:
        return fallback_available[:1]  # Premier fallback disponible

    # Dernier recours: premiers modèles disponibles
    return available_models[:1] if available_models else []


def render_model_selector(
    label: str = "Modèle LLM",
    key: str = "llm_model",
    preferred_order: Sequence[str] | None = None,
    help_text: str | None = None,
) -> str:
    """
    Rendu d'un sélecteur de modèle Streamlit avec informations.

    Args:
        label: Label du selectbox
        key: Clé du state Streamlit
        preferred_order: Ordre préféré des modèles
        help_text: Texte d'aide optionnel

    Returns:
        str: Nom du modèle sélectionné

    Example:
        >>> import streamlit as st
        >>> model = render_model_selector(
        ...     label="Modèle Analyst",
        ...     key="analyst_model",
        ...     preferred_order=RECOMMENDED_FOR_ANALYSIS
        ... )
    """
    import streamlit as st

    models = get_available_models_for_ui(preferred_order=preferred_order)

    if not help_text:
        help_text = "Sélectionnez un modèle LLM Ollama pour l'optimisation"

    selected = st.selectbox(
        label,
        models,
        key=key,
        help=help_text,
    )

    # Afficher les infos du modèle sélectionné
    if selected:
        info = get_model_info(selected)
        size = info["size_gb"]
        desc = info["description"]
        st.caption(f"📦 ~{size} GB | {desc}")

    return selected


__all__ = [
    "FALLBACK_LLM_MODELS",
    "RECOMMENDED_FOR_ANALYSIS",
    "RECOMMENDED_FOR_STRATEGY",
    "RECOMMENDED_FOR_CRITICISM",
    "RECOMMENDED_FOR_FAST",
    "OPTIMAL_CONFIG_BY_ROLE",
    "OPTIMAL_CONFIG_FALLBACK",
    "get_available_models_for_ui",
    "get_model_info",
    "get_optimal_config_for_role",
    "render_model_selector",
]
