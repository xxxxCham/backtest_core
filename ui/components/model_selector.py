"""
Backtest Core - LLM Model Selector
===================================

Helpers pour connecter l'interface Streamlit aux modèles Ollama.

Features:
- Récupère dynamiquement la liste des modèles disponibles via Ollama
- Fournit un fallback cohérent quand Ollama n'est pas accessible
- Centralise la logique pour que toutes les pages utilisent la même source

Usage:
    >>> from ui.components.model_selector import get_available_models_for_ui
    >>>
    >>> # Dans Streamlit
    >>> models = get_available_models_for_ui()
    >>> selected_model = st.selectbox("Modèle LLM", models)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from agents.ollama_manager import list_ollama_models
from utils.log import get_logger

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


def get_available_models_for_ui(
    preferred_order: Sequence[str] | None = None,
    fallback: Sequence[str] | None = None,
) -> List[str]:
    """
    Retourne la liste des modèles LLM à proposer dans l'UI.

    - Si Ollama répond, on retourne exactement les modèles installés.
    - Si Ollama est inaccessible ou ne retourne rien, on utilise une liste fallback.

    Args:
        preferred_order: Ordre conseillé pour le tri (facultatif).
        fallback: Fallback explicite (sinon FALLBACK_LLM_MODELS).

    Returns:
        list[str]: Noms de modèles Ollama.

    Example:
        >>> models = get_available_models_for_ui(
        ...     preferred_order=RECOMMENDED_FOR_STRATEGY
        ... )
        >>> st.selectbox("Modèle", models)
    """
    installed = list_ollama_models()

    if installed:
        if preferred_order:
            return _sort_with_preferred(installed, preferred_order)
        # Par défaut: tri alphabétique des modèles installés
        return sorted(set(installed))

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

    Args:
        model_name: Nom du modèle

    Returns:
        dict: Informations du modèle (size, description, etc.)
    """
    # Mapping des tailles approximatives (GB)
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

    # Descriptions
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
    "get_available_models_for_ui",
    "get_model_info",
    "render_model_selector",
]
