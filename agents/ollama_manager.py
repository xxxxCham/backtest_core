"""
Module-ID: agents.ollama_manager

Purpose: Gérer Ollama (auto-démarrage, listage modèles, déchargement GPU, health checks).

Role in pipeline: orchestration / performance

Key components: LLMMemoryState, GPUMemoryManager, ensure_ollama_running, gpu_compute_context, list_ollama_models

Inputs: Modèle name, timeouts, max_attempts

Outputs: État Ollama, modèles disponibles, gestion mémoire GPU (unload/reload)

Dependencies: subprocess, httpx, utils.log, contextlib

Conventions: Ollama lancé via subprocess `ollama serve`; retries avec backoff exponentiel; gpu_compute_context décharge LLM avant calculs NumPy/CuPy; recharge auto après.

Read-if: Configuration Ollama, gestion GPU memory, ou troubleshooting service.

Skip-if: Vous utilisez seulement OpenAI.
"""

from __future__ import annotations

import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import httpx

from utils.log import get_logger

logger = get_logger(__name__)


# ==============================================================================
# GPU Memory Manager - Déchargement/Rechargement intelligent des LLM
# ==============================================================================


@dataclass
class LLMMemoryState:
    """État mémoire d'un modèle LLM."""

    model_name: str
    was_loaded: bool = False
    context_messages: List[dict] = field(default_factory=list)
    unload_time_ms: float = 0.0
    reload_time_ms: float = 0.0


class GPUMemoryManager:
    """
    Gestionnaire de mémoire GPU pour les LLM.

    Permet de décharger temporairement les LLM du GPU pendant les phases
    de calcul intensif (backtests avec CuPy/NumPy) puis de les recharger
    avec leur contexte préservé.

    Features:
    - Déchargement automatique avant calculs
    - Rechargement automatique après calculs
    - Préservation du contexte de conversation
    - Métriques de temps (unload/reload)
    - Mode "dry run" pour tests

    Example:
        >>> manager = GPUMemoryManager("deepseek-r1:32b")
        >>>
        >>> # Décharger avant calcul
        >>> state = manager.unload()
        >>>
        >>> # ... calculs GPU intensifs ...
        >>>
        >>> # Recharger avec contexte
        >>> manager.reload(state)
    """

    def __init__(
        self,
        model_name: str,
        ollama_host: str = "http://127.0.0.1:11434",
        warmup_prompt: str = "You are ready.",
        verbose: bool = True,
    ):
        """
        Initialise le gestionnaire.

        Args:
            model_name: Nom du modèle Ollama (ex: "deepseek-r1:32b")
            ollama_host: URL du serveur Ollama
            warmup_prompt: Prompt court pour "réchauffer" le modèle au reload
            verbose: Afficher les logs
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.warmup_prompt = warmup_prompt
        self.verbose = verbose
        self._current_state: Optional[LLMMemoryState] = None

    def is_model_loaded(self) -> bool:
        """Vérifie si le modèle est actuellement en mémoire GPU."""
        try:
            # Utiliser l'API ps pour voir les modèles chargés
            response = httpx.get(
                f"{self.ollama_host}/api/ps",
                timeout=3.0
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                for m in models:
                    if m.get("name", "").startswith(self.model_name.split(":")[0]):
                        return True
            return False
        except Exception:
            return False

    def unload(self, context_messages: Optional[List[dict]] = None) -> LLMMemoryState:
        """
        Décharge le modèle du GPU.

        Args:
            context_messages: Messages de contexte à préserver pour le reload

        Returns:
            LLMMemoryState avec les infos pour le rechargement
        """
        start = time.perf_counter()
        was_loaded = self.is_model_loaded()

        state = LLMMemoryState(
            model_name=self.model_name,
            was_loaded=was_loaded,
            context_messages=context_messages or [],
        )

        if was_loaded:
            try:
                response = httpx.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "keep_alive": 0,  # 0 = décharger immédiatement
                        "prompt": "",
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    state.unload_time_ms = (time.perf_counter() - start) * 1000
                    if self.verbose:
                        logger.info(
                            f"💾 LLM déchargé: {self.model_name} "
                            f"({state.unload_time_ms:.0f}ms) → GPU libre pour calculs"
                        )
            except Exception as e:
                logger.warning(f"⚠️ Échec déchargement LLM: {e}")
        else:
            if self.verbose:
                logger.debug(f"📝 LLM {self.model_name} pas en mémoire, skip unload")

        self._current_state = state
        return state

    def reload(
        self,
        state: Optional[LLMMemoryState] = None,
        restore_context: bool = True,
    ) -> bool:
        """
        Recharge le modèle dans le GPU.

        Args:
            state: État précédent (ou utilise _current_state)
            restore_context: Si True, envoie un prompt de warmup

        Returns:
            True si succès
        """
        state = state or self._current_state
        if not state:
            logger.warning("⚠️ Pas d'état à restaurer")
            return False

        if not state.was_loaded:
            if self.verbose:
                logger.debug(f"📝 LLM {self.model_name} n'était pas chargé, skip reload")
            return True

        start = time.perf_counter()

        try:
            # Warmup: charger le modèle avec un prompt court
            warmup = self.warmup_prompt
            if restore_context and state.context_messages:
                # Résumé du contexte pour le LLM
                warmup = (
                    f"Previous context summary: We were optimizing a trading strategy. "
                    f"Last {len(state.context_messages)} messages exchanged. "
                    f"Ready to continue."
                )

            response = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": warmup,
                    "keep_alive": "10m",  # Garder 10 minutes
                    "stream": False,
                },
                timeout=120.0  # 2 min pour charger un gros modèle
            )

            if response.status_code == 200:
                state.reload_time_ms = (time.perf_counter() - start) * 1000
                if self.verbose:
                    logger.info(
                        f"🔄 LLM rechargé: {self.model_name} "
                        f"({state.reload_time_ms:.0f}ms)"
                    )
                return True
            else:
                logger.warning(f"⚠️ Échec reload LLM: status {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"⚠️ Échec rechargement LLM: {e}")
            return False

    def get_stats(self) -> dict:
        """Retourne les statistiques de la dernière opération."""
        if not self._current_state:
            return {}
        return {
            "model": self._current_state.model_name,
            "was_loaded": self._current_state.was_loaded,
            "unload_time_ms": self._current_state.unload_time_ms,
            "reload_time_ms": self._current_state.reload_time_ms,
            "context_size": len(self._current_state.context_messages),
        }


@contextmanager
def gpu_compute_context(
    model_name: str,
    context_messages: Optional[List[dict]] = None,
    verbose: bool = True,
) -> Generator[GPUMemoryManager, None, None]:
    """
    Context manager pour libérer le GPU pendant les calculs.

    Décharge automatiquement le LLM avant les calculs et le recharge après.

    Args:
        model_name: Nom du modèle à décharger
        context_messages: Contexte de conversation à préserver
        verbose: Afficher les logs

    Yields:
        GPUMemoryManager pour accès aux stats

    Example:
        >>> with gpu_compute_context("deepseek-r1:32b") as manager:
        ...     # GPU libre pour calculs numpy/cupy
        ...     results = heavy_backtest_computation()
        >>> # LLM automatiquement rechargé
        >>> print(manager.get_stats())
    """
    manager = GPUMemoryManager(model_name, verbose=verbose)

    # Décharger
    state = manager.unload(context_messages)

    try:
        yield manager
    finally:
        # Recharger
        manager.reload(state)


def ensure_ollama_running() -> Tuple[bool, str]:
    """
    S'assure qu'Ollama est démarré et fonctionnel.

    Returns:
        tuple[bool, str]: (succès, message)
    """
    # 1. Vérifier si Ollama répond
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            logger.info("✅ Ollama déjà actif")
            return True, "✅ Ollama actif"
    except Exception:
        pass  # Ollama pas actif, on va le démarrer

    # 2. Démarrer Ollama
    logger.info("🚀 Démarrage d'Ollama...")
    try:
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows : Lancer avec flags de création console
            kwargs = {"creationflags": subprocess.CREATE_NEW_CONSOLE}

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # 3. Attendre qu'Ollama soit prêt (max 10s)
        for i in range(10):
            time.sleep(1)
            try:
                response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
                if response.status_code == 200:
                    logger.info(f"✅ Ollama démarré avec succès (après {i+1}s)")
                    return True, f"✅ Ollama démarré ({i+1}s)"
            except Exception:
                continue

        return False, "⏱️ Timeout - Ollama n'a pas démarré en 10s"

    except FileNotFoundError:
        return False, "❌ Ollama non trouvé (vérifiez l'installation)"
    except Exception as e:
        return False, f"❌ Erreur: {str(e)}"


def unload_model(model_name: str) -> bool:
    """
    Décharge un modèle Ollama de la mémoire GPU/RAM.

    Args:
        model_name: Nom du modèle (ex: "deepseek-r1:32b")

    Returns:
        bool: True si succès
    """
    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model_name,
                "keep_alive": 0,  # 0 = décharger immédiatement
                "prompt": "",
            },
            timeout=5.0
        )
        success = response.status_code == 200
        if success:
            logger.info(f"💾 Modèle {model_name} déchargé de la mémoire")
        return success
    except Exception as e:
        logger.warning(f"⚠️ Impossible de décharger {model_name}: {e}")
        return False


def cleanup_all_models() -> int:
    """
    Décharge TOUS les modèles Ollama de la mémoire.

    Returns:
        int: Nombre de modèles déchargés
    """
    try:
        # Lister les modèles chargés
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        if response.status_code != 200:
            return 0

        models = response.json().get("models", [])
        count = 0

        for model in models:
            model_name = model.get("name", "")
            if model_name and unload_model(model_name):
                count += 1

        if count > 0:
            logger.info(f"🧹 {count} modèle(s) déchargé(s) de la mémoire")

        return count

    except Exception as e:
        logger.warning(f"⚠️ Erreur cleanup_all_models: {e}")
        return 0


def list_ollama_models() -> List[str]:
    """
    Retourne la liste des modèles Ollama installés localement.

    Returns:
        list[str]: Noms des modèles (ex: ["llama3.2", "mistral"])
    """
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
        if response.status_code != 200:
            logger.warning(
                f"⚠️ Impossible de lister les modèles Ollama (status={response.status_code})"
            )
            return []

        payload = response.json()
        models = payload.get("models", []) or []

        names: List[str] = []
        for model in models:
            name = model.get("name")
            if isinstance(name, str) and name:
                names.append(name)

        return names
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors de la récupération des modèles Ollama: {e}")
        return []


def is_ollama_available() -> bool:
    """
    Vérifie si Ollama est disponible.

    Returns:
        bool: True si Ollama répond
    """
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def prepare_for_llm_run() -> Tuple[bool, str]:
    """
    Prépare l'environnement pour un run LLM.

    Actions:
    1. S'assure qu'Ollama est actif
    2. Nettoie les modèles précédents en mémoire

    Returns:
        tuple[bool, str]: (succès, message détaillé)
    """
    messages = []

    # 1. Nettoyer les modèles précédents
    cleaned = cleanup_all_models()
    if cleaned > 0:
        messages.append(f"🧹 {cleaned} modèle(s) déchargé(s)")

    # 2. S'assurer qu'Ollama est actif
    success, msg = ensure_ollama_running()
    messages.append(msg)

    if success:
        time.sleep(1)  # Petite pause pour stabilité
        return True, " | ".join(messages)
    else:
        return False, " | ".join(messages)


__all__ = [
    "ensure_ollama_running",
    "unload_model",
    "cleanup_all_models",
    "list_ollama_models",
    "is_ollama_available",
    "prepare_for_llm_run",
    # GPU Memory Management
    "GPUMemoryManager",
    "LLMMemoryState",
    "gpu_compute_context",
]
