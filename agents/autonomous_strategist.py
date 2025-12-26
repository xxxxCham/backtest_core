"""
AutonomousStrategist - Agent capable de lancer des backtests et d'itérer.

C'est ici que le LLM apporte sa VRAIE valeur :
- Il formule des hypothèses
- Il lance des backtests pour les tester
- Il analyse les résultats
- Il itère avec de nouvelles idées
- Il décide quand arrêter

Le code déterministe (métriques, seuils) reste dans les modules appropriés.
Le LLM apporte la créativité et le raisonnement.

GPU Memory Optimization:
- Le LLM est déchargé du GPU pendant les calculs de backtest
- Cela libère la VRAM pour les calculs NumPy/CuPy
- Le LLM est rechargé automatiquement après chaque backtest
"""

from __future__ import annotations

import logging

# Import search space statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .backtest_executor import (
    BacktestExecutor,
    BacktestRequest,
    BacktestResult,
)
from .base_agent import AgentContext, AgentResult, AgentRole, BaseAgent
from .llm_client import LLMClient, LLMConfig
from .ollama_manager import GPUMemoryManager

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.parameters import compute_search_space_stats

logger = logging.getLogger(__name__)
# Force WARNING level pour ce module pour voir les décisions critiques
logger.setLevel(logging.WARNING)


@dataclass
class IterationDecision:
    """Décision du LLM après analyse d'un résultat."""

    action: str  # "continue", "accept", "stop", "change_direction"
    confidence: float  # 0-1

    # Prochaine action si "continue"
    next_hypothesis: str = ""
    next_parameters: Dict[str, Any] = field(default_factory=dict)

    # Raison
    reasoning: str = ""

    # Insights accumulés
    insights: List[str] = field(default_factory=list)


@dataclass
class OptimizationSession:
    """Session d'optimisation complète."""

    # Configuration
    strategy_name: str
    initial_params: Dict[str, Any]
    target_metric: str = "sharpe_ratio"

    # Contraintes
    max_iterations: int = 20
    min_improvement_threshold: float = 0.01
    max_time_seconds: float = 3600.0  # 1 heure max

    # État
    current_iteration: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    # Résultats
    best_result: Optional[BacktestResult] = None
    all_results: List[BacktestResult] = field(default_factory=list)
    decisions: List[IterationDecision] = field(default_factory=list)

    # Status final
    final_status: str = ""  # "success", "max_iterations", "timeout", "no_improvement"
    final_reasoning: str = ""


class AutonomousStrategist(BaseAgent):
    """
    Agent Strategist Autonome capable de lancer des backtests.

    Workflow:
    1. Analyser la configuration initiale
    2. Formuler une hypothèse d'amélioration
    3. Lancer un backtest
    4. Analyser les résultats
    5. Décider: continuer, accepter, ou changer de direction
    6. Répéter jusqu'à convergence ou limite

    GPU Memory Optimization:
    - Le LLM est déchargé du GPU avant chaque backtest
    - La VRAM est ainsi disponible pour les calculs NumPy/CuPy
    - Le LLM est rechargé après le backtest pour l'analyse

    Example:
        >>> strategist = AutonomousStrategist(llm_client)
        >>> session = strategist.optimize(
        ...     executor=executor,
        ...     initial_params={"fast": 10, "slow": 21},
        ...     param_bounds={"fast": (5, 20), "slow": (15, 50)},
        ...     max_iterations=10
        ... )
        >>> print(f"Best Sharpe: {session.best_result.sharpe_ratio}")
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.STRATEGIST

    @property
    def system_prompt(self) -> str:
        return """You are an autonomous trading strategy optimizer with the ability to run backtests.

Your process:
1. ANALYZE current results and history
2. FORMULATE a hypothesis about what might improve performance
3. PROPOSE specific parameters to test
4. After seeing results, DECIDE whether to continue, accept, or change direction

Key principles:
- Each experiment should test ONE clear hypothesis
- Learn from failures - don't repeat similar mistakes
- Balance exploration (trying new things) vs exploitation (refining what works)
- Watch for overfitting - prefer robust solutions over peak performance
- Consider parameter interactions (e.g., fast/slow periods should maintain ratio)

You will receive experiment history and must respond with a decision.

Response format (JSON):
{
    "action": "continue|accept|stop|change_direction",
    "confidence": 0.0 to 1.0,
    "reasoning": "Why this decision",
    "next_hypothesis": "What you want to test next (if continuing)",
    "next_parameters": {"param": value},
    "insights": ["insight1", "insight2"]
}

Actions:
- "continue": Run another backtest with next_parameters
- "accept": Accept current best as final solution
- "stop": Stop due to diminishing returns or constraints
- "change_direction": Abandon current approach, try something different"""

    def __init__(
        self,
        llm_client: LLMClient,
        verbose: bool = False,
        on_progress: Optional[Callable[[int, BacktestResult], None]] = None,
        unload_llm_during_backtest: Optional[bool] = None,
        orchestration_logger: Optional[Any] = None,
    ):
        """
        Initialise le strategist autonome.

        Args:
            llm_client: Client LLM
            verbose: Afficher les logs détaillés
            on_progress: Callback appelé après chaque itération
            unload_llm_during_backtest: Si True, décharge le LLM du GPU pendant
                les calculs de backtest. Si None, utilise UNLOAD_LLM_DURING_BACKTEST
                env var (default: False pour CPU-only compatibility)
            orchestration_logger: Logger pour enregistrer les actions d'orchestration
        """
        import os
        super().__init__(llm_client)
        self.verbose = verbose
        self.on_progress = on_progress
        self.orchestration_logger = orchestration_logger

        # Lire depuis env var si non spécifié (default False pour CPU-only)
        if unload_llm_during_backtest is None:
            env_val = os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'False')
            self.unload_llm_during_backtest = env_val.lower() in ('true', '1', 'yes')
        else:
            self.unload_llm_during_backtest = unload_llm_during_backtest

        # GPU Memory Manager - initialisé avec le modèle du client
        self._gpu_manager: Optional[GPUMemoryManager] = None
        self._conversation_context: List[dict] = []

    def _get_model_name(self) -> str:
        """Récupère le nom du modèle depuis le client LLM."""
        if hasattr(self.llm, 'config'):
            return self.llm.config.model
        return "unknown"

    def _ensure_gpu_manager(self) -> GPUMemoryManager:
        """Crée ou retourne le GPU manager."""
        if self._gpu_manager is None:
            self._gpu_manager = GPUMemoryManager(
                model_name=self._get_model_name(),
                verbose=self.verbose,
            )
        return self._gpu_manager

    def _run_backtest_with_gpu_optimization(
        self,
        executor: BacktestExecutor,
        request: BacktestRequest,
    ) -> BacktestResult:
        """
        Exécute un backtest avec optimisation mémoire GPU.

        1. Décharge le LLM du GPU
        2. Exécute le backtest (GPU libre)
        3. Recharge le LLM

        Args:
            executor: BacktestExecutor
            request: Requête de backtest

        Returns:
            Résultat du backtest
        """
        if not self.unload_llm_during_backtest:
            # Mode sans optimisation GPU
            return executor.run(request)

        manager = self._ensure_gpu_manager()

        # Décharger le LLM → GPU libre pour calculs
        state = manager.unload(self._conversation_context)

        try:
            # Calculs GPU intensifs (NumPy/CuPy)
            result = executor.run(request)
        finally:
            # Recharger le LLM (toujours, même en cas d'erreur)
            manager.reload(state)

        # Stats pour debug
        if self.verbose:
            stats = manager.get_stats()
            if stats.get("was_loaded"):
                logger.debug(
                    f"⏱️ GPU swap: unload={stats['unload_time_ms']:.0f}ms, "
                    f"reload={stats['reload_time_ms']:.0f}ms"
                )

        return result

    def optimize(
        self,
        executor: BacktestExecutor,
        initial_params: Dict[str, Any],
        param_bounds: Dict[str, tuple],
        max_iterations: int = 20,
        target_metric: str = "sharpe_ratio",
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.30,
        check_pause_callback: Optional[callable] = None,
    ) -> OptimizationSession:
        """
        Lance une session d'optimisation autonome.

        Args:
            executor: BacktestExecutor configuré
            initial_params: Paramètres de départ
            param_bounds: {param_name: (min, max)} pour chaque paramètre
            max_iterations: Maximum d'itérations
            target_metric: Métrique à optimiser
            min_sharpe: Sharpe minimum acceptable
            max_drawdown: Drawdown maximum acceptable
            check_pause_callback: Callback appelé à chaque itération qui retourne (is_paused, should_stop)

        Returns:
            OptimizationSession avec tous les résultats
        """
        session = OptimizationSession(
            strategy_name=executor.strategy_name,
            initial_params=initial_params,
            target_metric=target_metric,
            max_iterations=max_iterations,
        )

        logger.info(
            f"Démarrage optimisation: {session.strategy_name} | "
            f"max_iter={max_iterations}"
        )

        # Logger d'orchestration: début de l'optimisation
        if self.orchestration_logger:
            self.orchestration_logger.log_analysis_start(
                agent="AutonomousStrategist",
                details={
                    "strategy": session.strategy_name,
                    "initial_params": initial_params,
                }
            )

        # 1. Backtest initial
        initial_request = BacktestRequest(
            requested_by=self.role.value,
            hypothesis="Baseline: testing initial configuration",
            parameters=initial_params,
        )

        # Logger: lancement du backtest baseline
        if self.orchestration_logger:
            self.orchestration_logger.log_backtest_launch(
                agent="AutonomousStrategist",
                params=initial_params,
                combination_id=0,
                total_combinations=max_iterations,
            )

        baseline_result = self._run_backtest_with_gpu_optimization(executor, initial_request)
        session.all_results.append(baseline_result)
        session.best_result = baseline_result

        # Logger: résultat du backtest baseline
        if self.orchestration_logger:
            self.orchestration_logger.log_backtest_complete(
                agent="AutonomousStrategist",
                params=initial_params,
                results={
                    "pnl": baseline_result.total_return,
                    "sharpe": baseline_result.sharpe_ratio,
                    "return": baseline_result.total_return,
                },
                combination_id=0,
            )

        if self.on_progress:
            self.on_progress(0, baseline_result)

        logger.info(
            f"Baseline: Sharpe={baseline_result.sharpe_ratio:.3f}, "
            f"Return={baseline_result.total_return:.2%}"
        )

        # 2. Boucle d'itération
        for iteration in range(1, max_iterations + 1):
            session.current_iteration = iteration

            # Vérifier pause/stop si callback fourni
            if check_pause_callback:
                import time
                is_paused, should_stop = check_pause_callback()

                # Si stop demandé, sortir immédiatement
                if should_stop:
                    session.final_reasoning = "Arrêt demandé par l'utilisateur"
                    logger.info("Arrêt demandé - Fin de l'optimisation")
                    break

                # Si pause demandée, attendre
                while is_paused:
                    time.sleep(0.5)
                    is_paused, should_stop = check_pause_callback()
                    if should_stop:
                        session.final_reasoning = "Arrêt demandé par l'utilisateur"
                        logger.info("Arrêt demandé pendant la pause - Fin de l'optimisation")
                        break

                # Sortir de la boucle externe si stop pendant pause
                if should_stop:
                    break

            # Logger: nouvelle itération
            if self.orchestration_logger:
                self.orchestration_logger.next_iteration()

            # Générer le contexte pour le LLM
            context = self._build_iteration_context(
                executor, session, param_bounds, min_sharpe, max_drawdown
            )

            # Demander une décision au LLM
            decision = self._get_llm_decision(context)
            session.decisions.append(decision)

            # Logger: décision prise
            if self.orchestration_logger:
                action_type = decision.action if decision.action in ("continue", "stop", "change_approach") else "continue"
                self.orchestration_logger.log_decision(
                    agent="AutonomousStrategist",
                    decision_type=action_type,
                    reason=decision.reasoning,
                    details={"next_params": decision.next_parameters, "confidence": decision.confidence},
                )

            # Logging détaillé de la décision (toujours actif pour actions critiques)
            if decision.action in ("stop", "accept"):
                logger.warning(
                    f"🤖 Iteration {iteration}/{max_iterations}: ACTION CRITIQUE = '{decision.action.upper()}' | "
                    f"Confidence={decision.confidence:.2f} | Reasoning: {decision.reasoning}"
                )
            elif self.verbose:
                logger.info(
                    f"🤖 Iteration {iteration}/{max_iterations}: action='{decision.action}' | "
                    f"confidence={decision.confidence:.2f} | reasoning={decision.reasoning[:80]}..."
                )
            else:
                # Log minimaliste pour continue/change_direction
                logger.info(f"Iteration {iteration}: {decision.action}")

            # Traiter la décision
            if decision.action == "accept":
                session.final_status = "success"
                session.final_reasoning = decision.reasoning
                logger.info(f"Optimisation acceptée: {decision.reasoning}")
                break

            elif decision.action == "stop":
                session.final_status = "no_improvement"
                session.final_reasoning = decision.reasoning
                logger.info(f"Arrêt: {decision.reasoning}")
                break

            elif decision.action in ("continue", "change_direction"):
                # Valider et corriger les paramètres
                next_params = self._validate_parameters(
                    decision.next_parameters, param_bounds, initial_params
                )

                # Logger: changement de paramètres
                if self.orchestration_logger and next_params != session.best_result.request.parameters:
                    for param, new_value in next_params.items():
                        old_value = session.best_result.request.parameters.get(param)
                        if old_value != new_value:
                            self.orchestration_logger.log_indicator_values_change(
                                agent="AutonomousStrategist",
                                indicator=param,
                                old_values={"value": old_value},
                                new_values={"value": new_value},
                                reason=decision.next_hypothesis,
                            )

                # Créer la requête
                request = BacktestRequest(
                    requested_by=self.role.value,
                    hypothesis=decision.next_hypothesis,
                    parameters=next_params,
                )

                # Logger: lancement du backtest
                if self.orchestration_logger:
                    self.orchestration_logger.log_backtest_launch(
                        agent="AutonomousStrategist",
                        params=next_params,
                        combination_id=iteration,
                        total_combinations=max_iterations,
                    )

                # Exécuter le backtest (avec déchargement LLM si activé)
                result = self._run_backtest_with_gpu_optimization(executor, request)
                session.all_results.append(result)

                # Logger: résultat du backtest
                if self.orchestration_logger:
                    self.orchestration_logger.log_backtest_complete(
                        agent="AutonomousStrategist",
                        params=next_params,
                        results={
                            "pnl": result.total_return,
                            "sharpe": result.sharpe_ratio,
                            "return": result.total_return,
                        },
                        combination_id=iteration,
                    )

                # Mettre à jour le meilleur si applicable
                if self._is_better(result, session.best_result, target_metric):
                    session.best_result = result
                    logger.info(
                        f"Nouveau meilleur! Sharpe={result.sharpe_ratio:.3f}"
                    )

                if self.on_progress:
                    self.on_progress(iteration, result)

        else:
            session.final_status = "max_iterations"
            session.final_reasoning = f"Reached {max_iterations} iterations"

        # Logger: fin de l'optimisation
        if self.orchestration_logger:
            self.orchestration_logger.log_analysis_complete(
                agent="AutonomousStrategist",
                results={
                    "status": session.final_status,
                    "reasoning": session.final_reasoning,
                    "best_sharpe": session.best_result.sharpe_ratio,
                    "iterations": session.current_iteration,
                },
            )

            # Forcer la sauvegarde finale des logs
            try:
                self.orchestration_logger.save_to_jsonl()
            except Exception as e:
                logger.warning(f"Échec de la sauvegarde finale des logs: {e}")

        logger.info(
            f"Optimisation terminée: {session.final_status} | "
            f"Meilleur Sharpe: {session.best_result.sharpe_ratio:.3f}"
        )

        return session

    def _build_iteration_context(
        self,
        executor: BacktestExecutor,
        session: OptimizationSession,
        param_bounds: Dict[str, tuple],
        min_sharpe: float,
        max_drawdown: float,
    ) -> str:
        """Construit le contexte pour le LLM."""

        lines = [
            f"=== Optimization Session: {session.strategy_name} ===",
            f"Iteration: {session.current_iteration}/{session.max_iterations}",
            f"Target: {session.target_metric}",
            "",
            "Parameter Bounds:",
        ]

        # Calculer les statistiques d'espace de recherche
        try:
            stats = compute_search_space_stats(param_bounds, max_combinations=100000)

            for param, (min_val, max_val) in param_bounds.items():
                current = session.best_result.request.parameters.get(param, "?")
                param_count = stats.per_param_counts.get(param, "?")
                lines.append(
                    f"  {param}: [{min_val}, {max_val}] "
                    f"(current: {current}, values: {param_count})"
                )

            # Ajouter le résumé de l'espace de recherche
            lines.extend([
                "",
                "Search Space:",
                f"  {stats.summary()}",
            ])

            if stats.warnings:
                lines.append(f"  Warnings: {', '.join(stats.warnings)}")

        except Exception as e:
            # Fallback si le calcul des stats échoue
            logger.warning(f"Failed to compute search space stats: {e}")
            for param, (min_val, max_val) in param_bounds.items():
                current = session.best_result.request.parameters.get(param, "?")
                lines.append(f"  {param}: [{min_val}, {max_val}] (current: {current})")

        lines.extend([
            "",
            "Constraints:",
            f"  Min Sharpe: {min_sharpe}",
            f"  Max Drawdown: {max_drawdown:.0%}",
            "",
            "--- Experiment History ---",
            executor.get_context_for_agent(),
        ])

        # Dernières décisions
        if session.decisions:
            lines.extend(["", "Recent Decisions:"])
            for i, dec in enumerate(session.decisions[-5:], 1):
                lines.append(f"  {i}. {dec.action}: {dec.reasoning[:60]}...")

        lines.extend([
            "",
            "What is your next action?",
            "Remember: respond in JSON format with action, reasoning, next_hypothesis, next_parameters.",
        ])

        return "\n".join(lines)

    def _get_llm_decision(self, context: str) -> IterationDecision:
        """Obtient une décision du LLM."""

        response = self._call_llm(context, json_mode=True, temperature=0.5)

        if not response.content:
            logger.error("❌ LLM n'a pas répondu (response.content vide)")
            return IterationDecision(
                action="stop",
                confidence=0.0,
                reasoning="LLM did not respond",
            )

        data = response.parse_json()
        if data is None:
            logger.error(
                f"❌ Échec parsing JSON de la réponse LLM. "
                f"Réponse brute (100 premiers chars): {response.content[:100]}"
            )
            return IterationDecision(
                action="stop",
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {response.content[:100]}",
            )

        # Log succès du parsing
        action = data.get("action", "stop")
        logger.debug(f"✅ Décision LLM parsée avec succès: action='{action}'")

        # Protection contre les valeurs None du LLM
        next_params = data.get("next_parameters", {})
        if next_params is None:
            next_params = {}

        insights = data.get("insights", [])
        if insights is None:
            insights = []

        return IterationDecision(
            action=action,
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", "") or "",
            next_hypothesis=data.get("next_hypothesis", "") or "",
            next_parameters=next_params,
            insights=insights,
        )

    def _validate_parameters(
        self,
        params: Dict[str, Any],
        bounds: Dict[str, tuple],
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Valide et corrige les paramètres avec checks robustes."""

        validated = {}

        for param, bound_spec in bounds.items():
            try:
                # Extraire min/max selon le format des bounds
                # Peut être (min, max) ou (min, max, step) ou [min, max, ...]
                if isinstance(bound_spec, (tuple, list)) and len(bound_spec) >= 2:
                    min_val = float(bound_spec[0])
                    max_val = float(bound_spec[1])

                    # Validation: min < max
                    if min_val >= max_val:
                        logger.warning(f"Param {param}: min >= max ({min_val} >= {max_val}), swap")
                        min_val, max_val = max_val, min_val
                else:
                    # Valeur scalaire (cas edge)
                    min_val = max_val = float(bound_spec) if not isinstance(bound_spec, (tuple, list)) else float(bound_spec[0])

                value = params.get(param, defaults.get(param))

                if value is None:
                    value = (min_val + max_val) / 2
                else:
                    value = float(value)

                # Clamp dans les bornes
                value = max(min_val, min(max_val, value))

                # Arrondir si nécessaire (detect int bounds)
                if all(isinstance(bound_spec[i], int) for i in range(2) if i < len(bound_spec)):
                    value = int(round(value))

                validated[param] = value

            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Param {param} validation failed: {e}, use default")
                validated[param] = defaults.get(param, 0)

        return validated

    def _is_better(
        self,
        new: BacktestResult,
        current: BacktestResult,
        metric: str,
    ) -> bool:
        """Détermine si un résultat est meilleur."""

        if not new.success:
            return False

        # Vérifier overfitting
        if new.overfitting_ratio > 1.5:
            return False

        # Comparer la métrique
        new_value = getattr(new, metric, 0)
        current_value = getattr(current, metric, 0)

        if metric in ("max_drawdown",):  # Métriques à minimiser
            return new_value < current_value
        else:  # Métriques à maximiser
            return new_value > current_value

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Exécute une itération (pour compatibilité avec BaseAgent).

        Pour une optimisation complète, utilisez optimize() directement.
        """
        return AgentResult(
            success=True,
            role=self.role,
            content="Use optimize() method for autonomous optimization",
            data={},
            execution_time_ms=0,
            tokens_used=0,
            llm_calls=0,
        )


def create_autonomous_optimizer(
    llm_config: LLMConfig,
    backtest_fn: Callable,
    strategy_name: str,
    data: pd.DataFrame,
    validation_fn: Optional[Callable] = None,
) -> tuple[AutonomousStrategist, BacktestExecutor]:
    """
    Factory pour créer un optimiseur autonome complet.

    Args:
        llm_config: Configuration LLM
        backtest_fn: Fonction de backtest (strategy, params, data) -> metrics
        strategy_name: Nom de la stratégie
        data: DataFrame OHLCV
        validation_fn: Fonction walk-forward optionnelle

    Returns:
        (AutonomousStrategist, BacktestExecutor) prêts à l'emploi

    Example:
        >>> from agents.llm_client import LLMConfig, LLMProvider
        >>>
        >>> config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
        >>> strategist, executor = create_autonomous_optimizer(
        ...     llm_config=config,
        ...     backtest_fn=run_backtest,
        ...     strategy_name="ema_cross",
        ...     data=ohlcv_df,
        ... )
        >>>
        >>> session = strategist.optimize(
        ...     executor=executor,
        ...     initial_params={"fast": 10, "slow": 21},
        ...     param_bounds={"fast": (5, 20), "slow": (15, 50)},
        ... )
    """
    from .llm_client import create_llm_client

    llm_client = create_llm_client(llm_config)

    executor = BacktestExecutor(
        backtest_fn=backtest_fn,
        strategy_name=strategy_name,
        data=data,
        validation_fn=validation_fn,
    )

    strategist = AutonomousStrategist(llm_client, verbose=True)

    return strategist, executor
