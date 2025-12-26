"""
Orchestrator - Chef d'orchestre du workflow d'optimisation.

Coordonne:
- La State Machine (transitions validées)
- Les 4 Agents (Analyst, Strategist, Critic, Validator)
- L'exécution des backtests
- La convergence et les critères d'arrêt

Workflow complet:
    INIT → [ANALYZE → PROPOSE → CRITIQUE → VALIDATE]* → APPROVED/REJECTED
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .analyst import AnalystAgent
from .base_agent import AgentContext, MetricsSnapshot, ParameterConfig
from .critic import CriticAgent
from .integration import run_walk_forward_for_agent
from .llm_client import LLMConfig, create_llm_client
from .model_config import RoleModelConfig
from .state_machine import AgentState, StateMachine, ValidationResult
from .strategist import StrategistAgent
from .validator import ValidationDecision, ValidatorAgent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration de l'Orchestrator."""

    # Stratégie
    strategy_name: str = ""
    strategy_description: str = ""

    # Données
    data_path: str = ""
    data: Optional[Any] = None

    # Paramètres initiaux
    initial_params: Dict[str, Any] = field(default_factory=dict)
    param_specs: List[ParameterConfig] = field(default_factory=list)

    # Objectifs d'optimisation
    optimization_target: str = "sharpe_ratio"
    min_sharpe: float = 1.0
    max_drawdown_limit: float = 0.20
    min_trades: int = 30
    max_overfitting_ratio: float = 1.5

    # Limites
    max_iterations: int = 10
    max_proposals_per_iteration: int = 5

    # Exécution
    n_workers: int = 1
    session_id: Optional[str] = None
    orchestration_logger: Optional[Any] = None

    # LLM
    llm_config: Optional[LLMConfig] = None
    role_model_config: Optional[RoleModelConfig] = None

    # Walk-forward
    use_walk_forward: bool = True
    walk_forward_windows: int = 5
    train_ratio: float = 0.7

    # Callbacks (optionnels)
    on_state_change: Optional[Callable[[AgentState, AgentState], None]] = None
    on_iteration_complete: Optional[Callable[[int, Dict], None]] = None
    on_backtest_needed: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


@dataclass
class OrchestratorResult:
    """Résultat final de l'orchestration."""

    success: bool
    final_state: AgentState
    decision: str  # APPROVE, REJECT, ABORT

    # Configuration finale
    final_params: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Optional[MetricsSnapshot] = None

    # Métriques d'exécution
    total_iterations: int = 0
    total_backtests: int = 0
    total_time_s: float = 0.0
    total_llm_tokens: int = 0
    total_llm_calls: int = 0

    # Historique
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    # Rapport
    final_report: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Erreurs
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Orchestrator:
    """
    Orchestrator - Coordonne le workflow d'optimisation LLM.

    Garanties:
    - Transitions validées via State Machine
    - Pas de boucles infinies (max_iterations)
    - Traçabilité complète
    - Gestion des erreurs gracieuse

    Example:
        >>> config = OrchestratorConfig(
        ...     strategy_name="ema_cross",
        ...     data_path="data/BTCUSDT_1h.parquet",
        ...     initial_params={"fast_period": 12, "slow_period": 26},
        ...     param_specs=[...],
        ... )
        >>> orchestrator = Orchestrator(config)
        >>> result = orchestrator.run()
        >>> if result.success:
        ...     print(f"Optimized params: {result.final_params}")
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialise l'Orchestrator.

        Args:
            config: Configuration complète
        """
        self.config = config
        if config.session_id:
            self.session_id = str(config.session_id)
        elif config.orchestration_logger is not None and hasattr(config.orchestration_logger, "session_id"):
            self.session_id = str(getattr(config.orchestration_logger, "session_id"))
        else:
            self.session_id = str(uuid.uuid4())[:8]

        # State Machine
        self.state_machine = StateMachine(max_iterations=config.max_iterations)

        # LLM Client
        llm_config = config.llm_config or LLMConfig.from_env()
        self.llm_client = create_llm_client(llm_config)

        # Agents
        self.analyst = AnalystAgent(self.llm_client)
        self.strategist = StrategistAgent(self.llm_client)
        self.critic = CriticAgent(self.llm_client)
        self.validator = ValidatorAgent(self.llm_client)

        # Context partagé
        self.context = self._create_initial_context()

        # Tracking
        self._start_time: Optional[float] = None
        self._backtests_count = 0
        self._errors: List[str] = []
        self._warnings: List[str] = []

        # Données chargées (pour walk-forward)
        self._loaded_data: Optional[Any] = getattr(config, "data", None)

        # Orchestration logger (optionnel, non bloquant)
        self._orch_logger: Any = None
        self._init_orchestration_logger()

        logger.info(
            f"Orchestrator initialisé: session={self.session_id}, "
            f"strategy={config.strategy_name}, max_iter={config.max_iterations}"
        )

    def _create_initial_context(self) -> AgentContext:
        """Crée le contexte initial."""
        return AgentContext(
            session_id=self.session_id,
            iteration=0,
            strategy_name=self.config.strategy_name,
            strategy_description=self.config.strategy_description,
            current_params=self.config.initial_params.copy(),
            param_specs=self.config.param_specs,
            data_path=self.config.data_path,
            optimization_target=self.config.optimization_target,
            min_sharpe=self.config.min_sharpe,
            max_drawdown_limit=self.config.max_drawdown_limit,
            min_trades=self.config.min_trades,
            max_overfitting_ratio=self.config.max_overfitting_ratio,
        )

    def _init_orchestration_logger(self) -> None:
        """Initialise un logger d'orchestration s'il est disponible (non bloquant)."""
        # Utilise un logger fourni dans la config si présent
        if getattr(self.config, "orchestration_logger", None) is not None:
            self._orch_logger = self.config.orchestration_logger
            return
        # Tentative de création depuis le module dédié (si disponible)
        try:
            from .orchestration_logger import OrchestrationLogger  # type: ignore
            self._orch_logger = OrchestrationLogger(session_id=self.session_id)
        except Exception:
            self._orch_logger = None  # Mode dégradé: aucune trace persistée

    def _log_event(self, event_type: str, **payload: Any) -> None:
        """
        Ajoute un événement d'orchestration de manière non bloquante.
        Supporte plusieurs API possibles (log/add_event/append).
        """
        if not self._orch_logger:
            return
        try:
            entry = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "iteration": getattr(self.state_machine, "iteration", 0),
                **payload,
            }
            if hasattr(self._orch_logger, "log"):
                self._orch_logger.log(event_type, entry)  # type: ignore[attr-defined]
            elif hasattr(self._orch_logger, "add_event"):
                self._orch_logger.add_event(event_type, entry)  # type: ignore[attr-defined]
            elif hasattr(self._orch_logger, "append"):
                self._orch_logger.append(entry)  # type: ignore[attr-defined]
        except Exception as e:
            # Ne jamais bloquer le flux pour la traçabilité
            logger.debug("Orchestration log failed: %s", e)

    def _apply_role_model(self, role: str) -> Optional[str]:
        """Select and apply a model for the given role if configured."""
        config = self.config.role_model_config
        if not config:
            return None

        iteration = max(1, self.state_machine.iteration)
        model_name = config.get_model(
            role=role,
            iteration=iteration,
            random_selection=True,
        )
        if model_name and self.llm_client.config.model != model_name:
            self.llm_client.config.model = model_name
            logger.info("Role %s model set to %s", role, model_name)
        return model_name

    def run(self) -> OrchestratorResult:
        """
        Exécute le workflow d'optimisation complet.

        Returns:
            Résultat de l'orchestration
        """
        self._start_time = time.time()
        self._log_event(
            "run_start",
            strategy=self.config.strategy_name,
            max_iterations=self.config.max_iterations,
            data_path=bool(self.config.data_path),
        )
        logger.info(f"=== Démarrage orchestration {self.session_id} ===")

        try:
            # Transition vers INIT → ANALYZE
            self._run_workflow()

        except Exception as e:
            logger.error(f"Erreur orchestration: {e}", exc_info=True)
            self._log_event("error", scope="orchestration", message=str(e))
            self.state_machine.fail(str(e), e)
            self._errors.append(str(e))

        # Construire le résultat final
        result = self._build_result()
        self._log_event(
            "run_end",
            success=result.success,
            decision=result.decision,
            total_iterations=result.total_iterations,
            total_backtests=result.total_backtests,
            total_time_s=result.total_time_s,
            total_llm_calls=result.total_llm_calls,
            total_llm_tokens=result.total_llm_tokens,
            errors=len(result.errors),
            warnings=len(result.warnings),
        )

        # Forcer la sauvegarde finale des logs
        if self.config.orchestration_logger:
            try:
                self.config.orchestration_logger.save_to_jsonl()
            except Exception as e:
                logger.warning(f"Échec de la sauvegarde finale des logs: {e}")

        return result

    def _run_workflow(self) -> None:
        """Exécute la boucle principale du workflow."""

        while not self.state_machine.is_terminal:
            current = self.state_machine.current_state
            self._log_event("state_enter", state=current.name)
            logger.info(f"État actuel: {current.name}")

            # Dispatch selon l'état
            if current == AgentState.INIT:
                self._handle_init()
            elif current == AgentState.ANALYZE:
                self._handle_analyze()
            elif current == AgentState.PROPOSE:
                self._handle_propose()
            elif current == AgentState.CRITIQUE:
                self._handle_critique()
            elif current == AgentState.VALIDATE:
                self._handle_validate()
            elif current == AgentState.ITERATE:
                self._handle_iterate()
            else:
                logger.error(f"État non géré: {current}")
                self.state_machine.fail(f"État non géré: {current}")
                break

            # Callback + log de transition
            if self.config.on_state_change:
                self.config.on_state_change(current, self.state_machine.current_state)
            self._log_event(
                "state_change",
                state_from=current.name,
                state_to=self.state_machine.current_state.name,
            )

    def _handle_init(self) -> None:
        """Gère l'état INIT - Initialisation et validation."""
        self._log_event("phase_start", phase="INIT")
        logger.info("Phase INIT: Validation configuration et backtest initial")

        # Valider la configuration
        validation = self._validate_config()
        if not validation.is_valid:
            self._log_event("config_invalid", errors=validation.errors or [], message=validation.message)
            self.state_machine.fail(f"Configuration invalide: {validation.message}")
            return
        self._log_event("config_valid")

        # Exécuter le backtest initial
        initial_metrics = None
        try:
            initial_metrics = self._run_backtest(self.context.current_params)
            if initial_metrics:
                self.context.current_metrics = initial_metrics
                self.context.best_metrics = initial_metrics
                self.context.best_params = self.context.current_params.copy()
                self._log_event(
                    "initial_backtest_done",
                    sharpe=initial_metrics.sharpe_ratio,
                    total_return=initial_metrics.total_return,
                    max_drawdown=initial_metrics.max_drawdown,
                )
                logger.info(
                    f"Backtest initial: Sharpe={initial_metrics.sharpe_ratio:.3f}, "
                    f"Return={initial_metrics.total_return:.2%}"
                )
            else:
                self._warnings.append("Backtest initial sans métriques")
                self._log_event("warning", message="Backtest initial sans métriques")
        except Exception as e:
            self._warnings.append(f"Erreur backtest initial: {e}")
            self._log_event("warning", message=f"Erreur backtest initial: {e}")
            logger.error(f"Erreur backtest initial: {e}", exc_info=True)

        # Fallback: créer des métriques à zéro si le backtest a échoué
        if initial_metrics is None:
            logger.warning("Backtest initial échoué, utilisation de métriques par défaut (zéro)")
            initial_metrics = MetricsSnapshot(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
            )
            self.context.current_metrics = initial_metrics
            self._warnings.append("Utilisation de métriques par défaut (backtest échoué)")
            self._log_event("warning", message="Métriques par défaut utilisées")

        # Calculer les métriques walk-forward si données disponibles
        self._compute_walk_forward_metrics()

        # Transition vers ANALYZE
        self.state_machine.transition_to(AgentState.ANALYZE)

    def _handle_analyze(self) -> None:
        """Gère l'état ANALYZE - Exécution de l'Agent Analyst."""
        self._log_event("phase_start", phase="ANALYZE")
        logger.info("Phase ANALYZE: Exécution Agent Analyst")

        # Mettre à jour l'itération dans le contexte
        self.context.iteration = self.state_machine.iteration

        # Exécuter l'Analyst
        self._apply_role_model("analyst")
        self._log_event("agent_execute_start", role="analyst", model=self.llm_client.config.model)
        t0 = time.time()
        result = self.analyst.execute(self.context)
        dt = int((time.time() - t0) * 1000)
        self._log_event("agent_execute_end", role="analyst", success=result.success, latency_ms=dt)

        if not result.success:
            logger.error(f"Analyst échoué: {result.errors}")
            self._log_event("error", scope="analyst", message=str(result.errors))
            self._errors.extend(result.errors)
            # Continuer quand même - l'analyse n'est pas bloquante
            self.context.analyst_report = "Analyse non disponible"
        else:
            # Stocker le rapport
            self.context.analyst_report = result.content

            # Vérifier si on doit continuer l'optimisation
            proceed = result.data.get("proceed_to_optimization", True)
            self._log_event("analyst_result", proceed=bool(proceed))
            if not proceed:
                logger.info("Analyst recommande de ne pas optimiser")
                transition = self.state_machine.transition_to(AgentState.VALIDATE)
                if not transition.is_valid:
                    logger.warning(
                        "Transition ANALYZE -> VALIDATE refusee: %s",
                        transition.message,
                    )
                    self.state_machine.transition_to(AgentState.VALIDATE, force=True)
                return

        # Transition vers PROPOSE
        self.state_machine.transition_to(AgentState.PROPOSE)

    def _handle_propose(self) -> None:
        """Gère l'état PROPOSE - Exécution de l'Agent Strategist."""
        self._log_event("phase_start", phase="PROPOSE")
        logger.info("Phase PROPOSE: Exécution Agent Strategist")

        # Exécuter le Strategist
        self._apply_role_model("strategist")
        self._log_event("agent_execute_start", role="strategist", model=self.llm_client.config.model)
        t0 = time.time()
        result = self.strategist.execute(self.context)
        dt = int((time.time() - t0) * 1000)
        self._log_event("agent_execute_end", role="strategist", success=result.success, latency_ms=dt)

        if not result.success:
            logger.error(f"Strategist échoué: {result.errors}")
            self._log_event("error", scope="strategist", message=str(result.errors))
            self._errors.extend(result.errors)
            # Sans propositions, on va directement à la validation
            self.context.strategist_proposals = []
            self.state_machine.transition_to(AgentState.VALIDATE)
            return

        # Stocker les propositions
        proposals = result.data.get("proposals", [])
        self.context.strategist_proposals = proposals[:self.config.max_proposals_per_iteration]
        self._log_event("proposals_generated", count=len(self.context.strategist_proposals))
        logger.info(f"Strategist: {len(self.context.strategist_proposals)} propositions générées")

        # Transition vers CRITIQUE
        self.state_machine.transition_to(AgentState.CRITIQUE)

    def _handle_critique(self) -> None:
        """Gère l'état CRITIQUE - Exécution de l'Agent Critic."""
        self._log_event("phase_start", phase="CRITIQUE")
        logger.info("Phase CRITIQUE: Exécution Agent Critic")

        if not self.context.strategist_proposals:
            logger.warning("Aucune proposition à critiquer")
            self._log_event("warning", message="Aucune proposition à critiquer")
            self.state_machine.transition_to(AgentState.VALIDATE)
            return

        # Exécuter le Critic
        self._apply_role_model("critic")
        self._log_event("agent_execute_start", role="critic", model=self.llm_client.config.model)
        t0 = time.time()
        result = self.critic.execute(self.context)
        dt = int((time.time() - t0) * 1000)
        self._log_event("agent_execute_end", role="critic", success=result.success, latency_ms=dt)

        if not result.success:
            logger.error(f"Critic échoué: {result.errors}")
            self._log_event("error", scope="critic", message=str(result.errors))
            self._errors.extend(result.errors)
            # Continuer avec les propositions non filtrées
            self.context.critic_concerns = []
        else:
            # Mettre à jour avec les propositions filtrées
            approved = result.data.get("approved_proposals", [])
            if approved:
                self.context.strategist_proposals = approved

            self.context.critic_assessment = result.content
            self.context.critic_concerns = result.data.get("concerns", [])
            self._log_event(
                "critic_result",
                approved_count=len(approved),
                concerns_count=len(self.context.critic_concerns),
            )
            logger.info(
                f"Critic: {len(approved)} propositions approuvées, "
                f"{len(self.context.critic_concerns)} concerns"
            )

        # Tester les propositions approuvées
        self._test_proposals()

        # Transition vers VALIDATE
        self.state_machine.transition_to(AgentState.VALIDATE)

    def _handle_validate(self) -> None:
        """Gère l'état VALIDATE - Exécution de l'Agent Validator."""
        self._log_event("phase_start", phase="VALIDATE")
        logger.info("Phase VALIDATE: Exécution Agent Validator")

        # Exécuter le Validator
        self._apply_role_model("validator")
        self._log_event("agent_execute_start", role="validator", model=self.llm_client.config.model)
        t0 = time.time()
        result = self.validator.execute(self.context)
        dt = int((time.time() - t0) * 1000)
        self._log_event("agent_execute_end", role="validator", success=result.success, latency_ms=dt)

        if not result.success:
            logger.error(f"Validator échoué: {result.errors}")
            self._log_event("error", scope="validator", message=str(result.errors))
            self._errors.extend(result.errors)
            # Par défaut, on itère si le validator échoue
            decision = ValidationDecision.ITERATE
        else:
            decision_str = result.data.get("decision", "ITERATE")
            try:
                decision = ValidationDecision(decision_str)
            except ValueError:
                decision = ValidationDecision.ITERATE

        logger.info(f"Validator décision: {decision.value}")
        self._log_event("validator_decision", decision=decision.value)

        # Enregistrer l'historique de l'itération
        self._record_iteration()

        # Transition selon la décision
        if decision == ValidationDecision.APPROVE:
            self.state_machine.transition_to(AgentState.APPROVED)
        elif decision == ValidationDecision.REJECT:
            self.state_machine.transition_to(AgentState.REJECTED)
        elif decision == ValidationDecision.ABORT:
            self.state_machine.fail("Validator a décidé ABORT")
        else:  # ITERATE
            # Vérifier si on peut encore itérer
            if self.state_machine.can_transition_to(AgentState.ITERATE):
                self.state_machine.transition_to(AgentState.ITERATE)
            else:
                logger.info("Max iterations atteint, passage en REJECTED")
                self.state_machine.transition_to(AgentState.REJECTED)

    def _handle_iterate(self) -> None:
        """Gère l'état ITERATE - Préparation de l'itération suivante."""
        self._log_event("phase_start", phase="ITERATE")
        logger.info("Phase ITERATE: Préparation itération suivante")

        # Sélectionner la meilleure configuration testée
        best_tested = self._get_best_tested_config()
        if best_tested:
            self.context.current_params = best_tested["params"]
            if best_tested.get("metrics"):
                self.context.current_metrics = best_tested["metrics"]

                # Mettre à jour le best si meilleur
                if (self.context.best_metrics is None or
                    best_tested["metrics"].sharpe_ratio > self.context.best_metrics.sharpe_ratio):
                    self.context.best_metrics = best_tested["metrics"]
                    self.context.best_params = best_tested["params"].copy()

        # Nettoyer les propositions
        self.context.strategist_proposals = []
        self.context.critic_concerns = []

        # Callback itération complète
        if self.config.on_iteration_complete:
            self.config.on_iteration_complete(
                self.state_machine.iteration,
                {"metrics": self.context.current_metrics, "params": self.context.current_params}
            )

        # Transition vers ANALYZE
        self.state_machine.transition_to(AgentState.ANALYZE)

    def _validate_config(self) -> ValidationResult:
        """Valide la configuration initiale."""
        errors = []

        if not self.config.strategy_name:
            errors.append("strategy_name requis")

        if self.config.data_path:
            if not Path(self.config.data_path).exists():
                errors.append(f"data_path n'existe pas: {self.config.data_path}")
        elif self.config.on_backtest_needed is None:
            errors.append("data_path requis")

        if not self.config.param_specs:
            errors.append("param_specs requis (au moins un paramètre)")

        if errors:
            return ValidationResult.failure("; ".join(errors), errors)

        return ValidationResult.success()

    def _run_backtest(self, params: Dict[str, Any]) -> Optional[MetricsSnapshot]:
        """
        Exécute un backtest avec les paramètres donnés.

        Utilise le callback on_backtest_needed si fourni,
        sinon retourne None.
        """
        self._backtests_count += 1
        self._log_event("backtest_start", source="orchestrator", params=params)

        if self.config.on_backtest_needed:
            try:
                result = self.config.on_backtest_needed(params)
                if result:
                    metrics = MetricsSnapshot.from_dict(result)
                    self._log_event(
                        "backtest_end",
                        success=True,
                        sharpe=metrics.sharpe_ratio,
                        total_return=metrics.total_return,
                        max_drawdown=metrics.max_drawdown,
                    )
                    return metrics
            except Exception as e:
                logger.error(f"Erreur backtest: {e}")
                self._warnings.append(f"Backtest échoué: {e}")
                self._log_event("backtest_end", success=False, error=str(e))

        return None

    def _compute_walk_forward_metrics(self) -> None:
        """
        Calcule les métriques de walk-forward validation et met à jour le contexte.

        Charge les données si nécessaire et exécute une validation walk-forward
        pour détecter l'overfitting avec les métriques robustes.
        """
        # Si on a déjà des données en mémoire (UI), on les utilise.
        if self._loaded_data is not None:
            data_df = self._loaded_data
            try:
                self.context.data_rows = len(data_df)
            except Exception:
                pass

            try:
                wf_metrics = run_walk_forward_for_agent(
                    strategy_name=self.config.strategy_name,
                    params=self.context.current_params,
                    data=data_df,
                    n_windows=6,
                    train_ratio=0.75,
                    n_workers=self.config.n_workers,
                )

                self.context.overfitting_ratio = wf_metrics["overfitting_ratio"]
                self.context.classic_ratio = wf_metrics["classic_ratio"]
                self.context.degradation_pct = wf_metrics["degradation_pct"]
                self.context.test_stability_std = wf_metrics["test_stability_std"]
                self.context.n_valid_folds = wf_metrics["n_valid_folds"]
                self.context.walk_forward_windows = 6

                self._log_event(
                    "walk_forward_computed",
                    overfitting_ratio=float(wf_metrics["overfitting_ratio"]),
                    classic_ratio=float(wf_metrics["classic_ratio"]),
                    degradation_pct=float(wf_metrics["degradation_pct"]),
                    test_stability_std=float(wf_metrics["test_stability_std"]),
                    n_valid_folds=int(wf_metrics["n_valid_folds"]),
                )
            except Exception as e:
                logger.warning(f"Échec du calcul des métriques walk-forward: {e}")
                self._warnings.append(f"Walk-forward échoué: {e}")
                self._log_event("warning", message=f"Walk-forward échoué: {e}")

            return

        # Vérifier si un chemin de données est fourni
        if not self.config.data_path:
            logger.debug("Pas de data/data_path configuré, skip walk-forward metrics")
            return

        data_path = Path(self.config.data_path)
        if not data_path.exists():
            logger.warning(f"Fichier de données introuvable: {data_path}")
            return

        try:
            # Charger les données si pas déjà fait
            if self._loaded_data is None:
                logger.info(f"Chargement des données depuis {data_path}")
                import pandas as pd

                # Charger selon l'extension
                if data_path.suffix == '.csv':
                    self._loaded_data = pd.read_csv(data_path)
                elif data_path.suffix == '.parquet':
                    self._loaded_data = pd.read_parquet(data_path)
                else:
                    logger.warning(f"Format non supporté pour walk-forward: {data_path.suffix}")
                    return

                logger.info(f"  Données chargées: {len(self._loaded_data)} lignes")

            # Mettre à jour le contexte avec les infos sur les données
            self.context.data_rows = len(self._loaded_data)

            # Extraire la plage de dates si disponible
            if 'timestamp' in self._loaded_data.columns or 'date' in self._loaded_data.columns:
                date_col = 'timestamp' if 'timestamp' in self._loaded_data.columns else 'date'
                try:
                    import pandas as pd
                    dates = pd.to_datetime(self._loaded_data[date_col])
                    self.context.data_date_range = f"{dates.min()} → {dates.max()}"
                except Exception:
                    pass

            # Exécuter la validation walk-forward
            logger.info("Exécution de la validation walk-forward...")
            wf_metrics = run_walk_forward_for_agent(
                strategy_name=self.config.strategy_name,
                params=self.context.current_params,
                data=self._loaded_data,
                n_windows=6,
                train_ratio=0.75,
                n_workers=self.config.n_workers,
            )

            # Mettre à jour le contexte avec les métriques
            self.context.overfitting_ratio = wf_metrics["overfitting_ratio"]
            self.context.classic_ratio = wf_metrics["classic_ratio"]
            self.context.degradation_pct = wf_metrics["degradation_pct"]
            self.context.test_stability_std = wf_metrics["test_stability_std"]
            self.context.n_valid_folds = wf_metrics["n_valid_folds"]
            self.context.walk_forward_windows = 6

            logger.info(
                f"Walk-forward terminé: "
                f"overfitting_ratio={wf_metrics['overfitting_ratio']:.3f}, "
                f"degradation={wf_metrics['degradation_pct']:.1f}%, "
                f"stability_std={wf_metrics['test_stability_std']:.3f}"
            )

            # Journaliser l'événement
            self._log_event(
                "walk_forward_computed",
                overfitting_ratio=float(wf_metrics["overfitting_ratio"]),
                classic_ratio=float(wf_metrics["classic_ratio"]),
                degradation_pct=float(wf_metrics["degradation_pct"]),
                test_stability_std=float(wf_metrics["test_stability_std"]),
                n_valid_folds=int(wf_metrics["n_valid_folds"]),
            )

        except Exception as e:
            logger.warning(f"Échec du calcul des métriques walk-forward: {e}")
            self._warnings.append(f"Walk-forward échoué: {e}")
            self._log_event("warning", message=f"Walk-forward échoué: {e}")

    def _test_proposals(self) -> None:
        """Teste les propositions approuvées via backtest."""
        proposals = list(self.context.strategist_proposals or [])
        if not proposals:
            return

        def _eval_one(proposal: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[MetricsSnapshot]]:
            params = proposal.get("parameters", {})
            if not params:
                return proposal, None
            return proposal, self._run_backtest(params)

        n_workers = int(getattr(self.config, "n_workers", 1) or 1)

        # Séquentiel par défaut
        if n_workers <= 1 or len(proposals) <= 1:
            for proposal in proposals:
                params = proposal.get("parameters", {})
                if not params:
                    continue

                self._log_event(
                    "proposal_test_started",
                    proposal_id=proposal.get("id"),
                    proposal_name=proposal.get("name"),
                )
                logger.info(f"Test proposition {proposal.get('id')}: {proposal.get('name')}")

                metrics = self._run_backtest(params)
                if metrics:
                    proposal["tested_metrics"] = metrics.to_dict()
                    proposal["tested"] = True
                    self._log_event(
                        "proposal_test_ended",
                        proposal_id=proposal.get("id"),
                        tested=True,
                        sharpe=metrics.sharpe_ratio,
                        total_return=metrics.total_return,
                    )
                else:
                    proposal["tested"] = False
                    self._log_event(
                        "proposal_test_ended",
                        proposal_id=proposal.get("id"),
                        tested=False,
                    )
            return

        # Parallèle: le slider workers a enfin un effet réel en multi-agents
        from concurrent.futures import ThreadPoolExecutor, as_completed

        for proposal in proposals:
            if proposal.get("parameters", {}):
                self._log_event(
                    "proposal_test_started",
                    proposal_id=proposal.get("id"),
                    proposal_name=proposal.get("name"),
                )

        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for proposal in proposals:
                if not proposal.get("parameters", {}):
                    continue
                futures[pool.submit(_eval_one, proposal)] = proposal

            for fut in as_completed(futures):
                proposal = futures[fut]
                try:
                    _, metrics = fut.result()
                except Exception as e:
                    metrics = None
                    self._warnings.append(f"Backtest proposition échoué: {e}")
                    self._log_event(
                        "proposal_test_ended",
                        proposal_id=proposal.get("id"),
                        tested=False,
                        error=str(e),
                    )
                    continue

                if metrics:
                    proposal["tested_metrics"] = metrics.to_dict()
                    proposal["tested"] = True
                    self._log_event(
                        "proposal_test_ended",
                        proposal_id=proposal.get("id"),
                        tested=True,
                        sharpe=metrics.sharpe_ratio,
                        total_return=metrics.total_return,
                    )
                else:
                    proposal["tested"] = False
                    self._log_event(
                        "proposal_test_ended",
                        proposal_id=proposal.get("id"),
                        tested=False,
                    )

    def _get_best_tested_config(self) -> Optional[Dict[str, Any]]:
        """Retourne la meilleure configuration testée."""
        best = None
        best_sharpe = float("-inf")

        for proposal in self.context.strategist_proposals:
            if not proposal.get("tested"):
                continue

            metrics_dict = proposal.get("tested_metrics", {})
            sharpe = metrics_dict.get("sharpe_ratio", 0)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best = {
                    "params": proposal.get("parameters", {}),
                    "metrics": MetricsSnapshot.from_dict(metrics_dict),
                }

        return best

    def _record_iteration(self) -> None:
        """Enregistre l'itération actuelle dans l'historique."""
        entry = {
            "iteration": self.state_machine.iteration,
            "timestamp": datetime.now().isoformat(),
            "params": self.context.current_params.copy(),
        }

        if self.context.current_metrics:
            entry.update({
                "sharpe_ratio": self.context.current_metrics.sharpe_ratio,
                "total_return": self.context.current_metrics.total_return,
                "max_drawdown": self.context.current_metrics.max_drawdown,
            })

        entry["proposals_count"] = len(self.context.strategist_proposals)
        entry["concerns_count"] = len(self.context.critic_concerns)

        self.context.iteration_history.append(entry)

        # Journalisation non bloquante
        self._log_event(
            "iteration_recorded",
            iteration=entry["iteration"],
            sharpe=entry.get("sharpe_ratio"),
            total_return=entry.get("total_return"),
            max_drawdown=entry.get("max_drawdown"),
            proposals_count=entry.get("proposals_count", 0),
            concerns_count=entry.get("concerns_count", 0),
        )

    def _build_result(self) -> OrchestratorResult:
        """Construit le résultat final."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        # Déterminer la décision finale
        final_state = self.state_machine.current_state
        if final_state == AgentState.APPROVED:
            decision = "APPROVE"
            success = True
        elif final_state == AgentState.REJECTED:
            decision = "REJECT"
            success = False
        else:
            decision = "ABORT"
            success = False

        # Statistiques LLM (avec fallback si agent n'a pas de stats)
        def _get_agent_stats(agent):
            """Récupère les stats d'un agent de manière sûre."""
            stats = getattr(agent, "stats", {})
            return {
                "total_tokens": stats.get("total_tokens", 0),
                "execution_count": stats.get("execution_count", 0),
            }

        agents = [self.analyst, self.strategist, self.critic, self.validator]
        total_tokens = sum(_get_agent_stats(a)["total_tokens"] for a in agents)
        total_calls = sum(_get_agent_stats(a)["execution_count"] for a in agents)

        return OrchestratorResult(
            success=success,
            final_state=final_state,
            decision=decision,
            final_params=self.context.best_params or self.context.current_params,
            final_metrics=self.context.best_metrics,
            total_iterations=self.state_machine.iteration,
            total_backtests=self._backtests_count,
            total_time_s=elapsed,
            total_llm_tokens=total_tokens,
            total_llm_calls=total_calls,
            iteration_history=self.context.iteration_history,
            state_history=self.state_machine.get_summary()["history"],
            errors=self._errors,
            warnings=self._warnings,
        )
