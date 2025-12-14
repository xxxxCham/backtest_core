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

from .state_machine import AgentState, StateMachine, ValidationResult
from .llm_client import LLMClient, LLMConfig, create_llm_client
from .base_agent import AgentContext, AgentResult, MetricsSnapshot, ParameterConfig
from .analyst import AnalystAgent
from .strategist import StrategistAgent
from .critic import CriticAgent
from .validator import ValidatorAgent, ValidationDecision

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration de l'Orchestrator."""
    
    # Stratégie
    strategy_name: str = ""
    strategy_description: str = ""
    
    # Données
    data_path: str = ""
    
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
    
    # LLM
    llm_config: Optional[LLMConfig] = None
    
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
    
    def run(self) -> OrchestratorResult:
        """
        Exécute le workflow d'optimisation complet.
        
        Returns:
            Résultat de l'orchestration
        """
        self._start_time = time.time()
        logger.info(f"=== Démarrage orchestration {self.session_id} ===")
        
        try:
            # Transition vers INIT → ANALYZE
            self._run_workflow()
            
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}", exc_info=True)
            self.state_machine.fail(str(e), e)
            self._errors.append(str(e))
        
        # Construire le résultat final
        return self._build_result()
    
    def _run_workflow(self) -> None:
        """Exécute la boucle principale du workflow."""
        
        while not self.state_machine.is_terminal:
            current = self.state_machine.current_state
            
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
            
            # Callback de changement d'état
            if self.config.on_state_change:
                self.config.on_state_change(current, self.state_machine.current_state)
    
    def _handle_init(self) -> None:
        """Gère l'état INIT - Initialisation et validation."""
        logger.info("Phase INIT: Validation configuration et backtest initial")
        
        # Valider la configuration
        validation = self._validate_config()
        if not validation.is_valid:
            self.state_machine.fail(f"Configuration invalide: {validation.message}")
            return
        
        # Exécuter le backtest initial
        try:
            initial_metrics = self._run_backtest(self.context.current_params)
            if initial_metrics:
                self.context.current_metrics = initial_metrics
                self.context.best_metrics = initial_metrics
                self.context.best_params = self.context.current_params.copy()
                logger.info(
                    f"Backtest initial: Sharpe={initial_metrics.sharpe_ratio:.3f}, "
                    f"Return={initial_metrics.total_return:.2%}"
                )
            else:
                self._warnings.append("Backtest initial sans métriques")
        except Exception as e:
            self._warnings.append(f"Erreur backtest initial: {e}")
        
        # Transition vers ANALYZE
        self.state_machine.transition_to(AgentState.ANALYZE)
    
    def _handle_analyze(self) -> None:
        """Gère l'état ANALYZE - Exécution de l'Agent Analyst."""
        logger.info("Phase ANALYZE: Exécution Agent Analyst")
        
        # Mettre à jour l'itération dans le contexte
        self.context.iteration = self.state_machine.iteration
        
        # Exécuter l'Analyst
        result = self.analyst.execute(self.context)
        
        if not result.success:
            logger.error(f"Analyst échoué: {result.errors}")
            self._errors.extend(result.errors)
            # Continuer quand même - l'analyse n'est pas bloquante
            self.context.analyst_report = "Analyse non disponible"
        else:
            # Stocker le rapport
            self.context.analyst_report = result.content
            
            # Vérifier si on doit continuer l'optimisation
            proceed = result.data.get("proceed_to_optimization", True)
            if not proceed:
                logger.info("Analyst recommande de ne pas optimiser")
                self.state_machine.transition_to(AgentState.VALIDATE)
                return
        
        # Transition vers PROPOSE
        self.state_machine.transition_to(AgentState.PROPOSE)
    
    def _handle_propose(self) -> None:
        """Gère l'état PROPOSE - Exécution de l'Agent Strategist."""
        logger.info("Phase PROPOSE: Exécution Agent Strategist")
        
        # Exécuter le Strategist
        result = self.strategist.execute(self.context)
        
        if not result.success:
            logger.error(f"Strategist échoué: {result.errors}")
            self._errors.extend(result.errors)
            # Sans propositions, on va directement à la validation
            self.context.strategist_proposals = []
            self.state_machine.transition_to(AgentState.VALIDATE)
            return
        
        # Stocker les propositions
        proposals = result.data.get("proposals", [])
        self.context.strategist_proposals = proposals[:self.config.max_proposals_per_iteration]
        
        logger.info(f"Strategist: {len(self.context.strategist_proposals)} propositions générées")
        
        # Transition vers CRITIQUE
        self.state_machine.transition_to(AgentState.CRITIQUE)
    
    def _handle_critique(self) -> None:
        """Gère l'état CRITIQUE - Exécution de l'Agent Critic."""
        logger.info("Phase CRITIQUE: Exécution Agent Critic")
        
        if not self.context.strategist_proposals:
            logger.warning("Aucune proposition à critiquer")
            self.state_machine.transition_to(AgentState.VALIDATE)
            return
        
        # Exécuter le Critic
        result = self.critic.execute(self.context)
        
        if not result.success:
            logger.error(f"Critic échoué: {result.errors}")
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
        logger.info("Phase VALIDATE: Exécution Agent Validator")
        
        # Exécuter le Validator
        result = self.validator.execute(self.context)
        
        if not result.success:
            logger.error(f"Validator échoué: {result.errors}")
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
        
        if not self.config.data_path:
            errors.append("data_path requis")
        elif not Path(self.config.data_path).exists():
            errors.append(f"data_path n'existe pas: {self.config.data_path}")
        
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
        
        if self.config.on_backtest_needed:
            try:
                result = self.config.on_backtest_needed(params)
                if result:
                    return MetricsSnapshot.from_dict(result)
            except Exception as e:
                logger.error(f"Erreur backtest: {e}")
                self._warnings.append(f"Backtest échoué: {e}")
        
        return None
    
    def _test_proposals(self) -> None:
        """Teste les propositions approuvées via backtest."""
        for proposal in self.context.strategist_proposals:
            params = proposal.get("parameters", {})
            if not params:
                continue
            
            logger.info(f"Test proposition {proposal.get('id')}: {proposal.get('name')}")
            
            metrics = self._run_backtest(params)
            if metrics:
                proposal["tested_metrics"] = metrics.to_dict()
                proposal["tested"] = True
                
                logger.info(
                    f"  Résultat: Sharpe={metrics.sharpe_ratio:.3f}, "
                    f"Return={metrics.total_return:.2%}"
                )
            else:
                proposal["tested"] = False
    
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
        
        # Statistiques LLM
        total_tokens = sum([
            self.analyst.stats["total_tokens"],
            self.strategist.stats["total_tokens"],
            self.critic.stats["total_tokens"],
            self.validator.stats["total_tokens"],
        ])
        total_calls = sum([
            self.analyst.stats["execution_count"],
            self.strategist.stats["execution_count"],
            self.critic.stats["execution_count"],
            self.validator.stats["execution_count"],
        ])
        
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
