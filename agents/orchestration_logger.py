"""
Logging d'Orchestration LLM
============================

Système de logging structuré pour tracer toutes les actions des agents LLM
pendant l'optimisation autonome.
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.log import get_logger

logger = get_logger(__name__)


class OrchestrationActionType(Enum):
    """Types d'actions d'orchestration."""

    # Analyse
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    RESULT_EVALUATION = "result_evaluation"

    # Stratégie
    STRATEGY_SELECTION = "strategy_selection"
    STRATEGY_MODIFICATION = "strategy_modification"
    STRATEGY_VALIDATION = "strategy_validation"

    # Indicateurs
    INDICATOR_VALUES_CHANGE = "indicator_values_change"
    INDICATOR_ADD = "indicator_add"
    INDICATOR_REMOVE = "indicator_remove"
    INDICATOR_VALIDATION = "indicator_validation"

    # Tests
    BACKTEST_LAUNCH = "backtest_launch"
    BACKTEST_COMPLETE = "backtest_complete"
    BACKTEST_FAILED = "backtest_failed"

    # Décisions
    DECISION_CONTINUE = "decision_continue"
    DECISION_STOP = "decision_stop"
    DECISION_CHANGE_APPROACH = "decision_change_approach"

    # Agents
    AGENT_ANALYST_ACTION = "agent_analyst"
    AGENT_STRATEGIST_ACTION = "agent_strategist"
    AGENT_CRITIC_ACTION = "agent_critic"
    AGENT_VALIDATOR_ACTION = "agent_validator"


class OrchestrationStatus(Enum):
    """Statuts d'une action d'orchestration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass
class OrchestrationLogEntry:
    """Entrée de log d'orchestration."""

    timestamp: str
    action_type: OrchestrationActionType
    agent: Optional[str] = None  # Analyst, Strategist, Critic, Validator
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    details: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type.value,
            "agent": self.agent,
            "status": self.status.value,
            "details": self.details,
            "iteration": self.iteration,
            "session_id": self.session_id,
        }

    def format_for_ui(self) -> str:
        """Formate pour affichage UI."""
        emoji = self._get_emoji()
        agent_str = f"[{self.agent}]" if self.agent else ""
        status_str = self._get_status_str()

        return f"{emoji} {agent_str} {self.action_type.value}: {status_str}"

    def _get_emoji(self) -> str:
        """Retourne l'emoji approprié."""
        if self.status == OrchestrationStatus.COMPLETED:
            return "✅"
        elif self.status == OrchestrationStatus.FAILED:
            return "❌"
        elif self.status == OrchestrationStatus.VALIDATED:
            return "✔️"
        elif self.status == OrchestrationStatus.REJECTED:
            return "❌"
        elif self.status == OrchestrationStatus.IN_PROGRESS:
            return "⏳"
        else:
            return "⏹️"

    def _get_status_str(self) -> str:
        """Retourne une chaîne de statut formatée."""
        if self.details:
            # Formater les détails importants
            if "strategy" in self.details:
                return f"Strategy={self.details['strategy']}"
            elif "indicator" in self.details:
                return f"Indicator={self.details['indicator']}"
            elif "params" in self.details:
                param_str = str(self.details['params'])[:50]
                return f"Params={param_str}..."
            elif "result" in self.details:
                result = self.details['result']
                if isinstance(result, dict) and 'sharpe' in result:
                    return f"Sharpe={result['sharpe']:.2f}"
        return self.status.value


class OrchestrationLogger:
    """Logger centralisé pour l'orchestration LLM."""

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize le logger d'orchestration.

        Args:
            session_id: ID unique de session (généré auto si None)
        """
        self.session_id = session_id or self._generate_session_id()
        self.logs: List[OrchestrationLogEntry] = []
        self.current_iteration = 0

    def _generate_session_id(self) -> str:
        """Génère un ID de session unique."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _now(self) -> str:
        """Timestamp actuel."""
        return datetime.now().isoformat()

    def log_analysis_start(self, agent: str, details: Optional[Dict] = None):
        """Log le début d'une analyse."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.ANALYSIS_START,
            agent=agent,
            status=OrchestrationStatus.IN_PROGRESS,
            details=details or {},
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Analysis started - Iteration {self.current_iteration}")

    def log_analysis_complete(
        self,
        agent: str,
        results: Dict[str, Any],
        status: OrchestrationStatus = OrchestrationStatus.COMPLETED
    ):
        """Log la fin d'une analyse."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.ANALYSIS_COMPLETE,
            agent=agent,
            status=status,
            details={"results": results},
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Analysis complete - Status: {status.value}")

    def log_strategy_selection(
        self,
        agent: str,
        strategy_name: str,
        reason: str
    ):
        """Log la sélection d'une stratégie."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.STRATEGY_SELECTION,
            agent=agent,
            status=OrchestrationStatus.COMPLETED,
            details={"strategy": strategy_name, "reason": reason},
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Strategy selected: {strategy_name} - {reason}")

    def log_strategy_modification(
        self,
        agent: str,
        old_strategy: str,
        new_strategy: str,
        reason: str,
        status: OrchestrationStatus = OrchestrationStatus.PENDING
    ):
        """Log une modification de stratégie."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.STRATEGY_MODIFICATION,
            agent=agent,
            status=status,
            details={
                "old_strategy": old_strategy,
                "new_strategy": new_strategy,
                "reason": reason
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Strategy changed: {old_strategy} → {new_strategy}")

    def log_indicator_values_change(
        self,
        agent: str,
        indicator: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        reason: str
    ):
        """Log un changement de valeurs d'indicateurs."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.INDICATOR_VALUES_CHANGE,
            agent=agent,
            status=OrchestrationStatus.COMPLETED,
            details={
                "indicator": indicator,
                "old_values": old_values,
                "new_values": new_values,
                "reason": reason
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Indicator {indicator} values changed")

    def log_indicator_add(
        self,
        agent: str,
        indicator: str,
        params: Dict[str, Any],
        reason: str,
        status: OrchestrationStatus = OrchestrationStatus.PENDING
    ):
        """Log l'ajout d'un nouvel indicateur."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.INDICATOR_ADD,
            agent=agent,
            status=status,
            details={
                "indicator": indicator,
                "params": params,
                "reason": reason
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] New indicator proposed: {indicator}")

    def log_indicator_validation(
        self,
        agent: str,
        indicator: str,
        is_valid: bool,
        message: str
    ):
        """Log la validation d'un indicateur."""
        status = OrchestrationStatus.VALIDATED if is_valid else OrchestrationStatus.REJECTED
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.INDICATOR_VALIDATION,
            agent=agent,
            status=status,
            details={
                "indicator": indicator,
                "is_valid": is_valid,
                "message": message
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Indicator {indicator} validation: {is_valid}")

    def log_backtest_launch(
        self,
        agent: str,
        params: Dict[str, Any],
        combination_id: int,
        total_combinations: int
    ):
        """Log le lancement d'un backtest."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.BACKTEST_LAUNCH,
            agent=agent,
            status=OrchestrationStatus.IN_PROGRESS,
            details={
                "params": params,
                "combination_id": combination_id,
                "total_combinations": total_combinations
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Backtest launched: {combination_id}/{total_combinations}")

    def log_backtest_complete(
        self,
        agent: str,
        params: Dict[str, Any],
        results: Dict[str, Any],
        combination_id: int
    ):
        """Log la fin d'un backtest."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.BACKTEST_COMPLETE,
            agent=agent,
            status=OrchestrationStatus.COMPLETED,
            details={
                "params": params,
                "results": results,
                "combination_id": combination_id
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)

        # Extraire métriques clés
        pnl = results.get('pnl', 0)
        sharpe = results.get('sharpe', 0)
        logger.info(f"[{agent}] Backtest #{combination_id} complete - PnL: {pnl:.2f}, Sharpe: {sharpe:.2f}")

    def log_backtest_failed(
        self,
        agent: str,
        params: Dict[str, Any],
        error: str,
        combination_id: int
    ):
        """Log l'échec d'un backtest."""
        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=OrchestrationActionType.BACKTEST_FAILED,
            agent=agent,
            status=OrchestrationStatus.FAILED,
            details={
                "params": params,
                "error": error,
                "combination_id": combination_id
            },
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.error(f"[{agent}] Backtest #{combination_id} failed: {error}")

    def log_decision(
        self,
        agent: str,
        decision_type: str,  # "continue", "stop", "change_approach"
        reason: str,
        details: Optional[Dict] = None
    ):
        """Log une décision d'agent."""
        action_map = {
            "continue": OrchestrationActionType.DECISION_CONTINUE,
            "stop": OrchestrationActionType.DECISION_STOP,
            "change_approach": OrchestrationActionType.DECISION_CHANGE_APPROACH,
        }

        entry = OrchestrationLogEntry(
            timestamp=self._now(),
            action_type=action_map.get(decision_type, OrchestrationActionType.DECISION_CONTINUE),
            agent=agent,
            status=OrchestrationStatus.COMPLETED,
            details={"reason": reason, **(details or {})},
            iteration=self.current_iteration,
            session_id=self.session_id
        )
        self.logs.append(entry)
        logger.info(f"[{agent}] Decision: {decision_type} - {reason}")

    def next_iteration(self):
        """Passe à l'itération suivante."""
        self.current_iteration += 1
        logger.info(f"=== Iteration {self.current_iteration} START ===")

    def get_logs_for_iteration(self, iteration: int) -> List[OrchestrationLogEntry]:
        """Récupère les logs d'une itération spécifique."""
        return [log for log in self.logs if log.iteration == iteration]

    def get_logs_by_agent(self, agent: str) -> List[OrchestrationLogEntry]:
        """Récupère tous les logs d'un agent spécifique."""
        return [log for log in self.logs if log.agent == agent]

    def get_logs_by_type(self, action_type: OrchestrationActionType) -> List[OrchestrationLogEntry]:
        """Récupère tous les logs d'un type d'action."""
        return [log for log in self.logs if log.action_type == action_type]

    def save_to_file(self, filepath: Optional[Path] = None):
        """Sauvegarde les logs dans un fichier JSON."""
        if filepath is None:
            filepath = Path(f"orchestration_logs_{self.session_id}.json")

        data = {
            "session_id": self.session_id,
            "total_iterations": self.current_iteration,
            "total_logs": len(self.logs),
            "logs": [log.to_dict() for log in self.logs]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Logs saved to {filepath}")

    def generate_summary(self) -> str:
        """Génère un résumé textuel des logs."""
        lines = ["=" * 80]
        lines.append(f"ORCHESTRATION LOG SUMMARY - Session: {self.session_id}")
        lines.append("=" * 80)
        lines.append(f"Total Iterations: {self.current_iteration}")
        lines.append(f"Total Log Entries: {len(self.logs)}")
        lines.append("")

        # Compter par type d'action
        action_counts = {}
        for log in self.logs:
            action_type = log.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        lines.append("Actions Count:")
        for action, count in sorted(action_counts.items()):
            lines.append(f"  - {action}: {count}")
        lines.append("")

        # Compter par agent
        agent_counts = {}
        for log in self.logs:
            if log.agent:
                agent_counts[log.agent] = agent_counts.get(log.agent, 0) + 1

        lines.append("Agent Activity:")
        for agent, count in sorted(agent_counts.items()):
            lines.append(f"  - {agent}: {count} actions")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


# Instance globale (optionnelle)
_global_logger: Optional[OrchestrationLogger] = None


def generate_session_id() -> str:
    """Génère un ID de session unique."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_orchestration_logger(session_id: Optional[str] = None) -> OrchestrationLogger:
    """Récupère ou crée le logger d'orchestration global."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OrchestrationLogger(session_id=session_id)
    return _global_logger


def reset_orchestration_logger():
    """Réinitialise le logger d'orchestration global."""
    global _global_logger
    _global_logger = None


__all__ = [
    "OrchestrationActionType",
    "OrchestrationStatus",
    "OrchestrationLogEntry",
    "OrchestrationLogger",
    "generate_session_id",
    "get_orchestration_logger",
    "reset_orchestration_logger",
]
