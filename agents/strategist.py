"""
Agent Strategist - Proposition de nouvelles configurations.

Rôle:
- Proposer des ajustements de paramètres
- Générer des combinaisons créatives mais réalistes
- Prioriser les changements par impact potentiel
- Justifier chaque proposition

Input: Context + Rapport Analyst
Output: Liste de propositions de paramètres ordonnées
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from utils.template import render_prompt

from .base_agent import (
    AgentContext,
    AgentResult,
    AgentRole,
    BaseAgent,
)

logger = logging.getLogger(__name__)


class StrategistAgent(BaseAgent):
    """
    Agent Strategist - Expert en optimisation de stratégies.

    Propose:
    - Ajustements de paramètres basés sur l'analyse
    - Combinaisons créatives mais réalistes
    - Prioritisation par impact/risque
    - Justifications détaillées
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.STRATEGIST

    @property
    def system_prompt(self) -> str:
        return """You are a senior quantitative strategist specializing in algorithmic trading strategy optimization.

Your expertise includes:
- Parameter optimization for trading strategies
- Understanding indicator behavior across different settings
- Balancing risk/reward trade-offs
- Creative but grounded strategy improvements
- Avoiding overfitting while maximizing performance

When proposing parameter changes:
1. Consider the analyst's findings and recommendations
2. Propose changes that address identified weaknesses
3. Stay within reasonable parameter bounds
4. Prioritize robustness over maximum performance
5. Consider how parameters interact with each other
6. Avoid drastic changes that might cause instability

IMPORTANT CONSTRAINTS:
- Each parameter must stay within its min/max bounds
- Propose 3-5 parameter sets, ordered by priority
- First proposal should be conservative, later ones more aggressive
- Always explain the rationale for each change

Respond ONLY in valid JSON format with this exact structure:
{
    "analysis_summary": "Brief summary of analyst findings you're addressing",
    "optimization_strategy": "Overall approach to optimization",
    "proposals": [
        {
            "id": 1,
            "name": "Conservative Adjustment",
            "priority": "HIGH|MEDIUM|LOW",
            "risk_level": "LOW|MEDIUM|HIGH",
            "parameters": {
                "param_name": value,
                ...
            },
            "changes_from_current": {
                "param_name": {"from": old_value, "to": new_value, "change_percent": X}
            },
            "rationale": "Why this configuration might improve performance",
            "expected_impact": {
                "sharpe_ratio": "+X% to +Y%",
                "drawdown": "similar|reduced|increased",
                "trade_frequency": "similar|higher|lower"
            },
            "risks": ["risk1", "risk2"]
        }
    ],
    "constraints_respected": true,
    "fallback_recommendation": "What to do if all proposals fail"
}"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Génère des propositions de paramètres.

        Args:
            context: Contexte avec rapport analyst

        Returns:
            Liste de propositions ordonnées
        """
        start_time = time.time()

        # Construire le prompt
        user_prompt = self._build_proposal_prompt(context)

        # Appeler le LLM
        response = self._call_llm(user_prompt, json_mode=True, temperature=0.7)

        execution_time = (time.time() - start_time) * 1000

        # Vérifier la réponse
        if not response.content:
            return AgentResult.failure_result(
                self.role,
                "LLM n'a pas retourné de réponse",
                execution_time_ms=execution_time,
                raw_llm_response=response,
            )

        # Parser le JSON
        proposals_data = response.parse_json()
        if proposals_data is None:
            return AgentResult.failure_result(
                self.role,
                f"Échec parsing JSON: {response.parse_error}",
                execution_time_ms=execution_time,
                raw_llm_response=response,
            )

        # Valider et corriger les propositions
        proposals = proposals_data.get("proposals", [])
        validated_proposals = self._validate_and_fix_proposals(proposals, context)

        if not validated_proposals:
            return AgentResult.failure_result(
                self.role,
                "Aucune proposition valide générée",
                execution_time_ms=execution_time,
                raw_llm_response=response,
            )

        # Créer le résultat
        return AgentResult.success_result(
            self.role,
            content=proposals_data.get("optimization_strategy", ""),
            data={
                "proposals": validated_proposals,
                "analysis_summary": proposals_data.get("analysis_summary", ""),
                "optimization_strategy": proposals_data.get("optimization_strategy", ""),
                "fallback_recommendation": proposals_data.get("fallback_recommendation", ""),
                "total_proposals": len(validated_proposals),
            },
            execution_time_ms=execution_time,
            tokens_used=response.total_tokens,
            llm_calls=1,
            raw_llm_response=response,
        )

    def _build_proposal_prompt(self, context: AgentContext) -> str:
        """Construit le prompt de proposition via template Jinja2."""

        # Convertir MetricsSnapshot en dict pour le template
        current_metrics_dict = None
        if context.current_metrics:
            current_metrics_dict = context.current_metrics.to_dict()

        best_metrics_dict = None
        if context.best_metrics:
            best_metrics_dict = context.best_metrics.to_dict()

        template_context = {
            "strategy_name": context.strategy_name,
            "iteration": context.iteration,
            "param_specs": context.param_specs,
            "current_params": context.current_params,
            "current_metrics": current_metrics_dict,
            "overfitting_ratio": context.overfitting_ratio,
            "max_overfitting_ratio": context.max_overfitting_ratio,
            "analyst_report": context.analyst_report,
            "best_metrics": best_metrics_dict,
            "best_params": context.best_params,
            "optimization_target": context.optimization_target,
            "min_sharpe": context.min_sharpe,
            "max_drawdown_limit": context.max_drawdown_limit,
            "min_trades": context.min_trades,
            "iteration_history": context.iteration_history,
            # Résumé des paramètres déjà testés dans cette session
            "session_params_summary": getattr(context, 'session_params_summary', None),
            "memory_summary": context.memory_summary,
        }

        return render_prompt("strategist.jinja2", template_context)

    def _validate_and_fix_proposals(
        self,
        proposals: List[Dict[str, Any]],
        context: AgentContext,
    ) -> List[Dict[str, Any]]:
        """
        Valide et corrige les propositions.

        S'assure que tous les paramètres respectent les contraintes.
        """
        validated = []

        # Créer un dict des specs pour lookup rapide
        specs_dict = {spec.name: spec for spec in context.param_specs}

        for proposal in proposals:
            params = proposal.get("parameters", {})
            fixed_params = {}

            # Vérifier chaque paramètre
            for param_name, value in params.items():
                if param_name not in specs_dict:
                    # Paramètre inconnu - ignorer
                    logger.warning(f"Paramètre inconnu ignoré: {param_name}")
                    continue

                spec = specs_dict[param_name]

                # Forcer les contraintes
                if spec.min_value is not None and value < spec.min_value:
                    logger.warning(
                        f"Paramètre {param_name}={value} < min={spec.min_value}, corrigé"
                    )
                    value = spec.min_value

                if spec.max_value is not None and value > spec.max_value:
                    logger.warning(
                        f"Paramètre {param_name}={value} > max={spec.max_value}, corrigé"
                    )
                    value = spec.max_value

                # Arrondir au step si spécifié
                if spec.step is not None and spec.step > 0:
                    if isinstance(value, float) and isinstance(spec.step, (int, float)):
                        value = round(value / spec.step) * spec.step
                    elif isinstance(value, int) and isinstance(spec.step, int):
                        value = (value // spec.step) * spec.step

                fixed_params[param_name] = value

            # Ajouter les paramètres manquants avec valeurs actuelles
            for param_name in specs_dict:
                if param_name not in fixed_params:
                    fixed_params[param_name] = context.current_params.get(
                        param_name,
                        specs_dict[param_name].current_value
                    )

            if fixed_params:
                proposal["parameters"] = fixed_params
                proposal["validated"] = True
                validated.append(proposal)

        return validated
