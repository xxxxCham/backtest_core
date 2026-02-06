"""
Module-ID: agents.critic

Purpose: Évaluer critiquement les propositions pour détecter overfitting et risques cachés.

Role in pipeline: orchestration

Key components: CriticAgent, CriticEvaluation, CriticResponse

Inputs: AgentContext (proposals du Strategist, walk-forward metrics si dispos)

Outputs: CriticResponse (évals par proposition, concerns consolidés, propositions approuvées)

Dependencies: agents.base_agent, utils.template, backtest.validation (walk-forward)

Conventions: Ratios overfitting calculés à partir de walk-forward si dispos; concern_severity (LOW/MEDIUM/HIGH/CRITICAL); template Jinja2.

Read-if: Modification logique critique, seuils overfitting, ou intégration walk-forward.

Skip-if: Vous ne modifiez que analyze/propose/validate.
"""

from __future__ import annotations

# pylint: disable=logging-fstring-interpolation
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


class CriticAgent(BaseAgent):
    """
    Agent Critic - Expert en détection des risques.

    Évalue:
    - Risque d'overfitting pour chaque proposition
    - Cohérence des changements proposés
    - Risques cachés ou non évidents
    - Faisabilité et robustesse
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.CRITIC

    @property
    def system_prompt(self) -> str:
        return """You are a senior risk analyst and trading strategy auditor with a skeptical mindset.
Your role is to critically evaluate optimization proposals and identify potential issues.

NEW WALK-FORWARD METRICS AVAILABLE:
- classic_ratio: average train Sharpe / average test Sharpe
- overfitting_ratio: classic_ratio + penalty for test instability (higher = worse)
- degradation_pct: % drop from train to test performance (0% = perfect forward)
- test_stability_std: standard deviation of test Sharpe across folds (lower = stable)
- n_valid_folds: number of successful out-of-sample tests

RED FLAGS (reject or flag heavily):
- overfitting_ratio > 1.8
- degradation_pct > 40%
- test_stability_std > 0.5
- n_valid_folds < 4

ADDITIONAL RED FLAGS:
- Very specific parameter values (e.g., 17.3 instead of 15 or 20)
- Large improvements with small changes
- Parameters at extreme bounds
- Inconsistent with analyst findings
- Overly complex parameter interactions

SCORING GUIDELINES:
- overfitting_score: 0-100 (0=no risk, 100=certain overfitting)
- robustness_score: 0-100 (0=fragile, 100=very robust)
- recommendation: APPROVE|MODIFY|REJECT

When evaluating proposals:
1. Prioritize walk-forward metrics - they are the strongest indicator of overfitting
2. Be skeptical but fair - look for real problems, not imaginary ones
3. Consider if changes could be data-mined coincidences
4. Check if the proposal addresses the real weakness or just symptoms
5. Evaluate if the expected improvement is realistic
6. Consider edge cases and regime changes

Respond ONLY in valid JSON format with this exact structure:
{
    "overall_assessment": "Brief critical summary",
    "walk_forward_summary": "Specific assessment of out-of-sample stability and degradation",
    "market_regime_concerns": ["concern1", "concern2"],
    "statistical_concerns": ["concern1", "concern2"],
    "proposal_evaluations": [
        {
            "proposal_id": 1,
            "overfitting_score": 0-100,
            "robustness_score": 0-100,
            "recommendation": "APPROVE|MODIFY|REJECT",
            "critical_issues": ["issue1", "issue2"],
            "warnings": ["warning1", "warning2"],
            "suggested_modifications": ["modification1"],
            "reasoning": "Detailed reasoning for the evaluation"
        }
    ],
    "approved_proposals": [1, 2],
    "rejected_proposals": [3],
    "best_proposal_id": 1,
    "proceed_with_testing": true/false,
    "final_concerns": ["Any remaining concerns to flag"]
}"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Évalue critiquement les propositions.

        Args:
            context: Contexte avec propositions du Strategist

        Returns:
            Évaluation critique
        """
        start_time = time.time()

        # Vérifier qu'il y a des propositions
        if not context.strategist_proposals:
            return AgentResult.failure_result(
                self.role,
                "Aucune proposition à évaluer",
                execution_time_ms=0,
            )

        # Construire le prompt
        user_prompt = self._build_critique_prompt(context)

        # Appeler le LLM
        response = self._call_llm(user_prompt, json_mode=True, temperature=0.3)

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
        critique = response.parse_json()
        if critique is None:
            return AgentResult.failure_result(
                self.role,
                f"Échec parsing JSON: {response.parse_error}",
                execution_time_ms=execution_time,
                raw_llm_response=response,
            )

        # Valider la structure
        validation_errors = self._validate_critique(critique)
        if validation_errors:
            logger.warning(f"Critique partiellement invalide: {validation_errors}")

        # Extraire les informations clés
        approved = critique.get("approved_proposals", [])
        rejected = critique.get("rejected_proposals", [])
        best_id = critique.get("best_proposal_id")

        # Filtrer les propositions approuvées
        approved_proposals = []
        for prop in context.strategist_proposals:
            prop_id = prop.get("id")
            if prop_id in approved:
                # Trouver l'évaluation correspondante
                eval_data = next(
                    (e for e in critique.get("proposal_evaluations", [])
                     if e.get("proposal_id") == prop_id),
                    {}
                )
                prop["critic_evaluation"] = eval_data
                prop["is_best"] = (prop_id == best_id)
                approved_proposals.append(prop)

        # Collecter les concerns
        concerns = (
            critique.get("market_regime_concerns", []) +
            critique.get("statistical_concerns", []) +
            critique.get("final_concerns", [])
        )

        return AgentResult.success_result(
            self.role,
            content=critique.get("overall_assessment", ""),
            data={
                "critique": critique,
                "approved_proposals": approved_proposals,
                "rejected_count": len(rejected),
                "best_proposal_id": best_id,
                "proceed_with_testing": critique.get("proceed_with_testing", False),
                "concerns": concerns,
                "proposal_evaluations": critique.get("proposal_evaluations", []),
            },
            execution_time_ms=execution_time,
            tokens_used=response.total_tokens,
            llm_calls=1,
            raw_llm_response=response,
        )

    def _build_critique_prompt(self, context: AgentContext) -> str:
        """Construit le prompt de critique via template Jinja2."""

        # Convertir MetricsSnapshot en dict pour le template
        current_metrics_dict = None
        if context.current_metrics:
            current_metrics_dict = context.current_metrics.to_dict()

        template_context = {
            "strategy_name": context.strategy_name,
            "strategy_description": context.strategy_description,
            "iteration": context.iteration,
            "comparison_context": context.comparison_context,
            "current_metrics": current_metrics_dict,
            "overfitting_ratio": context.overfitting_ratio,
            "classic_ratio": getattr(context, "classic_ratio", None),
            "degradation_pct": getattr(context, "degradation_pct", None),
            "test_stability_std": getattr(context, "test_stability_std", None),
            "n_valid_folds": getattr(context, "n_valid_folds", None),
            "walk_forward_windows": getattr(context, "walk_forward_windows", None),
            "data_rows": getattr(context, "data_rows", None),
            "data_date_range": getattr(context, "data_date_range", None),
            "analyst_report": context.analyst_report,
            "strategist_proposals": context.strategist_proposals,
            "current_params": context.current_params,
            "param_specs": context.param_specs,
            "min_sharpe": context.min_sharpe,
            "min_trades": context.min_trades,
            "max_drawdown_limit": context.max_drawdown_limit,
            "max_overfitting_ratio": context.max_overfitting_ratio,
            "iteration_history": context.iteration_history,
            "memory_summary": context.memory_summary,
            "strategy_indicators_context": context.strategy_indicators_context,
            "readonly_indicators_context": context.readonly_indicators_context,
            "indicator_context_warnings": context.indicator_context_warnings,
        }

        return render_prompt("critic.jinja2", template_context)

    def _validate_critique(self, critique: Dict[str, Any]) -> List[str]:
        """Valide la structure de la critique."""
        errors = []

        required_fields = [
            "overall_assessment",
            "proposal_evaluations",
            "proceed_with_testing",
        ]

        for field in required_fields:
            if field not in critique:
                errors.append(f"Champ manquant: {field}")

        # Valider les évaluations
        evaluations = critique.get("proposal_evaluations", [])
        for eval_data in evaluations:
            if "proposal_id" not in eval_data:
                errors.append("proposal_id manquant dans évaluation")
            if "recommendation" not in eval_data:
                errors.append("recommendation manquante dans évaluation")
            elif eval_data["recommendation"] not in ["APPROVE", "MODIFY", "REJECT"]:
                errors.append(f"recommendation invalide: {eval_data['recommendation']}")

        return errors
