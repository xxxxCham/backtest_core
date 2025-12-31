"""
Module-ID: tests.test_template_robustness

Purpose: Tester robustesse templates Jinja2 (fallback N/A si champs manquent, pas de TypeError).

Role in pipeline: testing

Key components: test_format_float_handles_missing, test_critic_template_renders_without_walk_forward

Inputs: Template string/name, context dict (optionnel avec fields manquants)

Outputs: Rendered HTML string (pas d'exception)

Dependencies: pytest, utils.template

Conventions: Fallback N/A pour variables manquantes; pas d'erreurs rendu.

Read-if: Modification templates Jinja2 ou filters.

Skip-if: Tests templates non critiques.
"""

from utils.template import render_prompt, render_prompt_from_string


def test_format_float_handles_missing_variable() -> None:
    rendered = render_prompt_from_string("Value={{ missing|format_float(2) }}", {})
    assert "N/A" in rendered


def test_critic_template_renders_without_walk_forward_fields() -> None:
    # Simule le cas qui crashait: overfitting_ratio present (>0) mais classic_ratio absent du contexte.
    # Le rendu doit rester robuste (N/A) au lieu de lever TypeError.
    prompt = render_prompt(
        "critic.jinja2",
        {
            "strategy_name": "bollinger_atr",
            "iteration": 0,
            "current_metrics": None,
            "overfitting_ratio": 1.2,
            "analyst_report": "",
            "strategist_proposals": [],
            "current_params": {},
            "param_specs": [],
            "min_sharpe": 1.0,
            "min_trades": 30,
            "max_drawdown_limit": 0.2,
            "max_overfitting_ratio": 1.5,
            "iteration_history": [],
        },
    )

    assert "WALK-FORWARD VALIDATION" in prompt
    assert "Classic Ratio" in prompt


def test_validator_template_renders_with_walk_forward_fields() -> None:
    """Vérifie que le template validator.jinja2 rend correctement avec toutes les variables walk-forward."""
    prompt = render_prompt(
        "validator.jinja2",
        {
            "strategy_name": "bollinger_atr",
            "iteration": 2,
            "objective_check": {
                "sharpe_meets_minimum": True,
                "drawdown_within_limit": True,
                "overfitting_acceptable": True,
                "sufficient_trades": True,
                "critic_approved": True,
            },
            "current_metrics": {
                "sharpe_ratio": 1.5,
                "sortino_ratio": 1.8,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "sqn": 2.1,
                "total_trades": 50,
            },
            "min_sharpe": 1.0,
            "max_drawdown_limit": 0.2,
            "min_trades": 30,
            "overfitting_ratio": 1.3,
            "max_overfitting_ratio": 1.5,
            "classic_ratio": 1.2,
            "degradation_pct": 15.0,
            "test_stability_std": 0.3,
            "n_valid_folds": 5,
            "walk_forward_windows": 5,
            "analyst_report": "Test report",
            "strategist_proposals": [],
            "critic_concerns": [],
            "iteration_history": [],
            "best_metrics": None,
            "current_params": {},
        },
    )

    assert "WALK-FORWARD VALIDATION" in prompt
    assert "Degradation %" in prompt
    assert "Test Stability Std" in prompt
    assert "Valid Folds" in prompt
