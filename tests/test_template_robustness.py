import pytest

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
