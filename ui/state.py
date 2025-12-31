"""
Module-ID: ui.state

Purpose: Définit les structures de données pour l'état de l'interface utilisateur.

Role in pipeline: state management

Key components: SidebarState

Inputs: Paramètres utilisateur

Outputs: État structuré

Dependencies: dataclasses

Conventions: État immutable via dataclass

Read-if: Gestion d'état UI

Skip-if: Logique métier pure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SidebarState:
    debug_enabled: bool
    symbol: str
    timeframe: str
    use_date_filter: bool
    start_date: Optional[object]
    end_date: Optional[object]
    available_tokens: List[str]
    available_timeframes: List[str]
    strategy_key: str
    strategy_name: str
    strategy_info: Any
    strategy_instance: Any
    params: Dict[str, Any]
    param_ranges: Dict[str, Any]
    param_specs: Dict[str, Any]
    active_indicators: List[str]
    optimization_mode: str
    max_combos: int
    n_workers: int
    llm_config: Any
    llm_model: Optional[str]
    llm_use_multi_agent: bool
    role_model_config: Any
    llm_max_iterations: int
    llm_use_walk_forward: bool
    llm_unload_during_backtest: bool
    llm_compare_enabled: bool
    llm_compare_auto_run: bool
    llm_compare_strategies: List[str]
    llm_compare_tokens: List[str]
    llm_compare_timeframes: List[str]
    llm_compare_metric: str
    llm_compare_aggregate: str
    llm_compare_max_runs: int
    llm_compare_use_preset: bool
    llm_compare_generate_report: bool
    initial_capital: float
    leverage: Any
