"""
Backtest Core - Agents LLM Module
=================================

Système d'optimisation autonome par agents LLM.

Architecture:
- Orchestrator: Machine à états pilotant le workflow
- 4 Agents spécialisés: Analyst, Strategist, Critic, Validator
- AutonomousStrategist: Agent autonome avec capacité de lancer des backtests
- BacktestExecutor: Interface pour exécuter des backtests depuis les agents
- State Machine: Transitions validées à chaque étape

Workflows:

1. Mode Orchestré (analyse statique):
    INIT → ANALYZE → PROPOSE → CRITIQUE → VALIDATE → [APPROVE/REJECT/ITERATE]

2. Mode Autonome (avec backtests réels):
    >>> strategist.optimize(executor, params, bounds) → OptimizationSession

Usage Mode Autonome (RECOMMANDÉ):
    >>> from agents import create_autonomous_optimizer
    >>> from agents.llm_client import LLMConfig, LLMProvider
    >>> 
    >>> config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
    >>> strategist, executor = create_autonomous_optimizer(
    ...     llm_config=config,
    ...     backtest_fn=my_backtest_function,
    ...     strategy_name="ema_cross",
    ...     data=ohlcv_df,
    ... )
    >>> 
    >>> session = strategist.optimize(
    ...     executor=executor,
    ...     initial_params={"fast": 10, "slow": 21},
    ...     param_bounds={"fast": (5, 20), "slow": (15, 50)},
    ...     max_iterations=10,
    ... )
    >>> print(f"Best: {session.best_result.sharpe_ratio}")

Usage Mode Orchestré (analyse sans exécution):
    >>> from agents import Orchestrator, OrchestratorConfig
    >>> 
    >>> config = OrchestratorConfig(
    ...     strategy_name="ema_cross",
    ...     data_path="data/BTCUSDT_1h.parquet",
    ...     max_iterations=10,
    ... )
    >>> orchestrator = Orchestrator(config)
    >>> result = orchestrator.run()
"""

from .state_machine import (
    AgentState,
    StateTransition,
    StateMachine,
    ValidationResult,
)
from .llm_client import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    LLMProvider,
    OllamaClient,
    OpenAIClient,
    create_llm_client,
)
from .base_agent import BaseAgent, AgentContext, AgentResult
from .analyst import AnalystAgent
from .strategist import StrategistAgent
from .critic import CriticAgent
from .validator import ValidatorAgent
from .orchestrator import Orchestrator, OrchestratorConfig, OrchestratorResult
from .backtest_executor import (
    BacktestExecutor,
    BacktestRequest,
    BacktestResult,
    ExperimentHistory,
)
from .autonomous_strategist import (
    AutonomousStrategist,
    IterationDecision,
    OptimizationSession,
    create_autonomous_optimizer,
)
from .integration import (
    run_backtest_for_agent,
    run_walk_forward_for_agent,
    create_optimizer_from_engine,
    get_strategy_param_bounds,
    get_strategy_param_space,
    quick_optimize,
    create_orchestrator_with_backtest,
)
from .ollama_manager import (
    ensure_ollama_running,
    unload_model,
    cleanup_all_models,
    list_ollama_models,
    is_ollama_available,
    prepare_for_llm_run,
    # GPU Memory Management
    GPUMemoryManager,
    LLMMemoryState,
    gpu_compute_context,
)
from .model_config import (
    ModelCategory,
    ModelInfo,
    RoleModelAssignment,
    RoleModelConfig,
    KNOWN_MODELS,
    list_available_models,
    get_models_by_category,
    get_global_model_config,
    set_global_model_config,
)

__all__ = [
    # State Machine
    "AgentState",
    "StateTransition",
    "StateMachine",
    "ValidationResult",
    # LLM
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    "OllamaClient",
    "OpenAIClient",
    "create_llm_client",
    # Agents
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AnalystAgent",
    "StrategistAgent",
    "CriticAgent",
    "ValidatorAgent",
    # Autonomous Mode (RECOMMENDED)
    "AutonomousStrategist",
    "BacktestExecutor",
    "BacktestRequest",
    "BacktestResult",
    "ExperimentHistory",
    "IterationDecision",
    "OptimizationSession",
    "create_autonomous_optimizer",
    # Integration (FULL STACK)
    "run_backtest_for_agent",
    "run_walk_forward_for_agent",
    "create_optimizer_from_engine",
    "get_strategy_param_bounds",
    "get_strategy_param_space",
    "quick_optimize",
    "create_orchestrator_with_backtest",
    # Orchestrator (Static Mode)
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    # Ollama Manager
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
    # Multi-Model Configuration
    "ModelCategory",
    "ModelInfo",
    "RoleModelAssignment",
    "RoleModelConfig",
    "KNOWN_MODELS",
    "list_available_models",
    "get_models_by_category",
    "get_global_model_config",
    "set_global_model_config",
]
