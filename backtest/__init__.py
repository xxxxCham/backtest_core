"""
Module-ID: backtest

Purpose: Moteur de backtesting (pipeline complet + optimisation + validation) avec API stable pour UI/agents.

Role in pipeline: core / orchestration

Key components: BacktestEngine, RunResult, OptunaOptimizer, SweepEngine, WalkForwardValidator, BackendFacade

Inputs: Données OHLCV, stratégies, paramètres, configurations

Outputs: Résultats de backtest structurés (trades, métriques, rapports)

Dependencies: Tous les modules backtest/*, strategies, indicators, data, utils, optionnel: optuna, tabulate

Conventions: Pipeline données→indicateurs→signaux→trades→métriques; erreurs structurées; réponses typées.

Read-if: Vous importez le moteur depuis UI/agents ou modifiez l'architecture.

Skip-if: Vous travaillez sur une seule strat/indicateur.
"""

from .engine import BacktestEngine, RunResult
from .performance import PerformanceCalculator, calculate_metrics
from .simulator import Trade, simulate_trades
from .errors import (
    BacktestError,
    UserInputError,
    DataError,
    BackendInternalError,
    LLMUnavailableError,
    StrategyNotFoundError,
    ParameterValidationError,
)
from .facade import (
    BackendFacade,
    BacktestRequest,
    GridOptimizationRequest,
    LLMOptimizationRequest,
    BackendResponse,
    GridOptimizationResponse,
    LLMOptimizationResponse,
    ResponseStatus,
    ErrorCode,
    UIMetrics,
    UIPayload,
    ErrorInfo,
    get_facade,
    to_ui_payload,
)
from .storage import (
    ResultStorage,
    StoredResultMetadata,
    get_storage,
)

# Import conditionnel Optuna (peut ne pas être installé)
try:
    from .optuna_optimizer import (
        OptunaOptimizer,
        ParamSpec,
        OptimizationResult,
        MultiObjectiveResult,
        quick_optimize,
        suggest_param_space,
        OPTUNA_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None
    ParamSpec = None
    OptimizationResult = None
    MultiObjectiveResult = None
    quick_optimize = None
    suggest_param_space = None

__all__ = [
    # Engine
    "BacktestEngine",
    "RunResult",
    # Performance
    "PerformanceCalculator",
    "calculate_metrics",
    # Simulator
    "simulate_trades",
    "Trade",
    # Errors
    "BacktestError",
    "UserInputError",
    "DataError",
    "BackendInternalError",
    "LLMUnavailableError",
    "StrategyNotFoundError",
    "ParameterValidationError",
    # Facade
    "BackendFacade",
    "BacktestRequest",
    "GridOptimizationRequest",
    "LLMOptimizationRequest",
    "BackendResponse",
    "GridOptimizationResponse",
    "LLMOptimizationResponse",
    "ResponseStatus",
    "ErrorCode",
    "UIMetrics",
    "UIPayload",
    "ErrorInfo",
    "get_facade",
    "to_ui_payload",
    # Storage
    "ResultStorage",
    "StoredResultMetadata",
    "get_storage",
    # Optuna (conditional)
    "OPTUNA_AVAILABLE",
    "OptunaOptimizer",
    "ParamSpec",
    "OptimizationResult",
    "MultiObjectiveResult",
    "quick_optimize",
    "suggest_param_space",
]
