"""
Agents Integration - Pont entre les agents LLM et le moteur de backtest.

Ce module fournit:
1. run_backtest_for_agent() - Wrapper du BacktestEngine pour les agents
2. run_walk_forward_for_agent() - Wrapper du WalkForwardValidator
3. create_optimizer_from_engine() - Factory complète prête à l'emploi

Le but est de rendre le système autonome VRAIMENT fonctionnel en connectant
les composants abstraits aux implémentations concrètes du projet.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.validation import WalkForwardValidator
from strategies.base import get_strategy, list_strategies
from utils.config import Config
from utils.observability import (
    get_obs_logger,
    generate_run_id,
    trace_span,
)

from .backtest_executor import BacktestExecutor
from .autonomous_strategist import AutonomousStrategist, OptimizationSession
from .llm_client import LLMConfig, create_llm_client
from .model_config import RoleModelConfig

# Logger module-level (sans run_id spécifique)
_logger = get_obs_logger(__name__)


def run_backtest_for_agent(
    strategy_name: str,
    params: Dict[str, Any],
    data: pd.DataFrame,
    *,
    initial_capital: float = 10000.0,
    config: Optional[Config] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exécute un backtest et retourne les métriques pour un agent.

    C'est le pont entre BacktestExecutor et BacktestEngine.

    Args:
        strategy_name: Nom de la stratégie (ex: "ema_cross")
        params: Paramètres de la stratégie
        data: DataFrame OHLCV
        initial_capital: Capital de départ
        config: Configuration optionnelle
        run_id: Identifiant de corrélation (généré si None)

    Returns:
        Dict avec toutes les métriques nécessaires pour l'agent
    """
    # Générer run_id si absent
    run_id = run_id or generate_run_id()
    logger = get_obs_logger(__name__, run_id=run_id, strategy=strategy_name)

    logger.info(
        "agent_backtest_start params=%s bars=%s", params, len(data)
    )

    # Créer engine avec le même run_id pour corrélation
    engine = BacktestEngine(
        initial_capital=initial_capital, config=config, run_id=run_id
    )

    try:
        with trace_span(logger, "agent_backtest", strategy=strategy_name):
            result = engine.run(
                df=data,
                strategy=strategy_name,
                params=params,
            )

        # Extraire les métriques pour l'agent
        metrics = result.metrics.copy()

        total_return_pct = metrics.get("total_return_pct", 0)
        max_drawdown_pct = metrics.get("max_drawdown", 0)
        win_rate_pct = metrics.get("win_rate", 0)

        output = {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "total_return": total_return_pct / 100.0,
            "max_drawdown": max_drawdown_pct / 100.0,
            "win_rate": win_rate_pct / 100.0,
            "profit_factor": metrics.get("profit_factor", 0),
            "total_trades": metrics.get("total_trades", 0),
            "sqn": metrics.get("sqn", 0),
            "calmar_ratio": metrics.get("calmar_ratio", 0),
            "recovery_factor": metrics.get("recovery_factor", 0),
            # Données brutes pour analyse approfondie
            "equity_curve": (
                result.equity.tolist() if len(result.equity) < 10000 else None
            ),
            "trades": (
                result.trades.to_dict("records")
                if len(result.trades) < 1000 else None
            ),
            # Métadonnées de corrélation
            "run_id": run_id,
        }

        logger.info(
            "agent_backtest_end sharpe=%.2f trades=%s",
            output["sharpe_ratio"], output["total_trades"]
        )
        return output

    except Exception as e:
        logger.error("agent_backtest_error error=%s", str(e))
        raise


def run_walk_forward_for_agent(
    strategy_name: str,
    params: Dict[str, Any],
    data: pd.DataFrame,
    *,
    n_windows: int = 6,  # 6 fenêtres pour ~4 mois/test sur 24 mois
    train_ratio: float = 0.75,  # 75% train, 25% test (18m/6m optimal)
    initial_capital: float = 10000.0,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """
    Exécute une validation walk-forward et retourne les métriques.

    Configuration optimisée pour 2 ans de données :
    - 6 fenêtres → ~4 mois de test par fenêtre
    - 75% train → 18 mois d'entraînement, 6 mois de test
    - 2% embargo → évite le leakage entre train et test

    Args:
        strategy_name: Nom de la stratégie
        params: Paramètres de la stratégie
        data: DataFrame OHLCV
        n_windows: Nombre de fenêtres de validation
        train_ratio: Ratio train/total (1 - test_ratio)
        initial_capital: Capital de départ
        config: Configuration optionnelle

    Returns:
        Dict avec train_sharpe, test_sharpe, overfitting_ratio, métriques robustes
    """
    test_pct = 1.0 - train_ratio

    validator = WalkForwardValidator(
        n_folds=n_windows,
        test_pct=test_pct,
        embargo_pct=0.02,  # 2% d'embargo pour éviter le leakage
    )

    engine = BacktestEngine(initial_capital=initial_capital, config=config)

    folds = validator.split(data)

    train_sharpes = []
    test_sharpes = []

    for fold in folds:
        train_df, test_df = validator.get_data_splits(data, fold)

        try:
            # Backtest sur train
            train_result = engine.run(
                df=train_df,
                strategy=strategy_name,
                params=params,
            )
            train_sharpe = train_result.metrics.get("sharpe_ratio", 0)
            train_sharpes.append(train_sharpe)

            # Backtest sur test
            test_result = engine.run(
                df=test_df,
                strategy=strategy_name,
                params=params,
            )
            test_sharpe = test_result.metrics.get("sharpe_ratio", 0)
            test_sharpes.append(test_sharpe)

            # Stocker dans le fold
            fold.train_metrics = train_result.metrics
            fold.test_metrics = test_result.metrics

        except Exception as e:
            _logger.warning("fold_%s_failed error=%s", fold.fold_id, str(e))
            continue

    n_folds = len(train_sharpes)

    if n_folds == 0:
        return {
            "train_sharpe": 0.0,
            "test_sharpe": 0.0,
            "overfitting_ratio": 999.0,
            "classic_ratio": 999.0,
            "degradation_pct": 100.0,
            "test_stability_std": 0.0,
            "n_valid_folds": 0,
        }

    # Moyennes avec numpy
    avg_train = np.mean(train_sharpes)
    avg_test = np.mean(test_sharpes)
    std_test = np.std(test_sharpes)

    # Ratio classique avec garde-fou
    classic_ratio = avg_train / avg_test if avg_test > 1e-6 else 999.0

    # Dégradation % avec garde-fou
    degradation_pct = (avg_train - avg_test) / avg_train * 100 if avg_train > 1e-6 else 100.0
    degradation_pct = max(0.0, degradation_pct)

    # Ratio robuste = ratio classique + pénalité de stabilité
    stability_penalty = std_test * 2.0
    robust_ratio = classic_ratio + stability_penalty

    return {
        "train_sharpe": float(avg_train),
        "test_sharpe": float(avg_test),
        "overfitting_ratio": float(robust_ratio),
        "classic_ratio": float(classic_ratio),
        "degradation_pct": float(degradation_pct),
        "test_stability_std": float(std_test),
        "n_valid_folds": n_folds,
    }


def create_optimizer_from_engine(
    llm_config: LLMConfig,
    strategy_name: str,
    data: pd.DataFrame,
    *,
    initial_capital: float = 10000.0,
    config: Optional[Config] = None,
    use_walk_forward: bool = True,
    verbose: bool = True,
    unload_llm_during_backtest: Optional[bool] = None,
    orchestration_logger: Optional[Any] = None,
) -> Tuple[AutonomousStrategist, BacktestExecutor]:
    """
    Factory complète pour créer un optimiseur autonome connecté au vrai moteur.

    C'est LA fonction à utiliser pour une optimisation autonome fonctionnelle.

    Args:
        llm_config: Configuration du LLM (Ollama ou OpenAI)
        strategy_name: Nom de la stratégie à optimiser
        data: DataFrame OHLCV
        initial_capital: Capital de départ
        config: Configuration du backtest
        use_walk_forward: Activer validation walk-forward
        verbose: Logs détaillés
        unload_llm_during_backtest: Si True, décharge le LLM du GPU pendant les backtests
            pour libérer la VRAM. Si None, utilise la variable d'environnement
            UNLOAD_LLM_DURING_BACKTEST (défaut: False pour compatibilité CPU-only)
        orchestration_logger: Logger pour enregistrer les actions d'orchestration

    Returns:
        (AutonomousStrategist, BacktestExecutor) prêts à l'emploi

    Example:
        >>> from agents.integration import create_optimizer_from_engine
        >>> from agents.llm_client import LLMConfig, LLMProvider
        >>>
        >>> config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
        >>>
        >>> strategist, executor = create_optimizer_from_engine(
        ...     llm_config=config,
        ...     strategy_name="ema_cross",
        ...     data=ohlcv_df,
        ... )
        >>>
        >>> session = strategist.optimize(
        ...     executor=executor,
        ...     initial_params={"fast_period": 10, "slow_period": 21},
        ...     param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
        ...     max_iterations=10,
        ... )
        >>>
        >>> print(f"Best Sharpe: {session.best_result.sharpe_ratio}")
        >>> print(f"Best Params: {session.best_result.request.parameters}")
    """
    # Vérifier que la stratégie existe
    if strategy_name not in list_strategies():
        available = ", ".join(list_strategies())
        raise ValueError(
            f"Stratégie '{strategy_name}' inconnue. Disponibles: {available}"
        )

    # Créer le client LLM
    llm_client = create_llm_client(llm_config)

    # Créer la fonction de backtest
    def backtest_fn(
        strategy: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Dict[str, Any]:
        return run_backtest_for_agent(
            strategy_name=strategy,
            params=params,
            data=df,
            initial_capital=initial_capital,
            config=config,
        )

    # Créer la fonction de validation (optionnelle)
    validation_fn = None
    if use_walk_forward:
        def validation_fn(
            strategy: str,
            params: Dict[str, Any],
            df: pd.DataFrame,
            n_windows: int = 6,  # Optimisé pour 2 ans de données
            train_ratio: float = 0.75,  # 75/25 pour meilleur compromis
        ) -> Dict[str, Any]:
            return run_walk_forward_for_agent(
                strategy_name=strategy,
                params=params,
                data=df,
                n_windows=n_windows,
                train_ratio=train_ratio,
                initial_capital=initial_capital,
                config=config,
            )

    # Créer l'exécuteur
    executor = BacktestExecutor(
        backtest_fn=backtest_fn,
        strategy_name=strategy_name,
        data=data,
        validation_fn=validation_fn,
    )

    # Créer le strategist autonome
    strategist = AutonomousStrategist(
        llm_client,
        verbose=verbose,
        unload_llm_during_backtest=unload_llm_during_backtest,
        orchestration_logger=orchestration_logger,
    )

    _logger.info(
        "optimizer_created strategy=%s rows=%s walk_forward=%s",
        strategy_name, len(data), use_walk_forward
    )

    return strategist, executor


def get_strategy_param_bounds(
    strategy_name: str
) -> Dict[str, Tuple[float, float]]:
    """
    Récupère les bornes des paramètres d'une stratégie.

    Utilise les parameter_specs de la stratégie si disponibles,
    sinon retourne des bornes par défaut.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Dict {param_name: (min, max)}
    """
    strategy_class = get_strategy(strategy_name)
    strategy = strategy_class()

    bounds = {}

    # Essayer d'utiliser parameter_specs si disponible
    if hasattr(strategy, 'parameter_specs'):
        specs = strategy.parameter_specs
        # parameter_specs est un dict {name: ParameterSpec}
        if isinstance(specs, dict):
            for name, spec in specs.items():
                # ParameterSpec utilise min_val et max_val
                if hasattr(spec, 'min_val') and hasattr(spec, 'max_val'):
                    bounds[name] = (spec.min_val, spec.max_val)

    # Fallback: utiliser default_params avec ±50%
    if not bounds and hasattr(strategy, 'default_params'):
        for name, value in strategy.default_params.items():
            if isinstance(value, (int, float)) and value > 0:
                bounds[name] = (value * 0.5, value * 2.0)

    return bounds


def get_strategy_param_space(
    strategy_name: str,
    include_step: bool = True,
) -> Dict[str, Tuple]:
    """
    Récupère l'espace des paramètres avec step si disponible.

    Extension de get_strategy_param_bounds() pour permettre:
    - Le calcul unifié des stats d'espace de recherche
    - L'affichage d'estimation dans le mode LLM

    Args:
        strategy_name: Nom de la stratégie
        include_step: Inclure le step si disponible

    Returns:
        Dict {param_name: (min, max)} ou {param_name: (min, max, step)}
    """
    strategy_class = get_strategy(strategy_name)
    strategy = strategy_class()

    space = {}

    # Utiliser parameter_specs si disponible
    if hasattr(strategy, 'parameter_specs'):
        specs = strategy.parameter_specs
        if isinstance(specs, dict):
            for name, spec in specs.items():
                if hasattr(spec, 'min_val') and hasattr(spec, 'max_val'):
                    min_v = spec.min_val
                    max_v = spec.max_val

                    if include_step and hasattr(spec, 'step') and spec.step:
                        space[name] = (min_v, max_v, spec.step)
                    else:
                        space[name] = (min_v, max_v)

    # Fallback: param_ranges (si présent dans la stratégie)
    if not space and hasattr(strategy, 'param_ranges'):
        for name, (min_v, max_v) in strategy.param_ranges.items():
            space[name] = (min_v, max_v)

    # Dernier fallback: default_params avec ±50%
    if not space and hasattr(strategy, 'default_params'):
        for name, value in strategy.default_params.items():
            if isinstance(value, (int, float)) and value > 0:
                space[name] = (value * 0.5, value * 2.0)

    return space


def quick_optimize(
    llm_config: LLMConfig,
    strategy_name: str,
    data: pd.DataFrame,
    max_iterations: int = 10,
    **kwargs
) -> OptimizationSession:
    """
    Raccourci pour lancer une optimisation rapidement.

    Détecte automatiquement les paramètres initiaux et les bornes.

    Args:
        llm_config: Configuration LLM
        strategy_name: Nom de la stratégie
        data: DataFrame OHLCV
        max_iterations: Maximum d'itérations
        **kwargs: Arguments additionnels pour create_optimizer_from_engine

    Returns:
        OptimizationSession avec les résultats

    Example:
        >>> session = quick_optimize(
        ...     llm_config=config,
        ...     strategy_name="ema_cross",
        ...     data=df,
        ...     max_iterations=15,
        ... )
        >>> print(session.best_result.sharpe_ratio)
    """
    # Récupérer la stratégie pour les params par défaut
    strategy_class = get_strategy(strategy_name)
    strategy = strategy_class()

    # Paramètres initiaux
    initial_params = strategy.default_params.copy()

    # Bornes des paramètres
    param_bounds = get_strategy_param_bounds(strategy_name)

    if not param_bounds:
        raise ValueError(
            f"Impossible de déterminer les bornes pour '{strategy_name}'. "
            "Utilisez create_optimizer_from_engine avec des bornes explicites."
        )

    # Créer l'optimiseur
    strategist, executor = create_optimizer_from_engine(
        llm_config=llm_config,
        strategy_name=strategy_name,
        data=data,
        **kwargs
    )

    # Lancer l'optimisation
    session = strategist.optimize(
        executor=executor,
        initial_params=initial_params,
        param_bounds=param_bounds,
        max_iterations=max_iterations,
    )

    return session


def create_orchestrator_with_backtest(
    strategy_name: str,
    data: pd.DataFrame,
    initial_params: Dict[str, Any],
    llm_config: Optional[LLMConfig] = None,
    role_model_config: Optional[RoleModelConfig] = None,
    max_iterations: int = 10,
    initial_capital: float = 10000.0,
    config: Optional[Config] = None,
) -> "Orchestrator":
    """
    Crée un Orchestrator multi-agents branché sur le vrai backtest.

    L'Orchestrator par défaut nécessite un callback `on_backtest_needed`.
    Cette fonction le configure automatiquement avec `run_backtest_for_agent()`.

    Args:
        strategy_name: Nom de la stratégie
        data: DataFrame OHLCV
        initial_params: Paramètres initiaux
        llm_config: Configuration LLM (optionnel, défaut depuis env)
        role_model_config: Configuration multi-modeles par role
        max_iterations: Maximum d'itérations
        initial_capital: Capital de départ
        config: Configuration du backtest

    Returns:
        Orchestrator configuré et prêt à exécuter

    Example:
        >>> orchestrator = create_orchestrator_with_backtest(
        ...     strategy_name="ema_cross",
        ...     data=df,
        ...     initial_params={"fast_period": 12, "slow_period": 26},
        ... )
        >>> result = orchestrator.run()
        >>> if result.success:
        ...     print(f"Meilleurs params: {result.final_params}")
    """
    # Import ici pour éviter les imports circulaires
    from .orchestrator import Orchestrator, OrchestratorConfig
    from .base_agent import ParameterConfig

    # Récupérer les specs des paramètres
    param_space = get_strategy_param_space(strategy_name, include_step=True)
    param_specs = []
    for name, bounds in param_space.items():
        if len(bounds) == 3:
            min_v, max_v, step = bounds
        else:
            min_v, max_v = bounds
            step = None
        param_specs.append(ParameterConfig(
            name=name,
            min_value=min_v,
            max_value=max_v,
            step=step,
            current_value=initial_params.get(name, (min_v + max_v) / 2),
        ))

    # Créer le callback de backtest
    def on_backtest_needed(params: Dict[str, Any]) -> Dict[str, Any]:
        return run_backtest_for_agent(
            strategy_name=strategy_name,
            params=params,
            data=data,
            initial_capital=initial_capital,
            config=config,
        )

    # Créer la config
    orchestrator_config = OrchestratorConfig(
        strategy_name=strategy_name,
        initial_params=initial_params,
        param_specs=param_specs,
        max_iterations=max_iterations,
        llm_config=llm_config,
        role_model_config=role_model_config,
        on_backtest_needed=on_backtest_needed,
    )

    return Orchestrator(orchestrator_config)
