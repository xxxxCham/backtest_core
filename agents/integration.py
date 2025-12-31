"""
Module-ID: agents.integration

Purpose: Relier les agents LLM (abstraits) au moteur de backtest concret (BacktestEngine + WalkForwardValidator).

Role in pipeline: orchestration

Key components: run_backtest_for_agent, run_walk_forward_for_agent, create_optimizer_from_engine, create_orchestrator_with_backtest, validate_walk_forward_period

Inputs: DataFrame OHLCV, Config, stratégie (key/name), LLMConfig/RoleModelConfig, paramètres et options walk-forward

Outputs: Résultats de backtest/walk-forward adaptés aux agents, factories d’optimiseurs/orchestrator prêts à l’emploi

Dependencies: backtest.engine, backtest.validation, strategies.base, utils.config, utils.observability, agents.backtest_executor

Conventions: MIN_DAYS_FOR_WALK_FORWARD=180; timestamps détectés (index datetime ou colonne 'timestamp'); WF peut être désactivé si période insuffisante.

Read-if: Vous modifiez le wiring agents↔engine (run_backtest, walk-forward, factories).

Skip-if: Vous ne changez que les stratégies/indicateurs ou la UI.
"""

from __future__ import annotations

# pylint: disable=logging-fstring-interpolation

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.validation import ValidationFold, WalkForwardValidator
from metrics_types import normalize_metrics, pct_to_frac
from strategies.base import get_strategy, list_strategies
from utils.config import Config
from utils.observability import (
    generate_run_id,
    get_obs_logger,
    trace_span,
)

from .autonomous_strategist import AutonomousStrategist, OptimizationSession
from .backtest_executor import BacktestExecutor
from .llm_client import LLMConfig, create_llm_client
from .model_config import RoleModelConfig

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import Orchestrator

# Logger module-level (sans run_id spécifique)
_logger = get_obs_logger(__name__)

# Constantes pour validation walk-forward
MIN_DAYS_FOR_WALK_FORWARD = 180  # 6 mois minimum


def _normalize_engine_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return normalize_metrics(metrics, "pct")


class AgentBacktestMetrics(TypedDict):
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    sqn: float
    calmar_ratio: float
    recovery_factor: float
    equity_curve: Optional[List[float]]
    trades: Optional[List[Dict[str, Any]]]
    run_id: str


class WalkForwardMetrics(TypedDict):
    train_sharpe: float
    test_sharpe: float
    overfitting_ratio: float
    classic_ratio: float
    degradation_pct: float
    test_stability_std: float
    n_valid_folds: int


def extract_dataframe_timestamps(
    data: pd.DataFrame
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Extrait les timestamps de début et fin d'un DataFrame OHLCV.

    Gère automatiquement:
    - DatetimeIndex
    - Colonne 'timestamp' ou 'date' (datetime ou numérique ms/s)

    Args:
        data: DataFrame OHLCV

    Returns:
        Tuple (start_datetime, end_datetime)

    Raises:
        ValueError: Si aucun timestamp n'est trouvé ou format invalide

    Example:
        >>> start, end = extract_dataframe_timestamps(df)
        >>> duration_days = (end - start).days
    """
    # Cas 1: DatetimeIndex
    if isinstance(data.index, pd.DatetimeIndex):
        return data.index[0], data.index[-1]

    # Cas 2: Colonne timestamp/date
    col = None
    if "timestamp" in data.columns:
        col = "timestamp"
    elif "date" in data.columns:
        col = "date"
    else:
        raise ValueError(
            "DataFrame must have DatetimeIndex or 'timestamp'/'date' column"
        )

    ts_col = data[col]

    # Déjà en datetime
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        return ts_col.iloc[0], ts_col.iloc[-1]

    # Numérique: détecter ms vs s
    if pd.api.types.is_numeric_dtype(ts_col):
        first_val = ts_col.iloc[0]
        # Heuristique: >1e12 = millisecondes, sinon secondes
        unit = "ms" if first_val > 1e12 else "s"
        return (
            pd.to_datetime(first_val, unit=unit),
            pd.to_datetime(ts_col.iloc[-1], unit=unit)
        )

    raise ValueError(
        f"Column '{col}' must be datetime or numeric, got {ts_col.dtype}"
    )


def validate_walk_forward_period(
    data: pd.DataFrame,
    min_days: int = MIN_DAYS_FOR_WALK_FORWARD,
) -> tuple[bool, int, str]:
    """
    Valide si la période de données est suffisante pour une walk-forward validation.

    La walk-forward nécessite une période minimale pour avoir une signification
    statistique. En dessous de 6 mois, les folds sont trop courts et les résultats
    sont dominés par le bruit statistique.

    Args:
        data: DataFrame OHLCV avec index datetime ou colonne 'timestamp'
        min_days: Nombre de jours minimum requis (défaut: 180 = 6 mois)

    Returns:
        Tuple (is_valid, duration_days, message)
        - is_valid: True si période suffisante, False sinon
        - duration_days: Durée en jours de la période
        - message: Message explicatif

    Exemples:
        >>> is_valid, days, msg = validate_walk_forward_period(df)
        >>> if not is_valid:
        ...     print(f"⚠️ {msg}")
        ...     # Désactiver walk-forward
    """
    # Extraire les timestamps (fonction helper centralisée)
    start_dt, end_dt = extract_dataframe_timestamps(data)

    # Calculer la durée en jours
    duration = (end_dt - start_dt).days

    # Valider
    if duration < min_days:
        months = duration / 30.0
        min_months = min_days / 30.0
        message = (
            f"Période insuffisante pour walk-forward validation: "
            f"{duration} jours ({months:.1f} mois) < {min_days} jours ({min_months:.0f} mois minimum). "
            f"Walk-forward DÉSACTIVÉ automatiquement pour éviter des résultats non significatifs."
        )
        return False, duration, message

    months = duration / 30.0
    message = (
        f"Période validée pour walk-forward: "
        f"{duration} jours ({months:.1f} mois) ≥ {min_days} jours. "
        f"Walk-forward validation activée."
    )
    return True, duration, message


def run_backtest_for_agent(
    strategy_name: str,
    params: Dict[str, Any],
    data: pd.DataFrame,
    *,
    initial_capital: float = 10000.0,
    config: Optional[Config] = None,
    run_id: Optional[str] = None,
) -> AgentBacktestMetrics:
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
        metrics_pct = normalize_metrics(result.metrics, "pct")
        metrics_frac = pct_to_frac(metrics_pct)

        output: AgentBacktestMetrics = {
            "sharpe_ratio": metrics_frac.get("sharpe_ratio", 0),
            "sortino_ratio": metrics_frac.get("sortino_ratio", 0),
            "total_return": metrics_frac.get("total_return", 0),
            "max_drawdown": metrics_frac.get("max_drawdown", 0),
            "win_rate": metrics_frac.get("win_rate", 0),
            "profit_factor": metrics_frac.get("profit_factor", 0),
            "total_trades": metrics_frac.get("total_trades", 0),
            "sqn": metrics_frac.get("sqn", 0),
            "calmar_ratio": metrics_frac.get("calmar_ratio", 0),
            "recovery_factor": metrics_frac.get("recovery_factor", 0),
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
    n_workers: int = 1,
) -> WalkForwardMetrics:
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
        n_workers: Nombre de workers pour parallélisation (1 = séquentiel)

    Returns:
        Dict avec train_sharpe, test_sharpe, overfitting_ratio, métriques robustes
    """
    test_pct = 1.0 - train_ratio

    validator = WalkForwardValidator(
        n_folds=n_windows,
        test_pct=test_pct,
        embargo_pct=0.02,  # 2% d'embargo pour éviter le leakage
    )

    folds = validator.split(data)

    def _run_fold(fold: ValidationFold) -> tuple[ValidationFold, bool]:
        """Exécute un fold complet (train + test) - thread-safe."""
        # Créer une instance d'engine par thread pour éviter les problèmes de concurrence
        engine = BacktestEngine(initial_capital=initial_capital, config=config)
        train_df, test_df = validator.get_data_splits(data, fold)

        try:
            # Backtest sur train
            train_result = engine.run(
                df=train_df,
                strategy=strategy_name,
                params=params,
            )

            # Backtest sur test
            test_result = engine.run(
                df=test_df,
                strategy=strategy_name,
                params=params,
            )

            # Stocker dans le fold
            fold.train_metrics = _normalize_engine_metrics(train_result.metrics)
            fold.test_metrics = _normalize_engine_metrics(test_result.metrics)

            return fold, True

        except Exception as e:
            _logger.warning("fold_%s_failed error=%s", fold.fold_id, str(e))
            return fold, False

    # Mode séquentiel (par défaut)
    if n_workers <= 1 or len(folds) <= 1:
        for fold in folds:
            _run_fold(fold)
    else:
        # Mode parallèle - utiliser ThreadPoolExecutor pour paralléliser les folds
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _logger.info(f"Walk-forward parallèle avec {n_workers} workers sur {len(folds)} folds")

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_fold, fold): fold for fold in folds}

            for fut in as_completed(futures):
                original_fold = futures[fut]
                try:
                    updated_fold, success = fut.result()
                    if not success:
                        _logger.warning(f"Fold {original_fold.fold_id} a échoué")
                except Exception as e:
                    _logger.warning(f"Erreur lors de l'exécution du fold {original_fold.fold_id}: {e}")

    # Collecter les résultats des folds valides
    train_sharpes = []
    test_sharpes = []
    for fold in folds:
        if fold.train_metrics and fold.test_metrics:
            train_sharpes.append(fold.train_metrics.get("sharpe_ratio", 0))
            test_sharpes.append(fold.test_metrics.get("sharpe_ratio", 0))

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

    # Valider la période pour walk-forward (garde-fou)
    walk_forward_disabled_reason = None
    if use_walk_forward:
        is_valid, duration_days, message = validate_walk_forward_period(data)
        if not is_valid:
            _logger.warning(
                "walk_forward_auto_disabled duration_days=%s reason='period_too_short'",
                duration_days
            )
            _logger.warning(message)
            use_walk_forward = False  # Forcer désactivation
            walk_forward_disabled_reason = message

    # Créer le client LLM
    llm_client = create_llm_client(llm_config)

    # Créer la fonction de backtest
    def backtest_fn(
        strategy: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> AgentBacktestMetrics:
        return run_backtest_for_agent(
            strategy_name=strategy,
            params=params,
            data=df,
            initial_capital=initial_capital,
            config=config,
        )

    # Créer la fonction de validation (optionnelle)
    def _validation_fn(
        strategy: str,
        params: Dict[str, Any],
        df: pd.DataFrame,
        n_windows: int = 6,  # Optimisé pour 2 ans de données
        train_ratio: float = 0.75,  # 75/25 pour meilleur compromis
    ) -> WalkForwardMetrics:
        return run_walk_forward_for_agent(
            strategy_name=strategy,
            params=params,
            data=df,
            n_windows=n_windows,
            train_ratio=train_ratio,
            initial_capital=initial_capital,
            config=config,
        )

    validation_fn = _validation_fn if use_walk_forward else None

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

    # Si walk-forward désactivé automatiquement, logger pour rapport final
    if walk_forward_disabled_reason:
        _logger.info(
            "walk_forward_validation_status disabled_reason='%s'",
            walk_forward_disabled_reason
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
) -> Dict[str, Union[Tuple[float, float], Tuple[float, float, float]]]:
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
    data_symbol: str = "",
    data_timeframe: str = "",
    llm_config: Optional[LLMConfig] = None,
    role_model_config: Optional[RoleModelConfig] = None,
    use_walk_forward: bool = True,
    orchestration_logger: Optional[Any] = None,
    session_id: Optional[str] = None,
    n_workers: int = 1,
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
        data_symbol: Symbole (ex: "BTCUSDC")
        data_timeframe: Timeframe (ex: "1h")
        llm_config: Configuration LLM (optionnel, défaut depuis env)
        role_model_config: Configuration multi-modeles par role
        use_walk_forward: Activer la validation walk-forward (si possible)
        orchestration_logger: Logger d'orchestration (UI live/persistance)
        session_id: Forcer l'ID de session (corrélation UI)
        n_workers: Nombre de workers pour paralléliser les backtests de propositions
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
    from .base_agent import ParameterConfig
    from .orchestrator import Orchestrator, OrchestratorConfig

    # Valider la période pour walk-forward (garde-fou)
    walk_forward_disabled_reason = None
    if use_walk_forward:
        is_valid, duration_days, message = validate_walk_forward_period(data)
        if not is_valid:
            _logger.warning(
                "walk_forward_auto_disabled duration_days=%s reason='period_too_short'",
                duration_days
            )
            _logger.warning(message)
            use_walk_forward = False  # Forcer désactivation
            walk_forward_disabled_reason = message

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
    def on_backtest_needed(params: Dict[str, Any]) -> AgentBacktestMetrics:
        return run_backtest_for_agent(
            strategy_name=strategy_name,
            params=params,
            data=data,
            initial_capital=initial_capital,
            config=config,
        )

    data_date_range = ""
    try:
        start_dt, end_dt = extract_dataframe_timestamps(data)
        data_date_range = f"{start_dt} -> {end_dt}"
    except (ValueError, Exception):
        data_date_range = ""

    # Créer la config
    orchestrator_config = OrchestratorConfig(
        strategy_name=strategy_name,
        initial_params=initial_params,
        param_specs=param_specs,
        max_iterations=max_iterations,
        llm_config=llm_config,
        role_model_config=role_model_config,
        use_walk_forward=use_walk_forward,
        walk_forward_disabled_reason=walk_forward_disabled_reason,
        data=data,
        data_symbol=data_symbol,
        data_timeframe=data_timeframe,
        data_date_range=data_date_range,
        n_workers=n_workers,
        session_id=session_id,
        orchestration_logger=orchestration_logger,
        on_backtest_needed=on_backtest_needed,
    )

    _logger.info(
        "orchestrator_created strategy=%s rows=%s walk_forward=%s",
        strategy_name, len(data), use_walk_forward
    )

    # Si walk-forward désactivé automatiquement, logger pour rapport final
    if walk_forward_disabled_reason:
        _logger.info(
            "walk_forward_validation_status disabled_reason='%s'",
            walk_forward_disabled_reason
        )

    return Orchestrator(orchestrator_config)
