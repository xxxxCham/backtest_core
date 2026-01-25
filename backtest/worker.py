"""
Module worker isolé pour les backtests parallèles.

Ce module est séparé de l'UI Streamlit pour éviter les problèmes de pickling
quand Streamlit recharge ses modules (hot-reload).

La fonction `run_backtest_worker` est stable et ne change pas de référence
pendant l'exécution d'un sweep.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Tuple

# Ces imports sont faits au niveau du module pour être disponibles dans les workers
try:
    from backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None

# Variable globale pour le DataFrame (partagée entre tous les backtests d'un worker)
# Initialisée une seule fois par worker via init_worker_with_dataframe()
_worker_dataframe = None
_worker_strategy_key = None
_worker_symbol = None
_worker_timeframe = None
_worker_initial_capital = None
_worker_debug_enabled = False
_worker_fast_metrics = False
_worker_period_days = None
_worker_engine = None
_worker_indicator_cache = None  # Cache des indicateurs calculés


def init_worker_with_dataframe(
    df_or_path,
    strategy_key: str,
    symbol: str,
    timeframe: str,
    initial_capital: float,
    debug_enabled: bool,
    thread_limit: int,
    fast_metrics: bool = False,
    is_path: bool = False,
):
    """
    Initializer pour ProcessPoolExecutor - charge le DataFrame une seule fois.

    Cette fonction est appelée une fois par worker au démarrage du pool.
    Le DataFrame est stocké en variable globale pour éviter la sérialisation pickle
    répétée à chaque soumission de tâche.

    IMPORTANT: Application ROBUSTE des limites de threads BLAS pour éviter
    nested parallelism (8 workers × 16 threads BLAS = 128 threads → surcharge CPU).
    La limitation est appliquée PROGRAMMATIQUEMENT, pas seulement via env vars.

    Args:
        df_or_path: DataFrame OHLCV complet OU chemin vers fichier parquet (si is_path=True)
        strategy_key: Nom de la stratégie
        symbol: Symbole (ex: BTCUSDC)
        timeframe: Timeframe (ex: 1h)
        initial_capital: Capital initial
        debug_enabled: Activer logs de debug
        thread_limit: Limite de threads CPU (0 = auto-detect, recommandé: 1)
        fast_metrics: Utiliser les métriques rapides pour les sweeps
        is_path: Si True, df_or_path est un chemin de fichier à charger
    """
    global _worker_dataframe, _worker_strategy_key, _worker_symbol
    global _worker_timeframe, _worker_initial_capital, _worker_debug_enabled
    global _worker_fast_metrics, _worker_period_days, _worker_engine

    # Charger le DataFrame depuis le fichier ou utiliser celui fourni
    if is_path:
        import pandas as pd
        _worker_dataframe = pd.read_parquet(df_or_path)
    else:
        _worker_dataframe = df_or_path
    _worker_strategy_key = strategy_key
    _worker_symbol = symbol
    _worker_timeframe = timeframe
    _worker_initial_capital = initial_capital
    _worker_debug_enabled = debug_enabled
    _worker_fast_metrics = fast_metrics
    _worker_engine = None

    # Réinitialiser le cache d'indicateurs pour ce worker
    global _worker_indicator_cache
    _worker_indicator_cache = {}

    # Pré-calcul du nombre de jours (évite coût répété)
    _worker_period_days = None
    try:
        import pandas as pd
        start_day = pd.to_datetime(_worker_dataframe.index[0]).date()
        end_day = pd.to_datetime(_worker_dataframe.index[-1]).date()
        days = (end_day - start_day).days
        _worker_period_days = days if days > 0 else None
    except Exception:
        _worker_period_days = None

    # ═══════════════════════════════════════════════════════════════════════════
    # LIMITATION THREADS BLAS - APPLICATION PROGRAMMATIQUE ROBUSTE
    # ═══════════════════════════════════════════════════════════════════════════
    # PROBLÈME: Sans limitation, NumPy/SciPy utilise tous les cœurs par worker
    #   → 8 workers × 16 threads BLAS = 128 threads → surcharge CPU massive
    # SOLUTION: Forcer 1 thread BLAS par worker de manière programmatique
    #   → 8 workers × 1 thread BLAS = 8 threads → charge CPU optimale
    # ═══════════════════════════════════════════════════════════════════════════

    # Déterminer la limite effective (défaut: 1 thread si non spécifié)
    effective_limit = thread_limit if thread_limit > 0 else 1

    # 1️⃣ Variables d'environnement (fallback pour bibliothèques qui ne supportent pas threadpoolctl)
    os.environ["BACKTEST_WORKER_THREADS"] = str(effective_limit)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = str(effective_limit)

    # 2️⃣ threadpoolctl - APPLICATION PROGRAMMATIQUE (priorité haute)
    # C'est LA méthode robuste pour contrôler BLAS même sans env vars
    threadpoolctl_applied = False
    try:
        import threadpoolctl

        # Appliquer limite sur TOUS les thread pools (BLAS, OpenMP, TBB, etc.)
        threadpoolctl.threadpool_limits(limits=effective_limit)
        threadpoolctl_applied = True

        # Vérification (optionnel en debug)
        if debug_enabled:
            info = threadpoolctl.threadpool_info()
            total_threads = sum(pool.get("num_threads", 0) for pool in info)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                "Worker thread limit applied: %d thread(s) across %d pool(s)",
                total_threads,
                len(info),
            )
    except ImportError:
        # threadpoolctl non installé - fallback env vars uniquement
        if debug_enabled:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "threadpoolctl not available - using env vars only (less reliable)"
            )
    except Exception as e:
        if debug_enabled:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("threadpoolctl configuration failed: %s", e)

    # 3️⃣ PyTorch (si disponible)
    try:
        import torch
        torch.set_num_threads(effective_limit)
        torch.set_num_interop_threads(max(1, effective_limit // 2))
    except Exception:
        pass  # Torch non utilisé ou indisponible

    # 4️⃣ Vérification finale (optionnel - seulement en debug)
    if debug_enabled and not threadpoolctl_applied:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "⚠️ Thread limiting applied via env vars only - "
            "consider installing threadpoolctl for robust control: "
            "pip install threadpoolctl"
        )


def run_backtest_worker(param_combo: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function pour ProcessPoolExecutor - isolé du hot-reload Streamlit.

    Cette fonction est définie dans un module séparé (backtest/worker.py) pour
    éviter les erreurs de pickling quand Streamlit recharge ui/main.py.

    IMPORTANT: Le DataFrame et la configuration sont chargés depuis les variables
    globales initialisées par init_worker_with_dataframe(). Seul param_combo
    est passé en argument, ce qui évite la sérialisation pickle répétée du DataFrame.

    Args:
        param_combo: Dictionnaire des paramètres de la stratégie à tester

    Returns:
        Dict avec résultats du backtest ou erreur
    """
    # Récupérer les données depuis les variables globales du worker
    global _worker_engine

    df = _worker_dataframe
    strategy_key = _worker_strategy_key
    symbol = _worker_symbol
    timeframe = _worker_timeframe
    initial_capital = _worker_initial_capital
    debug_enabled = _worker_debug_enabled
    fast_metrics = _worker_fast_metrics

    # Validation
    if df is None:
        return {
            "params_dict": param_combo,
            "error": "Worker not initialized - DataFrame is None",
        }

    period_days = _worker_period_days

    try:
        if BacktestEngine is None:
            return {
                "params_dict": param_combo,
                "error": "BacktestEngine not available in worker",
            }

        # Créer l'engine localement (pas picklable donc recréé dans chaque process)
        # Note: Les limites de threads sont déjà appliquées par init_worker_with_dataframe()
        if _worker_engine is None:
            _worker_engine = BacktestEngine(initial_capital=initial_capital)
        engine = _worker_engine

        # Import local pour éviter les dépendances circulaires
        from ui.helpers import safe_run_backtest

        # IMPORTANT: ordre des arguments = (engine, df, strategy, params, symbol, timeframe)
        result_i, msg_i = safe_run_backtest(
            engine,
            df,
            strategy_key,
            param_combo,
            symbol=symbol,
            timeframe=timeframe,
            silent_mode=True,
            fast_metrics=fast_metrics,
        )

        if result_i is None:
            return {
                "params_dict": param_combo,
                "error": msg_i or "Backtest failed",
            }

        # Extraire les métriques avec fallback robuste
        m = {}
        if hasattr(result_i, "metrics") and result_i.metrics:
            m = result_i.metrics
        elif isinstance(result_i, dict):
            m = result_i.get("metrics", result_i)

        total_trades = m.get("total_trades", 0)
        return {
            "params_dict": param_combo,
            "total_pnl": m.get("total_pnl", 0.0),
            "sharpe": m.get("sharpe_ratio", 0.0),
            "win_rate": m.get("win_rate_pct", m.get("win_rate", 0.0)),
            "max_dd": m.get("max_drawdown_pct", m.get("max_drawdown", 0.0)),
            "account_ruined": m.get("account_ruined", False),
            "min_equity": m.get("min_equity", 0.0),
            "total_trades": total_trades,
            "trades": total_trades,
            "profit_factor": m.get("profit_factor", 0.0),
            "liquidation_total_pnl": m.get("liquidation_total_pnl", m.get("total_pnl", 0.0)),
            "liquidation_total_return_pct": m.get("liquidation_total_return_pct", m.get("total_return_pct", 0.0)),
            "liquidation_sharpe_ratio": m.get("liquidation_sharpe_ratio", m.get("sharpe_ratio", 0.0)),
            "liquidation_max_drawdown_pct": m.get("liquidation_max_drawdown_pct", m.get("max_drawdown_pct", 0.0)),
            "liquidation_triggered": m.get("liquidation_triggered", False),
            "liquidation_time": m.get("liquidation_time"),
            "consecutive_losses_max": m.get("consecutive_losses_max", 0),
            "avg_win_loss_ratio": m.get("avg_win_loss_ratio", 0.0),
            "robustness_score": m.get("robustness_score", 0.0),
            "data_coverage_pct": m.get("data_coverage_pct"),
            "period_days": period_days,
        }

    except Exception as e:
        # Capturer toute erreur d'exécution
        error_msg = str(e)
        if debug_enabled:
            import traceback
            error_msg = traceback.format_exc()
        return {
            "params_dict": param_combo,
            "error": error_msg,
        }
