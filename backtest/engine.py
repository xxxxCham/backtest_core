"""
Module-ID: backtest.engine

Purpose: Orchestrer le pipeline complet de backtesting (données → indicateurs → signaux → trades → métriques).

Role in pipeline: orchestration / core

Key components: BacktestEngine, RunResult, run_backtest

Inputs: DataFrame OHLCV, StrategyBase, Config optionnel

Outputs: RunResult (trades, metrics, equity curve, detailed report)

Dependencies: strategies.base, indicators.registry, backtest.simulator, backtest.performance, data.indicator_bank, utils.config

Conventions: Indicateurs calculés une fois; signaux normalisés (1/-1/0); prix en devise de base.

Read-if: Vous modifiez le pipeline principal ou l'API de BacktestEngine.

Skip-if: Vous ne faites que des stratégies/indicateurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from strategies.base import StrategyBase

from backtest.performance import calculate_metrics

# Import simulateur rapide (Numba) avec fallback
try:
    from backtest.simulator_fast import (
        simulate_trades_fast,
        calculate_equity_fast,
        calculate_returns_fast,
        HAS_NUMBA,
    )
    USE_FAST_SIMULATOR = True
except ImportError:
    USE_FAST_SIMULATOR = False
    HAS_NUMBA = False

# Import simulateur standard (fallback)
from backtest.simulator import (
    calculate_equity_curve,
    calculate_returns,
    simulate_trades,
)
from data.indicator_bank import get_indicator_bank
from indicators.registry import calculate_indicator
from utils.config import Config
from utils.data import detect_gaps
from utils.log import CountingHandler
from utils.observability import (
    get_obs_logger,
    generate_run_id,
    trace_span,
    PerfCounters,
)
from utils.version import get_git_commit

# Logger par défaut (sans run_id)
_default_logger = get_obs_logger(__name__)


@dataclass
class RunResult:
    """
    Résultat d'exécution d'un backtest.

    Attributes:
        equity: Courbe d'équité (pd.Series indexée par datetime)
        returns: Rendements par période (pd.Series)
        trades: DataFrame des trades exécutés
        metrics: Dict des métriques de performance calculées
        meta: Métadonnées d'exécution (durée, paramètres, etc.)
    """
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation des données."""
        if not isinstance(self.equity, pd.Series):
            raise TypeError("equity doit être une pd.Series")
        if not isinstance(self.returns, pd.Series):
            raise TypeError("returns doit être une pd.Series")
        if not isinstance(self.trades, pd.DataFrame):
            raise TypeError("trades doit être un pd.DataFrame")

    def summary(self) -> str:
        """Retourne un résumé textuel du résultat."""
        n_trades = len(self.trades)
        total_pnl = self.metrics.get("total_pnl", 0)
        sharpe = self.metrics.get("sharpe_ratio", 0)
        max_dd = self.metrics.get("max_drawdown", 0)
        win_rate = self.metrics.get("win_rate", 0)

        return f"""
Backtest Summary
================
Trades: {n_trades}
Total P&L: ${total_pnl:,.2f}
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.1f}%
Win Rate: {win_rate:.1f}%
"""


class BacktestEngine:
    """
    Moteur de backtesting principal.

    Orchestrateur simplifié qui exécute le pipeline complet:
    données → indicateurs → signaux → trades → métriques

    Usage:
        engine = BacktestEngine()
        result = engine.run(
            df=ohlcv_data,
            strategy=BollingerATRStrategy(),
            params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 3}
        )
        print(result.summary())

    Architecture modulaire pour extension future:
    - Stratégies interchangeables via interface StrategyBase
    - Indicateurs via registre extensible
    - Prêt pour réintégration LLM (strategy_instance paramètre)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        config: Optional[Config] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialise le moteur.

        Args:
            initial_capital: Capital de départ
            config: Configuration (optionnel)
            run_id: Identifiant de corrélation (généré si None)
        """
        self.initial_capital = initial_capital
        self.config = config or Config()
        self.run_id = run_id or generate_run_id()
        self.logger = get_obs_logger(__name__, run_id=self.run_id)
        self.last_run_meta: Dict[str, Any] = {}
        self.counters: Optional[PerfCounters] = None
        self.indicator_bank = None

        self.logger.info("BacktestEngine init capital=%s", initial_capital)

    def run(
        self,
        df: pd.DataFrame,
        strategy: Union[StrategyBase, str],
        params: Optional[Dict[str, Any]] = None,
        *,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m",
        seed: int = 42,
        silent_mode: bool = False
    ) -> RunResult:
        """
        Exécute un backtest complet.

        Args:
            df: DataFrame OHLCV avec colonnes (open, high, low, close, volume)
            strategy: Instance de stratégie ou nom de stratégie
            params: Paramètres de trading et stratégie
            symbol: Symbole de l'actif (pour logging)
            timeframe: Timeframe des données (pour ajustements)
            seed: Seed pour reproductibilité
            silent_mode: Si True, désactive les logs structurés (RUN_START, DATA_LOADED, etc.)
                        pour améliorer les performances en grid search

        Returns:
            RunResult avec equity, returns, trades, metrics et meta

        Raises:
            ValueError: Si données ou paramètres invalides
        """
        # Initialiser counters et contexte
        self.counters = PerfCounters()
        self.counters.start("total")

        # Initialiser le counting handler pour compter warnings/errors (seulement si pas en silent_mode)
        if not silent_mode:
            counting_handler = CountingHandler()
            # Attacher au logger sous-jacent (ObsLoggerAdapter.logger)
            underlying_logger = self.logger.logger if hasattr(self.logger, 'logger') else self.logger
            underlying_logger.addHandler(counting_handler)

        # Enrichir le logger avec contexte
        self.logger = self.logger.with_context(symbol=symbol, timeframe=timeframe)

        if not silent_mode:
            self.logger.info("pipeline_start strategy=%s bars=%s",
                             strategy if isinstance(strategy, str) else getattr(strategy, 'name', 'custom'),
                             len(df))

            # RUN_START : Log structuré du démarrage
            strategy_name_log = strategy if isinstance(strategy, str) else getattr(strategy, 'name', 'custom')
            self.logger.info(
                f"RUN_START run_id={self.run_id} git_commit={get_git_commit()} mode=backtest "
                f"symbol={symbol} timeframe={timeframe} strategy={strategy_name_log} "
                f"initial_capital={self.initial_capital} fees_bps={self.config.fees_bps} "
                f"slippage_bps={self.config.slippage_bps} leverage={params.get('leverage', 1) if params is not None else 1} "
                f"seed={seed} n_bars={len(df)} period_start={df.index[0]} period_end={df.index[-1]}"
            )

        # Seed pour déterminisme
        np.random.seed(seed)

        try:
            # 1. Validation des entrées
            with trace_span(self.logger, "validation"):
                self._validate_inputs(df, strategy, params)

            # DATA_LOADED : Log structuré des données (désactivé en silent_mode)
            if not silent_mode:
                gaps_info = detect_gaps(df)
                self.logger.info(
                    f"DATA_LOADED run_id={self.run_id} n_bars={len(df)} "
                    f"gaps_count={gaps_info.get('gaps_count', 0)} "
                    f"gaps_pct={gaps_info.get('gaps_pct', 0.0):.2f} "
                    f"timezone={str(df.index.tz)} "
                    f"open_min={df['open'].min():.2f} open_max={df['open'].max():.2f} "
                    f"volume_min={df['volume'].min():.0f} volume_max={df['volume'].max():.0f} "
                    f"n_nan_open={df['open'].isna().sum()} n_nan_close={df['close'].isna().sum()}"
                )

            # 2. Préparer la stratégie
            if isinstance(strategy, str):
                strategy = self._get_strategy_by_name(strategy)

            strategy_name = strategy.name
            self.logger = self.logger.with_context(strategy=strategy_name)

            # 3. Fusionner paramètres
            final_params = {
                "initial_capital": self.initial_capital,
                "fees_bps": self.config.fees_bps,
                "slippage_bps": self.config.slippage_bps,
                **strategy.default_params,
                **(params or {})
            }

            if not silent_mode:
                self.logger.debug("params=%s", final_params)

                # PARAMS_RESOLVED : Log structuré des paramètres finaux
                params_str = " ".join([f"{k}={v}" for k, v in final_params.items()])
                self.logger.info(
                    f"PARAMS_RESOLVED run_id={self.run_id} strategy={strategy_name} "
                    f"source={'user+defaults' if params else 'defaults_only'} {params_str}"
                )

            # 4. Calculer les indicateurs requis
            self.counters.start("indicators")
            with trace_span(self.logger, "indicators", count=len(strategy.required_indicators)):
                indicators = self._calculate_indicators(df, strategy, final_params)
            self.counters.stop("indicators")

            # 5. Générer les signaux
            self.counters.start("signals")
            with trace_span(self.logger, "signals"):
                signals = strategy.generate_signals(df, indicators, final_params)
                n_signals = int((signals != 0).sum())
            self.counters.stop("signals")
            self.counters.increment("signals_count", n_signals)
            self.logger.debug("signals_generated count=%s", n_signals)

            # 6. Simuler les trades (utilise version rapide si disponible)
            execution_engine = self._build_execution_engine(final_params)
            self.counters.start("simulation")
            with trace_span(self.logger, "simulation"):
                if USE_FAST_SIMULATOR and execution_engine is None:
                    trades_df = simulate_trades_fast(df, signals, final_params)
                else:
                    trades_df = simulate_trades(df, signals, final_params, execution_engine=execution_engine)
            self.counters.stop("simulation")
            self.counters.increment("trades_count", len(trades_df))

            # 7. Calculer équité et rendements (version rapide si disponible)
            self.counters.start("equity")
            if USE_FAST_SIMULATOR:
                equity = calculate_equity_fast(df, trades_df, self.initial_capital)
                returns = calculate_returns_fast(equity)
            else:
                equity = calculate_equity_curve(
                    df, trades_df, self.initial_capital,
                    run_id=self.run_id  # Propager run_id pour logs structurés
                )
                returns = calculate_returns(equity)
            self.counters.stop("equity")

            # 8. Calculer les métriques
            self.counters.start("metrics")
            periods_per_year = self._get_periods_per_year(timeframe)
            metrics = calculate_metrics(
                equity=equity,
                returns=returns,
                trades_df=trades_df,
                initial_capital=self.initial_capital,
                periods_per_year=periods_per_year,
                run_id=self.run_id  # Propager run_id pour logs structurés
            )
            self.counters.stop("metrics")

            # 9. Construire les métadonnées
            self.counters.stop("total")
            total_ms = self.counters.get_duration("total")

            meta = {
                "run_id": self.run_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy_name,
                "params": final_params,
                "duration_sec": total_ms / 1000,
                "n_bars": len(df),
                "period_start": str(df.index[0]),
                "period_end": str(df.index[-1]),
                "seed": seed,
                "perf_counters": self.counters.summary(),
            }

            self.last_run_meta = meta

            # 10. Construire le résultat
            result = RunResult(
                equity=equity,
                returns=returns,
                trades=trades_df,
                metrics=metrics,
                meta=meta
            )

            if not silent_mode:
                self.logger.info(
                    "pipeline_end duration_ms=%.1f trades=%s sharpe=%.2f pnl=%.2f",
                    total_ms, len(trades_df), metrics.get('sharpe_ratio', 0), metrics.get('total_pnl', 0)
                )

                # RUN_END_SUMMARY : Log structuré complet des résultats
                fees_total = trades_df['fees'].sum() if 'fees' in trades_df.columns else 0
                slippage_total = trades_df['slippage'].sum() if 'slippage' in trades_df.columns else 0
                self.logger.info(
                    f"RUN_END_SUMMARY run_id={self.run_id} "
                    f"total_return_pct={metrics.get('total_return_pct', 0):.2f} "
                    f"cagr={metrics.get('cagr', 0):.2f} "
                    f"sharpe={metrics.get('sharpe_ratio', 0):.2f} "
                    f"sortino={metrics.get('sortino_ratio', 0):.2f} "
                    f"calmar={metrics.get('calmar_ratio', 0):.2f} "
                    f"max_dd_pct={metrics.get('max_drawdown', 0):.2f} "
                    f"trades_count={len(trades_df)} "
                    f"win_rate_pct={metrics.get('win_rate', 0):.1f} "
                    f"profit_factor={metrics.get('profit_factor', 0):.2f} "
                    f"avg_trade_pnl={metrics.get('avg_trade_pnl', 0):.2f} "
                    f"fees_total={fees_total:.2f} "
                    f"slippage_total={slippage_total:.2f} "
                    f"duration_sec={total_ms / 1000:.3f} "
                    f"warnings_count={counting_handler.warnings} errors_count={counting_handler.errors}"
                )

            # Nettoyer le handler après utilisation (seulement si créé)
            if not silent_mode:
                underlying_logger.removeHandler(counting_handler)

            return result

        except Exception as e:
            self.counters.stop("total")
            self.logger.error("pipeline_error error=%s", str(e))
            # Nettoyer le handler en cas d'erreur
            if 'counting_handler' in locals() and 'underlying_logger' in locals():
                underlying_logger.removeHandler(counting_handler)
            raise

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        strategy: Union[StrategyBase, str],
        params: Optional[Dict[str, Any]]
    ) -> None:
        """Valide les entrées du backtest."""
        _ = params

        # Validation DataFrame
        if df.empty:
            raise ValueError("DataFrame vide")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index doit être DatetimeIndex")

        # Validation stratégie
        if not isinstance(strategy, (StrategyBase, str)):
            raise TypeError("strategy doit être StrategyBase ou str")

        self.logger.debug("✅ Validation des entrées OK")

    def _get_strategy_by_name(self, name: str) -> StrategyBase:
        """Récupère une stratégie par son nom depuis le registre global."""
        from strategies.base import get_strategy, list_strategies  # pylint: disable=import-outside-toplevel

        name_lower = name.lower().replace("-", "_").replace(" ", "_")

        try:
            strategy_class = get_strategy(name_lower)
            return strategy_class()
        except ValueError as exc:
            available = ", ".join(list_strategies())
            raise ValueError(f"Stratégie inconnue: '{name}'. Disponibles: {available}") from exc

    def _calculate_indicators(
        self,
        df: pd.DataFrame,
        strategy: StrategyBase,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcule les indicateurs requis par la stratégie."""
        indicators = {}
        bank = self._get_indicator_bank(params)
        data_hash = bank.get_data_hash(df) if bank is not None else None

        for indicator_name in strategy.required_indicators:
            self.logger.debug(f"  Calcul indicateur: {indicator_name}")

            # Extraire les paramètres spécifiques à l'indicateur
            indicator_params = self._extract_indicator_params(strategy, indicator_name, params)

            try:
                cached_result = None
                if bank is not None:
                    cached_result = bank.get(
                        indicator_name, indicator_params, df, data_hash=data_hash
                    )
                if cached_result is not None:
                    indicators[indicator_name] = cached_result
                else:
                    result = calculate_indicator(indicator_name, df, indicator_params)
                    indicators[indicator_name] = result
                    if bank is not None and result is not None:
                        bank.put(
                            indicator_name,
                            indicator_params,
                            df,
                            result,
                            data_hash=data_hash
                        )
            except Exception as e:
                self.logger.warning(f"  ⚠️ Erreur calcul {indicator_name}: {e}")
                indicators[indicator_name] = None

        return indicators

    def _get_indicator_bank(self, params: Dict[str, Any]):
        cache_enabled = params.get("indicator_cache", True)
        if not cache_enabled:
            return None
        if self.indicator_bank is None:
            cache_dir = params.get("indicator_cache_dir", ".indicator_cache")
            kwargs: Dict[str, Any] = {}
            if "indicator_cache_ttl" in params:
                kwargs["ttl"] = int(params["indicator_cache_ttl"])
            if "indicator_cache_max_size_mb" in params:
                kwargs["max_size_mb"] = float(params["indicator_cache_max_size_mb"])
            if "indicator_cache_memory_entries" in params:
                kwargs["memory_max_entries"] = int(params["indicator_cache_memory_entries"])
            self.indicator_bank = get_indicator_bank(cache_dir=cache_dir, **kwargs)
        return self.indicator_bank

    def _build_execution_engine(self, params: Dict[str, Any]):
        execution_model = params.get("execution_model")
        if execution_model is None:
            execution_model = getattr(self.config, "execution_model", None)
        if not execution_model:
            return None

        from backtest.execution import create_execution_engine  # pylint: disable=import-outside-toplevel

        exec_kwargs: Dict[str, Any] = {}
        for key in (
            "spread_bps",
            "latency_ms",
            "use_volatility_spread",
            "use_volume_slippage",
            "market_impact_bps",
            "min_spread_bps",
            "max_spread_bps",
            "min_slippage_bps",
            "max_slippage_bps",
            "volatility_window",
            "volume_window",
            "volatility_spread_factor",
            "volume_slippage_factor",
            "partial_fill_prob",
            "partial_fill_min",
            "partial_fill_max",
        ):
            if key in params:
                exec_kwargs[key] = params[key]

        slippage_bps = params.get("slippage_bps", self.config.slippage_bps)
        return create_execution_engine(
            model=str(execution_model),
            slippage_bps=slippage_bps,
            **exec_kwargs
        )

    def _extract_indicator_params(
        self,
        strategy,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrait les paramètres spécifiques à un indicateur."""

        # Option 1: Si la stratégie a une méthode get_indicator_params, l'utiliser
        if hasattr(strategy, 'get_indicator_params'):
            return strategy.get_indicator_params(indicator_name, params)

        # Option 2 (fallback): Extraction automatique avec mapping correct
        prefix_map = {
            "bollinger": ("bb_", {"std": "std_dev"}),  # Mapping std → std_dev
            "atr": ("atr_", {}),
            "rsi": ("rsi_", {}),
            "ema": ("ema_", {})
        }

        if indicator_name not in prefix_map:
            # Fallback pour indicateurs non mappés
            prefix = f"{indicator_name}_"
            renames = {}
        else:
            prefix, renames = prefix_map[indicator_name]

        indicator_params = {}

        # Extraire les paramètres avec le préfixe
        for key, value in params.items():
            if key.startswith(prefix):
                # Enlever le préfixe
                clean_key = key[len(prefix):]
                # Appliquer le mapping de renommage
                final_key = renames.get(clean_key, clean_key)
                indicator_params[final_key] = value

        # Paramètres directs (sans préfixe mais reconnus)
        direct_params = {
            "bollinger": ["period", "std_dev"],
            "atr": ["period", "method"],
            "rsi": ["period"],
            "ema": ["period"]
        }

        for param in direct_params.get(indicator_name, []):
            if param in params and param not in indicator_params:
                indicator_params[param] = params[param]

        return indicator_params

    def _get_periods_per_year(self, timeframe: str) -> int:
        """Retourne le nombre de périodes par an pour un timeframe."""
        timeframe_periods = {
            "1m": 365 * 24 * 60,      # 525600
            "5m": 365 * 24 * 12,      # 105120
            "15m": 365 * 24 * 4,      # 35040
            "30m": 365 * 24 * 2,      # 17520
            "1h": 365 * 24,           # 8760
            "4h": 365 * 6,            # 2190
            "1d": 365,                # 365
            "1w": 52                  # 52
        }

        return timeframe_periods.get(timeframe, 365 * 24 * 60)


# Fonction utilitaire pour usage simplifié
def quick_backtest(
    df: pd.DataFrame,
    strategy_name: str = "bollinger_atr",
    **params
) -> RunResult:
    """
    Lance un backtest rapide avec paramètres par défaut.

    Usage:
        result = quick_backtest(df, "bollinger_atr", leverage=3)
    """
    engine = BacktestEngine()
    return engine.run(df, strategy_name, params)


__all__ = ["BacktestEngine", "RunResult", "quick_backtest"]


# Docstring update summary
# - Docstring de module normalisée (LLM-friendly) centrée sur le pipeline
# - Conventions d'unités/normalisations explicitées (signaux 1/-1/0, prix en devise)
# - Read-if/Skip-if ajoutés pour tri rapide
