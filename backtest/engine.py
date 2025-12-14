"""
Backtest Core - Backtest Engine
===============================

Moteur de backtesting simplifié et robuste.

Pipeline:
1. Charger les données (ou recevoir un DataFrame)
2. Calculer les indicateurs requis par la stratégie
3. Générer les signaux de trading
4. Simuler les trades
5. Calculer les métriques de performance
6. Retourner le résultat complet
"""

from __future__ import annotations

import time
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
from indicators.registry import calculate_indicator
from utils.config import Config
from utils.observability import (
    get_obs_logger,
    generate_run_id,
    trace_span,
    safe_stats_df,
    PerfCounters,
    build_diagnostic_summary,
)

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

        self.logger.info("BacktestEngine init capital=%s", initial_capital)

    def run(
        self,
        df: pd.DataFrame,
        strategy: Union[StrategyBase, str],
        params: Optional[Dict[str, Any]] = None,
        *,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m",
        seed: int = 42
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

        Returns:
            RunResult avec equity, returns, trades, metrics et meta

        Raises:
            ValueError: Si données ou paramètres invalides
        """
        # Initialiser counters et contexte
        self.counters = PerfCounters()
        self.counters.start("total")
        
        # Enrichir le logger avec contexte
        self.logger = self.logger.with_context(symbol=symbol, timeframe=timeframe)
        self.logger.info("pipeline_start strategy=%s bars=%s", 
                         strategy if isinstance(strategy, str) else getattr(strategy, 'name', 'custom'),
                         len(df))

        # Seed pour déterminisme
        np.random.seed(seed)

        try:
            # 1. Validation des entrées
            with trace_span(self.logger, "validation"):
                self._validate_inputs(df, strategy, params)

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

            self.logger.debug("params=%s", final_params)

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
            self.counters.start("simulation")
            with trace_span(self.logger, "simulation"):
                if USE_FAST_SIMULATOR:
                    trades_df = simulate_trades_fast(df, signals, final_params)
                else:
                    trades_df = simulate_trades(df, signals, final_params)
            self.counters.stop("simulation")
            self.counters.increment("trades_count", len(trades_df))

            # 7. Calculer équité et rendements (version rapide si disponible)
            self.counters.start("equity")
            if USE_FAST_SIMULATOR:
                equity = calculate_equity_fast(df, trades_df, self.initial_capital)
                returns = calculate_returns_fast(equity)
            else:
                equity = calculate_equity_curve(df, trades_df, self.initial_capital)
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
                periods_per_year=periods_per_year
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

            self.logger.info(
                "pipeline_end duration_ms=%.1f trades=%s sharpe=%.2f pnl=%.2f",
                total_ms, len(trades_df), metrics.get('sharpe_ratio', 0), metrics.get('total_pnl', 0)
            )

            return result

        except Exception as e:
            self.counters.stop("total")
            self.logger.error("pipeline_error error=%s", str(e))
            raise

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        strategy: Union[StrategyBase, str],
        params: Optional[Dict[str, Any]]
    ) -> None:
        """Valide les entrées du backtest."""

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
        from strategies.base import get_strategy, list_strategies

        name_lower = name.lower().replace("-", "_").replace(" ", "_")

        try:
            strategy_class = get_strategy(name_lower)
            return strategy_class()
        except ValueError:
            available = ", ".join(list_strategies())
            raise ValueError(f"Stratégie inconnue: '{name}'. Disponibles: {available}")

    def _calculate_indicators(
        self,
        df: pd.DataFrame,
        strategy: StrategyBase,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcule les indicateurs requis par la stratégie."""
        indicators = {}

        for indicator_name in strategy.required_indicators:
            self.logger.debug(f"  Calcul indicateur: {indicator_name}")

            # Extraire les paramètres spécifiques à l'indicateur
            indicator_params = self._extract_indicator_params(indicator_name, params)

            try:
                indicators[indicator_name] = calculate_indicator(
                    indicator_name, df, indicator_params
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ Erreur calcul {indicator_name}: {e}")
                indicators[indicator_name] = None

        return indicators

    def _extract_indicator_params(
        self,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrait les paramètres spécifiques à un indicateur."""

        # Mapping des préfixes de paramètres
        prefix_map = {
            "bollinger": "bb_",
            "atr": "atr_",
            "rsi": "rsi_",
            "ema": "ema_"
        }

        prefix = prefix_map.get(indicator_name, f"{indicator_name}_")
        indicator_params = {}

        # Extraire les paramètres avec le préfixe
        for key, value in params.items():
            if key.startswith(prefix):
                # Enlever le préfixe
                param_name = key[len(prefix):]
                indicator_params[param_name] = value

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
