"""
Module-ID: backtest.performance

Purpose: Calculer les métriques de performance standard et avancées (rendement, risque, Sharpe, trades, etc.).

Role in pipeline: metrics

Key components: PerformanceMetrics, calculate_metrics, drawdown_series, format_metrics_report

Inputs: PnL array, trades list, returns array, capital initial, durée en jours

Outputs: PerformanceMetrics (dataclass complète), rapports formatés

Dependencies: numpy, pandas, backtest.metrics_tier_s, utils.log

Conventions: PnL en devise de base; rendements en fractions [0,1] ou pourcentages; durées en jours; Sharpe annualisé (252 jours).

Read-if: Calcul métriques, rapports perfo, ou modification formules.

Skip-if: Vous n'avez besoin que des résultats bruts sans métriques.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

import numpy as np
import pandas as pd

# Import des métriques Tier S
from backtest.metrics_tier_s import (
    TierSMetrics,
    calculate_tier_s_metrics,
    format_tier_s_report,
)
from metrics_types import normalize_metrics
from utils.log import get_logger

logger = get_logger(__name__)


class PerformanceMetricsDict(TypedDict, total=False):
    total_pnl: float
    total_return_pct: float
    annualized_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    volatility_annual: float
    total_trades: int
    win_rate: float
    win_rate_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float
    avg_trade_pnl: float
    calmar_ratio: float
    risk_reward_ratio: float
    expectancy: float
    account_ruined: bool
    min_equity: float
    tier_s: Optional[Dict[str, Any]]
    sqn: float
    recovery_factor: float
    ulcer_index: float
    martin_ratio: float
    gain_pain_ratio: float
    tier_s_score: float
    tier_s_grade: str


@dataclass
class PerformanceMetrics:
    """Container pour les métriques de performance."""

    # Rendement
    total_pnl: float
    total_return_pct: float
    annualized_return: float
    cagr: float

    # Risque
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: float
    volatility_annual: float

    # Trades
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float
    avg_trade_pnl: float

    # Ratios avancés
    calmar_ratio: float
    risk_reward_ratio: float
    expectancy: float

    # Protection ruine
    account_ruined: bool = False
    min_equity: float = 0.0

    # Métriques Tier S (optionnelles)
    tier_s: Optional[TierSMetrics] = None

    def to_dict(self) -> PerformanceMetricsDict:
        """Convertit en dictionnaire."""
        return {
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility_annual": self.volatility_annual,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "avg_trade_pnl": self.avg_trade_pnl,
            "calmar_ratio": self.calmar_ratio,
            "risk_reward_ratio": self.risk_reward_ratio,
            "expectancy": self.expectancy,
            "account_ruined": self.account_ruined,
            "min_equity": self.min_equity,
            "tier_s": self.tier_s.to_dict() if self.tier_s else None
        }


def equity_curve(
    returns: pd.Series,
    initial_capital: float = 10000.0
) -> pd.Series:
    """
    Calcule la courbe d'équité à partir des rendements.

    Args:
        returns: Série de rendements (fractionnaires)
        initial_capital: Capital initial

    Returns:
        Série d'équité
    """
    if returns.empty:
        return pd.Series([], dtype=np.float64)

    # Nettoyer les données
    returns_clean = returns.dropna()
    returns_clean = returns_clean.clip(-1.0, 10.0)  # Limites raisonnables

    # Calcul cumulatif
    cumulative = (1 + returns_clean).cumprod()
    equity = initial_capital * cumulative

    return equity


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Calcule la série de drawdown.

    Args:
        equity: Courbe d'équité

    Returns:
        Série de drawdown (valeurs négatives, 0 = au pic)
        Clampe à -100% max (ruine du compte)
    """
    if equity.empty:
        return pd.Series([], dtype=np.float64)

    running_max = equity.expanding().max()

    # Protection contre équité négative : calcul puis clamp à -100%
    drawdown_raw = equity - running_max  # Perte absolue
    drawdown = (drawdown_raw / running_max).clip(lower=-1.0)  # Ratio clampe

    return drawdown


def max_drawdown(equity: pd.Series, fast: bool = False) -> float:
    """
    Calcule le drawdown maximum.

    Args:
        equity: Courbe d'équité
        fast: Si True, utilise NumPy pur (4x plus rapide)

    Returns:
        Drawdown maximum en ratio (0.15 = 15%)
    """
    if fast:
        # MODE FAST: NumPy pur, sans conversion pandas
        arr = equity.values if isinstance(equity, pd.Series) else np.asarray(equity)
        if len(arr) < 2:
            return 0.0
        peak = np.maximum.accumulate(arr)
        # Protection division par zéro
        dd = (arr - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.clip(np.min(dd), -1.0, 0.0))  # Clamp à -100%

    # MODE STANDARD: utilise drawdown_series pandas
    if equity.empty:
        return 0.0

    dd = drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,  # Jours de trading par defaut
    method: str = "daily_resample",  # "standard", "trading_days", "daily_resample" ou "fast"
    equity: Optional[pd.Series] = None,  # Necessaire pour daily_resample
    run_id: Optional[str] = None  # Pour logging structuré
) -> float:
    '''
    Calcule le ratio de Sharpe annualise.

    Pour limiter les biais des equity curves "sparse", la methode daily_resample
    peut resampler l'equity en quotidien avant de calculer les rendements.
    Des gardes supplmentaires evitent les valeurs aberrantes lorsque seules
    quelques trades non nuls sont disponibles.

    MODES:
    - "fast": NumPy pur, pas de resample (RECOMMANDÉ pour optimisation)
    - "daily_resample": Resample equity quotidien (plus précis mais 10x plus lent)
    - "standard": Utilise tous les returns directement
    - "trading_days": Filtre les returns nuls
    '''
    # MODE FAST: NumPy pur, pas de conversion pandas, pas de resample
    if method == "fast":
        if isinstance(returns, pd.Series):
            arr = returns.values
        else:
            arr = np.asarray(returns)
        arr = arr[np.isfinite(arr)]  # Supprimer NaN/Inf
        if len(arr) < 3:
            return 0.0
        std = np.std(arr, ddof=1)
        if std < 1e-10:
            return 0.0
        mean_excess = np.mean(arr) - (risk_free / periods_per_year)
        sharpe = (mean_excess * np.sqrt(periods_per_year)) / std
        # Clamp to [-20, 20]
        return float(np.clip(sharpe, -20.0, 20.0))

    returns_series = returns.copy() if isinstance(returns, pd.Series) else pd.Series(returns)
    _ = run_id

    # SHARPE_INPUT - Log entrée (désactivé pour performance)
    # if run_id:
    #     logger.debug(
    #         f"SHARPE_INPUT run_id={run_id} series_type=returns "
    #         f"freq={'daily' if method == 'daily_resample' else 'step'} "
    #         f"risk_free={risk_free} annualization_factor={periods_per_year} "
    #         f"method={method} n_points={len(returns_series)}"
    #     )

    if returns_series.empty:
        # if run_id:  # Désactivé pour performance
        #     logger.warning(
        #         f"SHARPE_ZERO run_id={run_id} reason=returns_empty "
        #         f"n_points=0"
        #     )
        return 0.0

    if method == "daily_resample":
        if equity is None or (hasattr(equity, "empty") and equity.empty):
            # logger.warning("daily_resample necessite equity, fallback sur standard")  # Désactivé pour performance
            method = "standard"
        elif not isinstance(equity.index, pd.DatetimeIndex):
            # logger.warning("equity.index n'est pas DatetimeIndex, fallback sur standard")  # Désactivé pour performance
            method = "standard"
        else:
            equity_daily = equity.resample('D').last().dropna()
            if len(equity_daily) >= 2:
                returns_series = equity_daily.pct_change().dropna()
                periods_per_year = 252  # Annualisation coherente avec des returns quotidiens
            else:
                # if run_id:  # Désactivé pour performance
                #     logger.warning(
                #         f"SHARPE_ZERO run_id={run_id} reason=insufficient_daily_data "
                #         f"days={len(equity_daily)} min_required=2 fallback=use_provided_returns"
                #     )
                # else:
                #     logger.debug(
                #         "sharpe_ratio_insufficient_daily_data days=%s, fallback to provided returns",
                #         len(equity_daily),
                #     )
                pass
            method = "standard"

    returns_clean = (
        pd.Series(returns_series, dtype=np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # SHARPE_SANITY - Désactivé pour performance
    # if run_id:
    #     n_nan = returns_series.isna().sum()
    #     n_zeros = (returns_clean == 0).sum()
    #     n_inf = np.isinf(returns_series.replace([np.inf, -np.inf], np.nan).dropna()).sum()
    #     std_value = returns_clean.std() if len(returns_clean) > 0 else 0.0
    #
    #     logger.debug(
    #         f"SHARPE_SANITY run_id={run_id} n_points={len(returns_clean)} "
    #         f"n_nan={n_nan} n_zeros={n_zeros} n_inf={n_inf} "
    #         f"mean={returns_clean.mean():.6f} std={std_value:.6f} "
    #         f"min={returns_clean.min():.6f} max={returns_clean.max():.6f} "
    #         f"std_near_zero={std_value < 1e-6}"
    #     )

    # GARDES ADAPTATIFS pour backtests courts
    # Détecter durée du backtest pour assouplir les gardes si < 30 jours
    is_short_backtest = False
    backtest_days = None
    if equity is not None and isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 0:
        backtest_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
        is_short_backtest = backtest_days < 30

    # MIN_SAMPLES adaptatif : 2 pour backtests courts, 3 sinon
    min_samples_for_sharpe = 2 if is_short_backtest else 3
    min_non_zero_returns = 2 if is_short_backtest else 3
    if len(returns_clean) < min_samples_for_sharpe:
        # if run_id:  # Désactivé pour performance
        #     logger.warning(
        #         f"SHARPE_ZERO run_id={run_id} reason=min_samples "
        #         f"samples={len(returns_clean)} min_required={min_samples_for_sharpe} "
        #         f"adaptive=True backtest_days={backtest_days if backtest_days else 'N/A'}"
        #     )
        # else:
        #     logger.debug(
        #         "sharpe_ratio_insufficient_samples samples=%s < min=%s, returning 0.0",
        #         len(returns_clean),
        #         min_samples_for_sharpe,
        #     )
        return 0.0

    if method == "trading_days":
        returns_clean = returns_clean[returns_clean != 0.0]
        if len(returns_clean) < min_samples_for_sharpe:
            # if run_id:  # Désactivé pour performance
            #     logger.warning(
            #         f"SHARPE_ZERO run_id={run_id} reason=min_samples_after_trading_days_filter "
            #         f"samples={len(returns_clean)} min_required={min_samples_for_sharpe}"
            #     )
            # else:
            #     logger.debug(
            #         "sharpe_ratio_insufficient_samples_after_filter samples=%s < min=%s, returning 0.0",
            #         len(returns_clean),
            #         min_samples_for_sharpe,
            #     )
            return 0.0
    non_zero_count = int((returns_clean != 0.0).sum())
    if non_zero_count < min_non_zero_returns:
        # if run_id:  # Désactivé pour performance
        #     logger.warning(
        #         f"SHARPE_ZERO run_id={run_id} reason=min_non_zero "
        #         f"non_zero={non_zero_count} min_required={min_non_zero_returns} "
        #         f"total_samples={len(returns_clean)} adaptive=True backtest_days={backtest_days if backtest_days else 'N/A'}"
        #     )
        # else:
        #     logger.debug(
        #         "sharpe_ratio_insufficient_non_zero non_zero=%s < min=%s, returning 0.0",
        #         non_zero_count,
        #         min_non_zero_returns,
        #     )
        return 0.0

    periods_per_year = periods_per_year or 0
    rf_period = risk_free / periods_per_year if periods_per_year else 0.0

    excess_returns = returns_clean - rf_period
    mean_excess = excess_returns.mean()
    std_returns = float(returns_clean.std(ddof=1))

    # min_annual_vol adaptatif : assouplir pour backtests courts ou peu d'échantillons
    needs_relaxed_vol = is_short_backtest or len(returns_clean) < 10
    min_annual_vol = 0.0001 if needs_relaxed_vol else 0.001  # 0.01% vs 0.1%
    min_period_std = min_annual_vol / np.sqrt(periods_per_year or 1)

    if not np.isfinite(std_returns) or std_returns < min_period_std:
        # if run_id:  # Désactivé pour performance
        #     logger.warning(
        #         f"SHARPE_ZERO run_id={run_id} reason=low_volatility "
        #         f"std={std_returns:.6f} min_required={min_period_std:.6f} "
        #         f"samples={len(returns_clean)} is_finite={np.isfinite(std_returns)} "
        #         f"adaptive=True min_annual_vol={min_annual_vol} relaxed={needs_relaxed_vol}"
        #     )
        # else:
        #     logger.debug(
        #         "sharpe_ratio_zero_volatility std=%.6f < min=%.6f, returns=%s samples",
        #         std_returns,
        #         min_period_std,
        #         len(returns_clean),
        #     )
        return 0.0

    # SHARPE_CALC - Désactivé pour performance
    # if run_id:
    #     logger.debug(
    #         f"SHARPE_CALC run_id={run_id} mean_excess={mean_excess:.6f} "
    #         f"std_returns={std_returns:.6f} periods_per_year={periods_per_year}"
    #     )

    sharpe = (mean_excess * np.sqrt(periods_per_year)) / std_returns if periods_per_year else 0.0

    max_sharpe = 20.0
    if abs(sharpe) > max_sharpe:
        # logger.warning(  # Désactivé pour performance
        #     "sharpe_ratio_clamped value=%.2f clamped_to=+/-%.1f std=%.6f mean=%.6f samples=%s",
        #     sharpe,
        #     max_sharpe,
        #     std_returns,
        #     mean_excess,
        #     len(returns_clean),
        # )
        sharpe = np.sign(sharpe) * max_sharpe
    # SHARPE_OUTPUT - Désactivé pour performance
    # if run_id:
    #     logger.info(
    #         f"SHARPE_OUTPUT run_id={run_id} sharpe_final={float(sharpe):.4f} "
    #         f"fallback_reason=clamped "
    #         f"thresholds_applied=min_samples:{min_samples_for_sharpe},min_non_zero:{min_non_zero_returns},min_vol:{min_annual_vol}"
    #     )

    return float(sharpe)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    method: str = "daily_resample",
    equity: Optional[pd.Series] = None
) -> float:
    """
    Calcule le ratio de Sortino (ne pénalise que la volatilité baissière).

    Args:
        returns: Série de rendements
        risk_free: Taux sans risque annuel
        periods_per_year: Nombre de périodes par an pour l'annualisation
        method: "standard", "trading_days", "daily_resample" ou "fast"
        equity: Série d'equity (requis si method="daily_resample")

    Returns:
        Ratio de Sortino annualisé
    """
    # MODE FAST: NumPy pur, pas de resample
    if method == "fast":
        if isinstance(returns, pd.Series):
            arr = returns.values
        else:
            arr = np.asarray(returns)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return 0.0
        rf_period = risk_free / periods_per_year
        excess = arr - rf_period
        mean_excess = np.mean(excess)
        downside = arr[arr < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-10:
            return 0.0
        sortino = (mean_excess * np.sqrt(periods_per_year)) / downside_std
        return float(np.clip(sortino, -20.0, 20.0))

    if isinstance(returns, pd.Series) and returns.empty:
        return 0.0

    # Méthode daily_resample : resample equity en quotidien
    if method == "daily_resample":
        if equity is None or equity.empty:
            logger.warning("daily_resample nécessite equity, fallback sur standard")
            method = "standard"
        else:
            if not isinstance(equity.index, pd.DatetimeIndex):
                logger.warning("equity.index n'est pas DatetimeIndex, fallback sur standard")
                method = "standard"
            else:
                equity_daily = equity.resample('D').last().dropna()
                if len(equity_daily) < 2:
                    return 0.0
                returns = equity_daily.pct_change().dropna()
                method = "standard"
                # ⚠️ IMPORTANT: Après resample quotidien, forcer periods_per_year = 252 (jours de trading)
                periods_per_year = 252

    returns_clean = returns.dropna()
    if returns_clean.empty:
        return 0.0

    # Filtrer returns nuls si méthode trading_days
    if method == "trading_days":
        returns_clean = returns_clean[returns_clean != 0.0]
        if len(returns_clean) < 2:
            return 0.0

    rf_period = risk_free / periods_per_year
    excess_returns = returns_clean - rf_period
    mean_excess = excess_returns.mean()

    # Volatilité baissière seulement
    downside_returns = returns_clean[returns_clean < 0]
    if len(downside_returns) < 2:
        return 0.0

    downside_std = downside_returns.std(ddof=1)

    if downside_std <= 1e-10:
        return 0.0

    sortino = (mean_excess * np.sqrt(periods_per_year)) / downside_std

    return float(sortino)


def profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Calcule le profit factor (gains bruts / pertes brutes).
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0

    gross_profits = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

    if gross_losses == 0:
        return float("inf") if gross_profits > 0 else 1.0

    return gross_profits / gross_losses


def calculate_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    periods_per_year: int = 252,  # Jours de trading par défaut
    include_tier_s: bool = False,
    sharpe_method: str = "daily_resample",  # "standard", "trading_days", "daily_resample" ou "fast"
    run_id: Optional[str] = None  # Pour logging structuré
) -> PerformanceMetricsDict:
    """
    Calcule toutes les métriques de performance.

    Args:
        equity: Courbe d'équité
        returns: Série de rendements (par barre)
        trades_df: DataFrame des trades
        initial_capital: Capital initial
        periods_per_year: Périodes par an pour annualisation du Sharpe
                         (défaut: 252 jours de trading, standard industrie)
        include_tier_s: Inclure métriques Tier S avancées
        sharpe_method: Méthode de calcul du Sharpe/Sortino:
                      - "fast": NumPy pur, pas de resample (10x plus rapide, RECOMMANDÉ pour optimisation)
                      - "daily_resample": Resample equity en quotidien (précis, standard industrie)
                      - "trading_days": Filtre les returns nuls (incomplet)
                      - "standard": Utilise tous les returns directement

    Returns:
        Dict de toutes les métriques

    Notes:
        - Mode "fast" recommandé pour Optuna/sweep (performance 10x)
        - Mode "daily_resample" recommandé pour résultats finaux (précision)
    """
    metrics: PerformanceMetricsDict = {}

    # === Métriques de rendement ===
    if not equity.empty:
        final_equity = equity.iloc[-1]
        total_pnl = final_equity - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100

        # Rendement annualisé (calendrier si index datetime)
        annualized_return = 0.0
        years = 0.0
        if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
            elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
            years = elapsed_days / 365 if elapsed_days > 0 else 0.0
        elif periods_per_year and len(equity) > 1:
            years = len(equity) / periods_per_year

        if years > 0 and final_equity > 0:
            annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        total_pnl = 0.0
        total_return_pct = 0.0
        annualized_return = 0.0

    metrics["total_pnl"] = total_pnl
    metrics["total_return_pct"] = total_return_pct
    metrics["annualized_return"] = annualized_return

    # === Métriques de risque ===
    # Détection de la ruine du compte (avant calcul Sharpe pour pénaliser)
    account_ruined = bool((equity <= 0).any()) if not equity.empty else False
    metrics["account_ruined"] = account_ruined

    if account_ruined:
        min_equity = float(equity.min())
        metrics["min_equity"] = min_equity
        # OPTUNA FIX: Retourner un Sharpe très négatif basé sur le return pour
        # permettre à Optuna de différencier les mauvaises stratégies
        # Plus la perte est grande, plus le Sharpe est négatif
        # -100% return => Sharpe = -10, -200% return => Sharpe = -20, etc.
        synthetic_sharpe = total_return_pct / 10.0  # -112% => -11.2
        synthetic_sharpe = max(synthetic_sharpe, -20.0)  # Clamp à -20 max
        metrics["sharpe_ratio"] = synthetic_sharpe
        metrics["sortino_ratio"] = synthetic_sharpe  # Même logique
        logger.warning(
            "ACCOUNT_RUINED equity_min=%.2f trades_until_ruin=%s synthetic_sharpe=%.2f",
            min_equity,
            len(equity[equity > 0]),
            synthetic_sharpe,
        )
    else:
        metrics["sharpe_ratio"] = sharpe_ratio(
            returns,
            periods_per_year=periods_per_year,
            method=sharpe_method,
            equity=equity,  # Passer equity pour daily_resample
            run_id=run_id  # Propager run_id pour logs
        )
        metrics["sortino_ratio"] = sortino_ratio(
            returns,
            periods_per_year=periods_per_year,
            method=sharpe_method,
            equity=equity  # Passer equity pour daily_resample
        )

    metrics["max_drawdown"] = max_drawdown(equity, fast=(sharpe_method == "fast")) * 100  # En %

    # Log additionnel si ruine détectée (déplacé après calcul)
    if account_ruined:
        logger.warning(
            "ACCOUNT_RUINED equity_min=%.2f trades_until_ruin=%s",
            metrics["min_equity"],
            len(equity[equity > 0]),
        )

    # Volatilité annualisée
    if sharpe_method == "fast":
        # MODE FAST: NumPy pur
        arr = returns.values if isinstance(returns, pd.Series) else np.asarray(returns)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 2:
            vol = float(np.std(arr, ddof=1) * np.sqrt(periods_per_year) * 100)
        else:
            vol = 0.0
        metrics["volatility_annual"] = vol
    else:
        # MODE STANDARD: avec resample si possible
        volatility_returns = returns
        vol_annualization = periods_per_year
        if sharpe_method == "daily_resample" and isinstance(equity.index, pd.DatetimeIndex):
            daily_equity = equity.resample("D").last().dropna()
            if len(daily_equity) >= 2:
                volatility_returns = daily_equity.pct_change().dropna()
                vol_annualization = 252

        if not volatility_returns.empty and vol_annualization:
            vol = volatility_returns.std(ddof=1) * np.sqrt(vol_annualization) * 100
            metrics["volatility_annual"] = vol
        else:
            metrics["volatility_annual"] = 0.0
    metrics["cagr"] = annualized_return

    # Durée max du drawdown
    if not equity.empty:
        dd = drawdown_series(equity)
        if (dd < 0).any():
            if isinstance(dd.index, pd.DatetimeIndex):
                dd_periods = []
                start_ts = None
                last_in_dd_ts = None
                for ts, in_dd in (dd < 0).items():
                    if in_dd:
                        if start_ts is None:
                            start_ts = ts
                        last_in_dd_ts = ts
                    elif start_ts is not None:
                        # Utiliser last_in_dd_ts au lieu de ts pour mesurer la durée réelle du DD
                        if last_in_dd_ts is not None:
                            dd_periods.append(last_in_dd_ts - start_ts)
                        start_ts = None
                        last_in_dd_ts = None
                if start_ts is not None and last_in_dd_ts is not None:
                    dd_periods.append(last_in_dd_ts - start_ts)

                metrics["max_drawdown_duration_days"] = (
                    max((p.total_seconds() / 86400 for p in dd_periods))
                    if dd_periods else 0.0
                )
            else:
                in_dd = dd < 0
                dd_lengths = []
                current = 0
                for val in in_dd:
                    if val:
                        current += 1
                    else:
                        if current > 0:
                            dd_lengths.append(current)
                        current = 0
                if current > 0:
                    dd_lengths.append(current)

                max_dd_bars = max(dd_lengths) if dd_lengths else 0
                metrics["max_drawdown_duration_days"] = max_dd_bars / (periods_per_year or 1)
        else:
            metrics["max_drawdown_duration_days"] = 0.0
    else:
        metrics["max_drawdown_duration_days"] = 0.0

    # === Métriques de trades ===
    if not trades_df.empty and "pnl" in trades_df.columns:
        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        metrics["total_trades"] = n_trades
        metrics["win_rate"] = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
        metrics["profit_factor"] = profit_factor(trades_df)

        metrics["avg_win"] = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        metrics["avg_loss"] = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0
        metrics["largest_win"] = winning_trades["pnl"].max() if len(winning_trades) > 0 else 0
        metrics["largest_loss"] = losing_trades["pnl"].min() if len(losing_trades) > 0 else 0

        # Durée moyenne des trades
        if "entry_ts" in trades_df.columns and "exit_ts" in trades_df.columns:
            entry_ts = pd.to_datetime(trades_df["entry_ts"], errors="coerce")
            exit_ts = pd.to_datetime(trades_df["exit_ts"], errors="coerce")

            if entry_ts.dt.tz is None and exit_ts.dt.tz is not None:
                entry_ts = entry_ts.dt.tz_localize(exit_ts.dt.tz)
            elif entry_ts.dt.tz is not None and exit_ts.dt.tz is None:
                exit_ts = exit_ts.dt.tz_localize(entry_ts.dt.tz)
            elif entry_ts.dt.tz is not None and exit_ts.dt.tz is not None and entry_ts.dt.tz != exit_ts.dt.tz:
                entry_ts = entry_ts.dt.tz_convert("UTC")
                exit_ts = exit_ts.dt.tz_convert("UTC")

            durations = (exit_ts - entry_ts).dt.total_seconds() / 3600
            durations = durations.replace([np.inf, -np.inf], np.nan).dropna()
            metrics["avg_trade_duration_hours"] = durations.mean() if not durations.empty else 0
        else:
            metrics["avg_trade_duration_hours"] = 0

        # Expectancy (espérance mathématique par trade)
        metrics["expectancy"] = trades_df["pnl"].mean() if n_trades > 0 else 0

        # Risk/Reward ratio
        if metrics["avg_loss"] != 0:
            metrics["risk_reward_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"])
        else:
            metrics["risk_reward_ratio"] = 0
    else:
        metrics["total_trades"] = 0
        metrics["win_rate"] = 0
        metrics["profit_factor"] = 0
        metrics["avg_win"] = 0
        metrics["avg_loss"] = 0
        metrics["largest_win"] = 0
        metrics["largest_loss"] = 0
        metrics["avg_trade_duration_hours"] = 0
        metrics["expectancy"] = 0
        metrics["risk_reward_ratio"] = 0

    # === Ratios avancés ===
    # Calmar ratio (rendement annualisé / max drawdown)
    if metrics["max_drawdown"] != 0:
        metrics["calmar_ratio"] = metrics["annualized_return"] / abs(metrics["max_drawdown"])
    else:
        metrics["calmar_ratio"] = 0

    # === Métriques Tier S (optionnel) ===
    if include_tier_s:
        trades_pnl = trades_df["pnl"] if not trades_df.empty and "pnl" in trades_df.columns else pd.Series([])
        tier_s_metrics = calculate_tier_s_metrics(
            returns=returns,
            equity=equity,
            trades_pnl=trades_pnl,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year
        )
        metrics["tier_s"] = tier_s_metrics.to_dict()
        # Ajouter les métriques clés au niveau supérieur
        metrics["sqn"] = tier_s_metrics.sqn
        metrics["recovery_factor"] = tier_s_metrics.recovery_factor
        metrics["ulcer_index"] = tier_s_metrics.ulcer_index
        metrics["martin_ratio"] = tier_s_metrics.martin_ratio
        metrics["gain_pain_ratio"] = tier_s_metrics.gain_pain_ratio
        metrics["tier_s_score"] = tier_s_metrics.tier_s_score
        metrics["tier_s_grade"] = tier_s_metrics.tier_s_grade
    else:
        metrics["tier_s"] = None

    return normalize_metrics(metrics, "pct")


class PerformanceCalculator:
    """
    Calculateur de performance avec API orientée objet.
    """

    def __init__(self, initial_capital: float = 10000.0, include_tier_s: bool = False):
        self.initial_capital = initial_capital
        self.include_tier_s = include_tier_s
        self._last_metrics: Optional[PerformanceMetricsDict] = None
        self._last_tier_s: Optional[TierSMetrics] = None

    def summarize(
        self,
        returns: pd.Series,
        trades_df: pd.DataFrame,
        periods_per_year: int = 252,
        sharpe_method: str = "daily_resample"
    ) -> PerformanceMetricsDict:
        """
        Calcule un résumé complet des performances.

        Args:
            returns: Série de rendements
            trades_df: DataFrame des trades
            periods_per_year: Périodes par an (défaut: 252 jours de trading)
            sharpe_method: Méthode de calcul Sharpe ("daily_resample", "trading_days" ou "standard")

        Returns:
            Dict des métriques calculées
        """
        # Calculer l'équité
        eq = equity_curve(returns, self.initial_capital)

        # Calculer toutes les métriques
        metrics = calculate_metrics(
            equity=eq,
            returns=returns,
            trades_df=trades_df,
            initial_capital=self.initial_capital,
            periods_per_year=periods_per_year,
            include_tier_s=self.include_tier_s,
            sharpe_method=sharpe_method
        )

        self._last_metrics = metrics

        # Stocker les métriques Tier S si calculées
        if self.include_tier_s and metrics.get("tier_s"):
            trades_pnl = trades_df["pnl"] if not trades_df.empty and "pnl" in trades_df.columns else pd.Series([])
            self._last_tier_s = calculate_tier_s_metrics(
                returns=returns,
                equity=eq,
                trades_pnl=trades_pnl,
                initial_capital=self.initial_capital,
                periods_per_year=periods_per_year
            )

        return metrics

    def format_report(self, metrics: Optional[PerformanceMetricsDict] = None) -> str:
        """
        Formate un rapport lisible des métriques.
        """
        if metrics is None:
            metrics = self._last_metrics
        if metrics is None:
            return "Aucune métrique disponible"

        report = """
╔══════════════════════════════════════════════════════════╗
║              RAPPORT DE PERFORMANCE                       ║
╠══════════════════════════════════════════════════════════╣
║ RENDEMENT                                                 ║
║   P&L Total:           ${total_pnl:>12,.2f}               ║
║   Rendement Total:     {total_return_pct:>12.2f}%         ║
║   Rendement Annualisé: {annualized_return:>12.2f}%        ║
╠══════════════════════════════════════════════════════════╣
║ RISQUE                                                    ║
║   Sharpe Ratio:        {sharpe_ratio:>12.2f}              ║
║   Sortino Ratio:       {sortino_ratio:>12.2f}             ║
║   Max Drawdown:        {max_drawdown:>12.2f}%             ║
║   Volatilité Ann.:     {volatility_annual:>12.2f}%        ║
╠══════════════════════════════════════════════════════════╣
║ TRADES                                                    ║
║   Nombre de Trades:    {total_trades:>12d}                ║
║   Win Rate:            {win_rate:>12.1f}%                 ║
║   Profit Factor:       {profit_factor:>12.2f}             ║
║   Gain Moyen:          ${avg_win:>12,.2f}                 ║
║   Perte Moyenne:       ${avg_loss:>12,.2f}                ║
╚══════════════════════════════════════════════════════════╝
""".format(**metrics)  # pylint: disable=consider-using-f-string

        # Ajouter rapport Tier S si disponible
        if self._last_tier_s:
            report += format_tier_s_report(self._last_tier_s)

        return report


# Docstring update summary
# - Docstring de module normalisée (LLM-friendly) centrée sur calcul des métriques
# - Conventions unités (devise, fractions, jours, annualisé) explicitées
# - Read-if/Skip-if ajoutés pour tri rapide


__all__ = [
    "PerformanceCalculator",
    "PerformanceMetrics",
    "PerformanceMetricsDict",
    "TierSMetrics",
    "calculate_metrics",
    "calculate_tier_s_metrics",
    "equity_curve",
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "format_tier_s_report",
]
