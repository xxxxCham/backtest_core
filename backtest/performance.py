"""
Backtest Core - Performance Calculator
======================================

Calcul des métriques de performance standard et avancées.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.log import get_logger

# Import des métriques Tier S
from backtest.metrics_tier_s import (
    TierSMetrics,
    calculate_tier_s_metrics,
    format_tier_s_report,
)

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container pour les métriques de performance."""

    # Rendement
    total_pnl: float
    total_return_pct: float
    annualized_return: float

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

    # Ratios avancés
    calmar_ratio: float
    risk_reward_ratio: float
    expectancy: float
    
    # Métriques Tier S (optionnelles)
    tier_s: Optional[TierSMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
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
            "calmar_ratio": self.calmar_ratio,
            "risk_reward_ratio": self.risk_reward_ratio,
            "expectancy": self.expectancy,
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
    """
    if equity.empty:
        return pd.Series([], dtype=np.float64)

    running_max = equity.expanding().max()
    drawdown = (equity / running_max) - 1.0

    return drawdown


def max_drawdown(equity: pd.Series) -> float:
    """Calcule le drawdown maximum."""
    if equity.empty:
        return 0.0

    dd = drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,  # Jours de trading par défaut
    method: str = "daily_resample",  # "standard", "trading_days" ou "daily_resample"
    equity: Optional[pd.Series] = None  # Nécessaire pour daily_resample
) -> float:
    """
    Calcule le ratio de Sharpe annualisé.

    IMPORTANT: Pour éviter les biais liés aux returns "sparse" (equity qui ne change
    qu'aux trades), cette fonction peut resampler l'equity en fréquence quotidienne.

    Args:
        returns: Série de rendements (fractionnaires, ex: 0.01 = 1%)
        risk_free: Taux sans risque annuel (défaut: 0.0)
        periods_per_year: Nombre de périodes par an pour l'annualisation
                         (défaut: 252 jours de trading)
        method: Méthode de calcul:
                - "standard": Utilise tous les returns (peut donner des valeurs aberrantes)
                - "trading_days": Filtre les returns nuls (incomplet, non recommandé)
                - "daily_resample": Resample equity en quotidien (RECOMMANDÉ, standard industrie)
        equity: Série d'equity (requis si method="daily_resample")

    Returns:
        Ratio de Sharpe annualisé

    Notes:
        - Si std == 0 (rendements constants), retourne 0.0 pour éviter division par zéro
        - Si < 2 observations non-nulles, retourne 0.0 (Sharpe non calculable)
        - La méthode "daily_resample" est recommandée et évite tous les biais liés
          aux equity sparse (qui ne changent qu'aux trades)
    """
    # Gérer numpy array et pandas Series
    if isinstance(returns, np.ndarray):
        if returns.size == 0:
            return 0.0
    elif hasattr(returns, 'empty') and returns.empty:
        return 0.0

    # Méthode daily_resample : resample equity en quotidien
    if method == "daily_resample":
        if equity is None or (hasattr(equity, 'empty') and equity.empty):
            logger.warning("daily_resample nécessite equity, fallback sur standard")
            method = "standard"
        else:
            # Resample equity en fréquence quotidienne
            if not isinstance(equity.index, pd.DatetimeIndex):
                logger.warning("equity.index n'est pas DatetimeIndex, fallback sur standard")
                method = "standard"
            else:
                # Resample en prenant la dernière valeur de chaque jour
                equity_daily = equity.resample('D').last().dropna()

                if len(equity_daily) < 2:
                    return 0.0

                # Calculer returns quotidiens
                returns = equity_daily.pct_change().dropna()

                # Continuer avec la méthode standard sur ces returns quotidiens
                method = "standard"
                # periods_per_year reste 252 (jours de trading)

    # Gérer numpy array et pandas Series pour dropna
    if isinstance(returns, np.ndarray):
        returns_clean = returns[~np.isnan(returns)]
    else:
        returns_clean = returns.dropna()
    
    if len(returns_clean) < 2:
        return 0.0

    # Filtrer les returns nuls si méthode trading_days
    if method == "trading_days":
        returns_clean = returns_clean[returns_clean != 0.0]
        if len(returns_clean) < 2:
            return 0.0

    # Taux sans risque par période
    rf_period = risk_free / periods_per_year

    excess_returns = returns_clean - rf_period
    mean_excess = excess_returns.mean()
    std_returns = returns_clean.std(ddof=1)

    # ⚠️ GARDE EPSILON RENFORCÉE pour éviter Sharpe aberrants
    # Si variance trop faible (<0.1% annualisé), considérer comme constant
    min_annual_vol = 0.001  # 0.1% minimum de volatilité annualisée
    min_period_std = min_annual_vol / np.sqrt(periods_per_year)
    
    if std_returns < min_period_std:
        # Volatilité trop faible : rendements quasi-constants
        # ⚠️ Retourner 0 au lieu de Sharpe aberrant (±50, ±100, etc.)
        logger.debug(
            "sharpe_ratio_zero_volatility std=%.6f < min=%.6f, returns=%s samples",
            std_returns, min_period_std, len(returns_clean)
        )
        return 0.0

    # Annualisation
    sharpe = (mean_excess * np.sqrt(periods_per_year)) / std_returns
    
    # ⚠️ PLAFONNEMENT pour éviter valeurs aberrantes dues à variance instable
    # En réalité, un Sharpe >10 est extrêmement rare (hedge funds top: 3-5)
    MAX_SHARPE = 20.0
    if abs(sharpe) > MAX_SHARPE:
        logger.warning(
            "sharpe_ratio_clamped value=%.2f clamped_to=±%.1f std=%.6f mean=%.6f samples=%s",
            sharpe, MAX_SHARPE, std_returns, mean_excess, len(returns_clean)
        )
        sharpe = np.sign(sharpe) * MAX_SHARPE

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
        method: "standard", "trading_days" ou "daily_resample"
        equity: Série d'equity (requis si method="daily_resample")

    Returns:
        Ratio de Sortino annualisé
    """
    if returns.empty:
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
    sharpe_method: str = "daily_resample"  # "standard", "trading_days" ou "daily_resample"
) -> Dict[str, Any]:
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
                      - "daily_resample": Resample equity en quotidien (RECOMMANDÉ, standard industrie)
                      - "trading_days": Filtre les returns nuls (incomplet, non recommandé)
                      - "standard": Utilise tous les returns (peut donner valeurs aberrantes)

    Returns:
        Dict de toutes les métriques

    Notes:
        - Le Sharpe/Sortino sont calculés avec periods_per_year=252 par défaut
          (jours de trading), indépendamment du timeframe des données
        - La méthode "daily_resample" évite les biais liés aux equity "sparse"
          (qui ne changent qu'aux trades, créant beaucoup de returns nuls)
    """
    metrics = {}

    # === Métriques de rendement ===
    if not equity.empty:
        final_equity = equity.iloc[-1]
        total_pnl = final_equity - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100

        # Rendement annualisé
        n_periods = len(equity)
        if n_periods > 1 and final_equity > 0:
            years = n_periods / periods_per_year
            if years > 0:
                annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
    else:
        total_pnl = 0.0
        total_return_pct = 0.0
        annualized_return = 0.0

    metrics["total_pnl"] = total_pnl
    metrics["total_return_pct"] = total_return_pct
    metrics["annualized_return"] = annualized_return

    # === Métriques de risque ===
    metrics["sharpe_ratio"] = sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        method=sharpe_method,
        equity=equity  # Passer equity pour daily_resample
    )
    metrics["sortino_ratio"] = sortino_ratio(
        returns,
        periods_per_year=periods_per_year,
        method=sharpe_method,
        equity=equity  # Passer equity pour daily_resample
    )
    metrics["max_drawdown"] = max_drawdown(equity) * 100  # En %

    # Volatilité annualisée
    if not returns.empty:
        vol = returns.std() * np.sqrt(periods_per_year) * 100
        metrics["volatility_annual"] = vol
    else:
        metrics["volatility_annual"] = 0.0

    # Durée max du drawdown
    if not equity.empty:
        dd = drawdown_series(equity)
        if (dd < 0).any():
            # Trouver les périodes en drawdown
            in_dd = dd < 0
            # Compter les barres consécutives
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
            # Convertir en jours (approximatif selon timeframe)
            metrics["max_drawdown_duration_days"] = max_dd_bars / (24 * 60)  # Assume 1m
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
            durations = (trades_df["exit_ts"] - trades_df["entry_ts"]).dt.total_seconds() / 3600
            metrics["avg_trade_duration_hours"] = durations.mean()
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

    return metrics


class PerformanceCalculator:
    """
    Calculateur de performance avec API orientée objet.
    """

    def __init__(self, initial_capital: float = 10000.0, include_tier_s: bool = False):
        self.initial_capital = initial_capital
        self.include_tier_s = include_tier_s
        self._last_metrics: Optional[Dict[str, Any]] = None
        self._last_tier_s: Optional[TierSMetrics] = None

    def summarize(
        self,
        returns: pd.Series,
        trades_df: pd.DataFrame,
        periods_per_year: int = 252,
        sharpe_method: str = "daily_resample"
    ) -> Dict[str, Any]:
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

    def format_report(self, metrics: Optional[Dict[str, Any]] = None) -> str:
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
""".format(**metrics)

        # Ajouter rapport Tier S si disponible
        if self._last_tier_s:
            report += format_tier_s_report(self._last_tier_s)

        return report


__all__ = [
    "PerformanceCalculator",
    "PerformanceMetrics",
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
