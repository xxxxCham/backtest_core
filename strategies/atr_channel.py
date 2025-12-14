"""
Backtest Core - ATR Channel Strategy
====================================

Stratégie de breakout basée sur les canaux ATR.
Achète sur cassure de la bande supérieure (EMA + ATR*mult).
Vend sur cassure de la bande inférieure (EMA - ATR*mult).

Indicateurs requis:
- ATR: Average True Range pour la volatilité
- EMA: Moyenne mobile exponentielle pour le centre du canal
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, StrategyResult, register_strategy
from utils.parameters import ParameterSpec


@register_strategy("atr_channel")
class ATRChannelStrategy(StrategyBase):
    """
    Stratégie de canal ATR (breakout volatilité).

    Signaux:
        - LONG (+1): Prix casse au-dessus de EMA + ATR * mult
        - SHORT (-1): Prix casse en-dessous de EMA - ATR * mult

    Paramètres:
        - atr_period: Période ATR et EMA (défaut: 14)
        - atr_mult: Multiplicateur ATR pour les bandes (défaut: 2.0)
        - leverage: Multiplicateur de position (défaut: 1)
    """

    def __init__(self, name: str = "ATRChannel"):
        super().__init__(name)

    @property
    def required_indicators(self) -> List[str]:
        """Indicateurs requis par la stratégie."""
        return ["atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        """Paramètres par défaut."""
        return {
            "atr_period": 14,
            "atr_mult": 2.0,
            "leverage": 1,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Spécifications des paramètres pour l'UI et l'optimisation."""
        return {
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=7,
                max_val=30,
                default=14,
                param_type=int,
                description="Période ATR et EMA"
            ),
            "atr_mult": ParameterSpec(
                name="atr_mult",
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type=float,
                description="Multiplicateur ATR pour les bandes"
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=10,
                default=1,
                param_type=int,
                description="Levier de trading"
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Génère les signaux de trading basés sur le canal ATR.

        Args:
            df: DataFrame OHLCV
            indicators: Dictionnaire contenant 'atr' et 'ema'
            params: Paramètres de la stratégie

        Returns:
            Series de signaux (-1, 0, +1)
        """
        signals = pd.Series(0.0, index=df.index)

        # Récupérer ATR
        if "atr" not in indicators or indicators["atr"] is None:
            return signals
        atr_values = indicators["atr"]

        # Récupérer ou calculer EMA
        if "ema" in indicators and indicators["ema"] is not None:
            ema_values = indicators["ema"]
        else:
            # Calcul interne si non fourni
            period = int(params.get("atr_period", 14))
            if len(df) < period:
                # Pas assez de données pour calculer l'EMA
                return signals
            ema_values = df["close"].ewm(span=period, adjust=False).mean().values

        # Convertir en Series si nécessaire
        if isinstance(atr_values, np.ndarray):
            atr_values = pd.Series(atr_values, index=df.index)
        if isinstance(ema_values, np.ndarray):
            ema_values = pd.Series(ema_values, index=df.index)

        mult = params.get("atr_mult", 2.0)

        # Calculer les bandes du canal
        upper_band = ema_values + (atr_values * mult)
        lower_band = ema_values - (atr_values * mult)

        close = df["close"]
        close_prev = close.shift(1)
        upper_prev = upper_band.shift(1)
        lower_prev = lower_band.shift(1)

        # Signal LONG: breakout haussier
        breakout_up = (close > upper_band) & (close_prev <= upper_prev)
        signals[breakout_up] = 1.0

        # Signal SHORT: breakout baissier
        breakout_down = (close < lower_band) & (close_prev >= lower_prev)
        signals[breakout_down] = -1.0

        return signals

    def describe(self) -> str:
        """Description de la stratégie."""
        return (
            "ATR Channel Strategy: Génère des signaux sur les cassures "
            "du canal ATR (EMA ± ATR * multiplicateur). "
            "Achat sur breakout haut, vente sur breakout bas."
        )

    def run(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any] = None
    ) -> StrategyResult:
        """
        Exécute la stratégie.
        """
        if params is None:
            params = self.default_params

        signals = self.generate_signals(df, indicators, params)

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()

        self._last_result = StrategyResult(
            signals=signals,
            params_used=params,
            metadata={
                "strategy": self.name,
                "total_signals": n_long + n_short,
                "long_signals": int(n_long),
                "short_signals": int(n_short),
                "atr_period": params.get("atr_period", 14),
                "atr_mult": params.get("atr_mult", 2.0),
            }
        )

        return self._last_result
