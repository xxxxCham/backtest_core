"""
Backtest Core - Bollinger Dual Strategy
========================================

Stratégie basée sur les bandes de Bollinger avec double condition d'entrée:
- Signal LONG : Prix < Bande Basse + Franchissement haussier MA
- Signal SHORT: Prix > Bande Haute + Franchissement baissier MA
- Trailing stop dynamique à partir de la médiane

Basé sur ThreadX Framework.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, register_strategy
from utils.parameters import ParameterSpec, Preset


@dataclass
class BollingerDualParams:
    """Paramètres de la stratégie Bollinger Dual."""

    # Bollinger Bands
    bb_window: int = 20
    bb_std: float = 2.0

    # Moving Average pour franchissement
    ma_window: int = 10
    ma_type: str = 'sma'  # 'sma' ou 'ema'

    # Trailing Stop (0.8 = 80% entre bande et médiane)
    trailing_pct: float = 0.8
    median_activated: bool = True

    # Stop Loss fixe SHORT (0.37 = 37% au-dessus entrée)
    short_stop_pct: float = 0.37

    # Risk Management
    max_hold_bars: int = 100


# Preset par défaut avec ParameterSpec
BOLLINGER_DUAL_PRESET = Preset(
    name="bollinger_dual_default",
    description="Bollinger Dual - Configuration par défaut",
    parameters={
        "bb_window": ParameterSpec(
            name="bb_window",
            min_val=10,
            max_val=50,
            default=20,
            param_type="int",
            description="Période Bollinger"
        ),
        "bb_std": ParameterSpec(
            name="bb_std",
            min_val=1.5,
            max_val=3.5,
            default=2.0,
            param_type="float",
            description="Multiplicateur écart-type"
        ),
        "ma_window": ParameterSpec(
            name="ma_window",
            min_val=5,
            max_val=30,
            default=10,
            param_type="int",
            description="Période MA franchissement"
        ),
        "trailing_pct": ParameterSpec(
            name="trailing_pct",
            min_val=0.5,
            max_val=1.0,
            default=0.8,
            param_type="float",
            description="Trailing stop (0.8 = 80%)"
        ),
        "short_stop_pct": ParameterSpec(
            name="short_stop_pct",
            min_val=0.1,
            max_val=0.5,
            default=0.37,
            param_type="float",
            description="Stop loss SHORT en %"
        ),
    },
    indicators=["bollinger", "sma", "ema"],
    default_granularity=0.5,
)


@register_strategy("bollinger_dual")
class BollingerDualStrategy(StrategyBase):
    """
    Stratégie Bollinger Dual.

    Logique de trading:
    1. LONG : Prix < Bande Basse + Franchissement haussier MA
    2. SHORT: Prix > Bande Haute + Franchissement baissier MA

    Cette version génère des signaux (1=long, -1=short, 0=flat).
    Le trailing stop est géré au niveau du simulateur.
    """

    name = "bollinger_dual"
    description = "Bollinger Bands + MA crossover with dual entry conditions"

    @property
    def required_indicators(self) -> List[str]:
        """Liste des indicateurs nécessaires."""
        return ["bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        """Paramètres par défaut."""
        return {
            "bb_window": 20,
            "bb_std": 2.0,
            "ma_window": 10,
            "ma_type": "sma",
            "trailing_pct": 0.8,
            "short_stop_pct": 0.37,
        }

    @property
    def param_ranges(self) -> Dict[str, tuple]:
        """Plages de paramètres pour optimisation."""
        return {
            "bb_window": (10, 50),
            "bb_std": (1.5, 3.5),
            "ma_window": (5, 30),
            "trailing_pct": (0.5, 1.0),
            "short_stop_pct": (0.1, 0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Génère les signaux de trading.

        Args:
            df: DataFrame OHLCV
            indicators: Dict des indicateurs précalculés
            params: Paramètres de la stratégie

        Returns:
            pd.Series de signaux (1=long, -1=short, 0=flat)
        """
        bb_window = int(params.get("bb_window", 20))
        bb_std = float(params.get("bb_std", 2.0))
        ma_window = int(params.get("ma_window", 10))
        ma_type = params.get("ma_type", "sma")

        close = df["close"].values
        n = len(close)

        # Récupérer Bollinger
        from indicators import bollinger_bands, ema, sma

        upper, middle, lower = bollinger_bands(
            df["close"], period=bb_window, std_dev=bb_std
        )

        # Calculer MA
        if ma_type == "ema":
            ma_values = ema(df["close"], period=ma_window)
        else:
            ma_values = sma(df["close"], period=ma_window)

        # Initialisation signaux
        signals = np.zeros(n, dtype=int)

        # Période de warmup
        warmup = max(bb_window, ma_window) + 1

        for i in range(warmup, n):
            # Skip si données invalides
            if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ma_values[i]):
                continue

            # Détection franchissement MA
            ma_cross_up = close[i] > ma_values[i] and close[i - 1] <= ma_values[i - 1]
            ma_cross_down = close[i] < ma_values[i] and close[i - 1] >= ma_values[i - 1]

            # LONG: Prix < Bande Basse + Franchissement haussier MA
            if close[i] < lower[i] and ma_cross_up:
                signals[i] = 1

            # SHORT: Prix > Bande Haute + Franchissement baissier MA
            elif close[i] > upper[i] and ma_cross_down:
                signals[i] = -1

        return pd.Series(signals, index=df.index)

    def calculate_indicators(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcule les indicateurs nécessaires.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de la stratégie

        Returns:
            Dict avec les indicateurs calculés
        """
        from indicators import bollinger_bands, ema, sma

        bb_window = int(params.get("bb_window", 20))
        bb_std = float(params.get("bb_std", 2.0))
        ma_window = int(params.get("ma_window", 10))
        ma_type = params.get("ma_type", "sma")

        upper, middle, lower = bollinger_bands(
            df["close"], period=bb_window, std_dev=bb_std
        )

        if ma_type == "ema":
            ma_values = ema(df["close"], period=ma_window)
        else:
            ma_values = sma(df["close"], period=ma_window)

        return {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "ma": ma_values,
        }


__all__ = ["BollingerDualStrategy", "BollingerDualParams", "BOLLINGER_DUAL_PRESET"]
