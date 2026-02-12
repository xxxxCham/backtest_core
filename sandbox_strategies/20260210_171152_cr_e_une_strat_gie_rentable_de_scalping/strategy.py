from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: Bollinger Band Breakout with ATR Risk Control
    Objective: Crée une stratégie rentable de scalping/swing court sur DOGE en timeframes courts (5min à 30min). 

Contraintes obligatoires :
- Priorité au Profit Factor élevé (> 1.3) and au contrôle du drawdown (max 25%)
- Stop-loss ATR obligatoire (1.5-2.5× ATR) et take-profit ATR (2-4× ATR)
- Pas d'overtrading : filtrer les trades avec ADX ou volatilité
- Favoriser des entrées précises (2-3 indicateurs max) plutôt qu'un grand nombre de trades
- Leverage 1-2× maximum

Indicateurs: bollinger, atr
    """

    def __init__(self):
        super().__init__(name="Bollinger Band Breakout with ATR Risk Control")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                type="float",
                default=1.5,
                description="Stop-loss multiplier in terms of ATR",
                min=0.5,
                max=3.0,
            ),
            "tp_atr_mult": ParameterSpec(
                type="float",
                default=2.5,
                description="Take-profit multiplier in terms of ATR",
                min=1.0,
                max=5.0,
            ),
            "atr_threshold": ParameterSpec(
                type="float",
                default=0.5,
                description="ATR threshold (in percentage of price) to avoid low-volatility trades",
                min=0.01,
                max=5.0,
            ),
            "rsi_overbought": ParameterSpec(
                type="float",
                default=50.0,
                description="RSI overbought level",
                min=20.0,
                max=80.0,
            ),
            "rsi_oversold": ParameterSpec(
                type="float",
                default=50.0,
                description="RSI oversold level",
                min=20.0,
                max=80.0,
            ),
            "adx_threshold": ParameterSpec(
                type="float",
                default=25.0,
                description="ADX threshold to filter out overtrading",
                min=0.0,
                max=100.0,
            ),
            "leverage": ParameterSpec(
                type="float",
                default=1.5,
                description="Maximum leverage",
                min=1.0,
                max=2.0,
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])

        atr_val = np.nan_to_num(indicators["atr"])

        # Get parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)
        atr_threshold = params.get("atr_threshold", 0.5)
        rsi_overbought = params.get("rsi_overbought", 50.0)
        rsi_oversold = params.get("rsi_oversold", 50.0)
        adx_threshold = params.get("adx_threshold", 25.0)
        leverage = params.get("leverage", 1.5)

        # Calculate ATR percentage threshold
        atr_pct = (atr_val / df["close"]) * 100

        # Filter for high volatility
        high_volatility = atr_pct > atr_threshold

        # Get RSI (needed for the proposal)
        rsi_val = np.nan_to_num(indicators["rsi"])
        # The proposal does not explicitly mention using ADX, but the constraint says "Pas d'overtrading : filtrer les trades avec ADX ou volatilité"
        # Let's add an ADX filter as per the constraint (even though the initial proposal didn't include it)
        adx_val = np.nan_to_num(indicators["adx"]["adx"])

        # Generate signals
        # LONG signals
        long_conditions = (
            (df["close"] > upper_bb)
            & (rsi_val > rsi_overbought)
            & (high_volatility)
            & (adx_val < adx_threshold)  # Filter out strong trending markets (ADX low)
        )

        # SHORT signals
        short_conditions = (
            (df["close"] < lower_bb)
            & (rsi_val < rsi_oversold)
            & (high_volatility)
            & (adx_val < adx_threshold)  # Filter out strong trending markets (ADX low)
        )

        # Apply leverage (optional: could be used for position sizing, but signals are 1.0/-1.0)
        # For now, leverage is not directly used in signal generation, as the engine handles risk management

        signals[long_conditions] = 1.0
        signals[short_conditions] = -1.0

        return signals