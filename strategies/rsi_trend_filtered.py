"""
RSI Reversal avec filtre de tendance.
N'achète en survente que si tendance haussière (EMA rapide > EMA lente).
"""
from typing import Any, Dict, List
import pandas as pd
from indicators.registry import calculate_indicator
from strategies.base import StrategyBase, register_strategy
from utils.parameters import ParameterSpec


@register_strategy("rsi_trend_filtered")
class RSITrendFilteredStrategy(StrategyBase):
    """RSI avec filtre EMA pour éviter trades contre-tendance."""
    
    @property
    def required_indicators(self) -> List[str]:
        return ["rsi"]
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "ema_fast": 20,
            "ema_slow": 50,
            "leverage": 1,
        }
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 14, int, "Période RSI"),
            "oversold_level": ParameterSpec("oversold_level", 10, 40, 30, int, "Survente"),
            "overbought_level": ParameterSpec("overbought_level", 60, 90, 70, int, "Surachat"),
            "ema_fast": ParameterSpec("ema_fast", 10, 50, 20, int, "EMA rapide"),
            "ema_slow": ParameterSpec("ema_slow", 30, 100, 50, int, "EMA lente"),
            "leverage": ParameterSpec("leverage", 1, 5, 1, int, "Levier"),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)
        
        rsi_period = int(params.get("rsi_period", 14))
        ema_fast_period = int(params.get("ema_fast", 20))
        ema_slow_period = int(params.get("ema_slow", 50))

        rsi_values = indicators.get("rsi")
        if rsi_values is None:
            try:
                rsi_values = calculate_indicator(
                    "rsi", df, {"period": rsi_period}
                )
            except Exception:
                return signals

        rsi = pd.Series(rsi_values, index=df.index)
        close = df["close"]
        
        # EMA pour filtre de tendance
        ema_fast = close.ewm(
            span=ema_fast_period, adjust=False
        ).mean()
        ema_slow = close.ewm(
            span=ema_slow_period, adjust=False
        ).mean()
        
        # Filtre de tendance
        uptrend = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow
        
        # Signaux RSI
        rsi_prev = rsi.shift(1)
        rsi_oversold = (rsi < params["oversold_level"]) & (rsi_prev >= params["oversold_level"])
        rsi_overbought = (rsi > params["overbought_level"]) & (rsi_prev <= params["overbought_level"])
        
        # LONG seulement si tendance haussière
        signals[rsi_oversold & uptrend] = 1.0
        
        # SHORT seulement si tendance baissière
        signals[rsi_overbought & downtrend] = -1.0
        
        return signals
    
    def describe(self) -> str:
        return "RSI Reversal avec filtre de tendance EMA (évite trades contre-tendance)"
