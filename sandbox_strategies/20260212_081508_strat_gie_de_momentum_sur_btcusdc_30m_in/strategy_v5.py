from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_trading_btcusdc_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "macd_periods": {"fast": 12, "signal": 9, "slow": 26}, "rsi_overbought": 70, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(type_=(int,), description="Period for ATR calculation"),
            "macd_periods": ParameterSpec(type_=(dict,), description="Periods for MACD calculation", default={"fast": 12, "signal": 9, "slow": 26}),
            "rsi_overbought": ParameterSpec(type_=(int,), description="Overbought level for RSI"),
            "rsi_period": ParameterSpec(type_=(int,), description="Period for RSI calculation"),
            "stop_atr_mult": ParameterSpec(type_=(float,), description="Stop-loss multiplier based on ATR"),
            "tp_atr_mult": ParameterSpec(type_=(float,), description="Take-profit multiplier based on ATR"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        rsi_arr = np.nan_to_num(indicators["rsi"])
        macd_histogram = np.nan_to_num(indicators["macd"]["histogram"])
        atr_arr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params["rsi_overbought"]
        fast_period = params["macd_periods"]["fast"]
        signal_period = params["macd_periods"]["signal"]
        slow_period = params["macd_periods"]["slow"]
        
        for i in range(warmup, len(df)):
            current_rsi = rsi_arr[i]
            prev_rsi = rsi_arr[i-1]
            current_histogram = macd_histogram[i]
            prev_histogram = macd_histogram[i-1]
            current_atr = atr_arr[i]
            
            price = df["close"].iloc[i]
            prev_price = df["close"].iloc[i-1]
            
            if current_rsi > rsi_overbought and current_histogram > 0:
                signals.iloc[i] = -1.0
                continue
                
            if signals.iloc[i-1] == -1.0:
                signals.iloc[i] = 1.0
                continue
                
            if current_rsi < rsi_overbought or current_histogram < 0:
                signals.iloc[i] = 0.0
                continue
                
            signals.iloc[i] = -1.0

        return signals

    def calculate_returns(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        returns = {"total_pips": 0, "win_rate": 0.0, "rr": 0.0, "drawdown": 0.0}
        equity = 10000.0
        max_equity = 10000.0
        trades_list = []
        
        for trade in trades:
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            size = trade["size"]
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]
            
            if exit_time is None:
                continue
                
            pips = size * abs(entry_price - exit_price)
            returns["total_pips"] += pips
            
            if pips > 0:
                returns["win_rate"] += 1
                profit = pips
            else:
                loss = -pips
                
            rr = profit / loss if loss > 0 else 0.0
            returns["rr"] = returns["rr"] + (profit / loss if loss > 0 else 0.0) / len(trades)
            
            max_equity = max(max_equity, equity + profit if profit > 0 else -loss)
            drawdown = (max_equity - equity) / max_equity
            returns["drawdown"] = drawdown
            
            equity = equity + profit if profit > 0 else -loss
            trades_list.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "profit": profit,
                "loss": -loss if loss < 0 else None,
            })
        
        returns["win_rate"] = returns["win_rate"] / len(trades_list) if len(trades_list) > 0 else 0.0
        returns["rr"] = returns["rr"] / len(trades_list) if len(trades_list) > 0 else 0.0
        returns["drawdown"] = drawdown * 100 if len(trades_list) > 0 else 0.0
        returns["trades"] = len(trades_list)
        return returns