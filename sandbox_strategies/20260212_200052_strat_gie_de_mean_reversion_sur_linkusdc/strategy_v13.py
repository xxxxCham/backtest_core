from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_filtered_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 1.0, "stoch_rsi_overbought": 75, "stoch_rsi_oversold": 25, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        close = df['close'].values
        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb['upper'])
        bb_middle = np.nan_to_num(bb['middle'])
        bb_lower = np.nan_to_num(bb['lower'])
        stoch_rsi = indicators['stoch_rsi']
        stoch_rsi_k = np.nan_to_num(stoch_rsi['k'])
        atr = np.nan_to_num(indicators['atr'])
        
        stoch_rsi_oversold = params['stoch_rsi_oversold']
        stoch_rsi_overbought = params['stoch_rsi_overbought']
        stop_atr_mult = params['stop_atr_mult']
        tp_atr_mult = params['tp_atr_mult']
        
        long_entry = (close <= bb_lower) & (stoch_rsi_k < stoch_rsi_oversold)
        long_exit = (close >= bb_middle) | (stoch_rsi_k > stoch_rsi_overbought)
        
        position = 0
        entry_price = 0.0
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entry[i]:
                    signals.iloc[i] = 1.0
                    position = 1
                    entry_price = close[i]
            elif position == 1:
                stop_loss = entry_price - stop_atr_mult * atr[i]
                take_profit = entry_price + tp_atr_mult * atr[i]
                if close[i] <= stop_loss or close[i] >= take_profit or long_exit[i]:
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    signals.iloc[i] = 1.0
        
        return signals