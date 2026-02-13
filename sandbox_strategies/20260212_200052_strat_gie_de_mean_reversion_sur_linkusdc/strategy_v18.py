from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_filtered_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "atr_period": 14, "bollinger_period": 20, "bollinger_std_dev": 2, "stoch_rsi_d": 3, "stoch_rsi_k": 3, "stoch_rsi_rsi_period": 14, "stoch_rsi_stoch_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

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
        bb_lower = np.nan_to_num(bb['lower'])
        bb_middle = np.nan_to_num(bb['middle'])
        stoch_rsi = indicators['stoch_rsi']
        stoch_rsi_k = np.nan_to_num(stoch_rsi['k'])
        stoch_rsi_d = np.nan_to_num(stoch_rsi['d'])
        adx_dict = indicators['adx']
        adx = np.nan_to_num(adx_dict['adx'])
        atr = np.nan_to_num(indicators['atr'])

        long_entry = (
            (close <= bb_lower) & 
            (stoch_rsi_k < 15) & 
            (stoch_rsi_d < 20) & 
            (adx < 25)
        )

        long_exit = (
            (close >= bb_middle) | 
            (stoch_rsi_k > 85)
        )

        position = 0
        entry_price = 0.0
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entry[i]:
                    position = 1
                    entry_price = close[i]
                    signals.iloc[i] = 1.0
                else:
                    signals.iloc[i] = 0.0
            elif position == 1:
                if long_exit[i]:
                    position = 0
                    signals.iloc[i] = 0.0
                else:
                    stop_loss = entry_price - (atr[i] * params['stop_atr_mult'])
                    take_profit = entry_price + (atr[i] * params['tp_atr_mult'])
                    if close[i] <= stop_loss or close[i] >= take_profit:
                        position = 0
                        signals.iloc[i] = 0.0
                    else:
                        signals.iloc[i] = 1.0

        return signals