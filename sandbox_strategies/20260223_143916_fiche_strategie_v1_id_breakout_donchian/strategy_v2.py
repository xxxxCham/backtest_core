from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 16,
         'atr_period': 14,
         'donchian_period': 10,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=16,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Prepare indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        dc = indicators['donchian']
        indicators['donchian']['upper'] = np.nan_to_num(dc["upper"])
        indicators['donchian']['middle'] = np.nan_to_num(dc["middle"])
        indicators['donchian']['lower'] = np.nan_to_num(dc["lower"])
        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])

        # Helper for cross_any
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_middle = np.roll(indicators['donchian']['middle'], 1)
        prev_middle[0] = np.nan
        cross_up = (close > indicators['donchian']['middle']) & (prev_close <= prev_middle)
        cross_down = (close < indicators['donchian']['middle']) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down

        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Entry conditions
        long_mask = (close > indicators['donchian']['upper']) & (adx_val > 25.0)
        short_mask = (close < indicators['donchian']['lower']) & (adx_val > 25.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit condition: cross_any or adx < 20
        exit_mask = cross_any | (adx_val < 20.0)
        signals[exit_mask] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
