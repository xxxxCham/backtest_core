from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='donchian_adx_ema_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 4.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=4.0,
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

        # Prepare indicator arrays
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        dc = indicators['donchian']
        # Ensure sub‑keys are numeric arrays
        dc_upper = np.nan_to_num(dc["upper"])
        dc_middle = np.nan_to_num(dc["middle"])
        dc_lower = np.nan_to_num(dc["lower"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (close > dc_upper) & (adx_val > 25) & (close > ema)
        short_mask = (close < dc_lower) & (adx_val > 25) & (close < ema)

        # Apply warmup
        signals.iloc[:warmup] = 0.0

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_close = np.roll(close, 1)
        prev_mid = np.roll(dc_middle, 1)
        prev_close[0] = np.nan
        prev_mid[0] = np.nan
        cross_any = ((close > dc_middle) & (prev_close <= prev_mid)) | ((close < dc_middle) & (prev_close >= prev_mid))
        exit_mask = cross_any | (adx_val < 20)
        signals[exit_mask] = 0.0

        # ATR-based SL/TP columns
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 4.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        signals.iloc[:warmup] = 0.0
        return signals