from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_trxusdc_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'supertrend', 'vwap']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_volatility_threshold': 0.0005,
         'leverage': 1,
         'stop_atr_mult': 2.4,
         'tp_atr_mult': 6.1,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_volatility_threshold': ParameterSpec(
                name='atr_volatility_threshold',
                min_val=0.0001,
                max_val=0.002,
                default=0.0005,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0,
                max_val=10.0,
                default=6.1,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        supertrend_line = np.nan_to_num(indicators['supertrend']["supertrend"])
        vwap = np.nan_to_num(indicators['vwap'])

        vol_high = atr > params["atr_volatility_threshold"]

        long_breakout = vol_high & (close > supertrend_line)
        long_mean_rev = (~vol_high) & (close < vwap)
        long_mask = long_breakout | long_mean_rev

        short_breakout = vol_high & (close < supertrend_line)
        short_mean_rev = (~vol_high) & (close > vwap)
        short_mask = short_breakout | short_mean_rev

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down = (close < supertrend_line) & (prev_close >= supertrend_line)
        cross_up = (close > supertrend_line) & (prev_close <= supertrend_line)

        long_exit = cross_down
        short_exit = cross_up

        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
