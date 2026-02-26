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
        return {'adx_period': 17,
         'atr_period': 14,
         'donchian_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 2.5,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=17,
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
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
        # Boolean masks for long and short entry
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        close = np.nan_to_num(df["close"].values)
        dc = indicators['donchian']
        adx_d = indicators['adx']
        atr_arr = np.nan_to_num(indicators['atr'])

        indicators['donchian']['upper'] = np.nan_to_num(dc["upper"])
        indicators['donchian']['lower'] = np.nan_to_num(dc["lower"])
        indicators['donchian']['middle'] = np.nan_to_num(dc["middle"])
        adx_val = np.nan_to_num(adx_d["adx"])

        # Entry conditions
        long_mask = (close > indicators['donchian']['upper']) & (adx_val > 25)
        short_mask = (close < indicators['donchian']['lower']) & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based stop/TP levels on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.25)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        df.loc[signals == 1.0, "bb_stop_long"] = close[signals == 1.0] - stop_atr_mult * atr_arr[signals == 1.0]
        df.loc[signals == 1.0, "bb_tp_long"] = close[signals == 1.0] + tp_atr_mult * atr_arr[signals == 1.0]
        df.loc[signals == -1.0, "bb_stop_short"] = close[signals == -1.0] + stop_atr_mult * atr_arr[signals == -1.0]
        df.loc[signals == -1.0, "bb_tp_short"] = close[signals == -1.0] - tp_atr_mult * atr_arr[signals == -1.0]
        signals.iloc[:warmup] = 0.0
        return signals
