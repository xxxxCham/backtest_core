from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_adjusted')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_entry_threshold': 20,
         'adx_exit_threshold': 15,
         'leverage': 1,
         'stop_atr_mult': 2.25,
         'tp_atr_mult': 5.5,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_entry_threshold': ParameterSpec(
                name='adx_entry_threshold',
                min_val=10,
                max_val=35,
                default=20,
                param_type='int',
                step=1,
            ),
            'adx_exit_threshold': ParameterSpec(
                name='adx_exit_threshold',
                min_val=5,
                max_val=30,
                default=15,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.5,
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

        # Extract indicator arrays
        close = df["close"].values
        donchian = indicators['donchian']
        upper = np.nan_to_num(indicators['donchian']["upper"])
        lower = np.nan_to_num(indicators['donchian']["lower"])
        middle = np.nan_to_num(indicators['donchian']["middle"])
        adx = indicators['adx']
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        adx_entry = params.get("adx_entry_threshold", 20)
        adx_exit = params.get("adx_exit_threshold", 15)
        stop_atr = params.get("stop_atr_mult", 2.25)
        tp_atr = params.get("tp_atr_mult", 5.5)

        # Entry conditions
        long_mask = (close > upper) & (adx_val > adx_entry)
        short_mask = (close < lower) & (adx_val > adx_entry)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warm‑up protection
        signals.iloc[:50] = 0.0

        # ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
