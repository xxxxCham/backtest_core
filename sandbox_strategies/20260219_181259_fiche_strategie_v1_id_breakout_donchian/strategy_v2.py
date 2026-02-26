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
        return {'leverage': 1, 'stop_atr_mult': 2.25, 'tp_atr_mult': 3.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            donchian = indicators['donchian']
            adx_d = indicators['adx']
            atr = np.nan_to_num(indicators['atr'])
            close = df['close'].values

            long_mask = (close > indicators['donchian']["upper"]) & (np.nan_to_num(adx_d['adx']) > 35)
            short_mask = (close < indicators['donchian']["lower"]) & (np.nan_to_num(adx_d['adx']) > 35)

            prev_close = np.roll(close, 1)
            prev_close[0] = np.nan

            signals[long_mask] = 1.0
            signals[short_mask] = -1.0

            stop_atr_mult = params["stop_atr_mult"]
            tp_atr_mult = params["tp_atr_mult"]

            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan

            long_entry_mask = (signals == 1.0)
            df.loc[long_entry_mask, "bb_stop_long"] = close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
            df.loc[long_entry_mask, "bb_tp_long"] = close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
