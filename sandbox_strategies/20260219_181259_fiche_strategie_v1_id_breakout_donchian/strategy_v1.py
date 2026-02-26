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
                min_val=2.0,
                max_val=5.0,
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
        # initialize signals series with zeros
        signals = pd.Series(0.0, index=df.index)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals[:warmup] = 0.0

        donchian = indicators['donchian']
        adx_d = indicators['adx']
        close = df['close'].values
        atr = np.nan_to_num(indicators['atr'])
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.5)

        # calculate long and short signals
        long_mask = (close > indicators['donchian']["upper"]) & (adx_d['adx'] > 35)
        short_mask = (close < indicators['donchian']["lower"]) & (adx_d['adx'] > 35)

        # exit conditions
        exit_mask = ((close > indicators['donchian']["middle"]) | (close < indicators['donchian']["middle"])) | (adx_d['adx'] < 25)
        long_exit_mask = long_mask & exit_mask
        short_exit_mask = short_mask & exit_mask

        # apply signals
        signals[long_mask] = 1.0
        signals[long_exit_mask] = 0.0
        signals[short_mask] = -1.0
        signals[short_exit_mask] = 0.0

        # apply ATR-based stop loss and take profit levels
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        stop_long = prev_close - stop_atr_mult * atr
        tp_long = prev_close + tp_atr_mult * atr
        df['bb_stop_long'] = np.where(long_mask, stop_long, np.nan)
        df['bb_tp_long'] = np.where(long_exit_mask, tp_long, np.nan)

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
