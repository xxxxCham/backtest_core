from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_sur_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        # Cross detection
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_bb_lower[0] = np.nan
        cross_up_ema = (close > ema) & (prev_ema <= ema)
        cross_down_ema = (close < ema) & (prev_ema >= ema)
        # Entry conditions
        long_condition = (close > indicators['bollinger']['upper']) & (volume_osc > 0) & (st_direction > 0)
        short_condition = (close < indicators['bollinger']['lower']) & (volume_osc < 0) & (st_direction < 0)
        long_mask = cross_up_ema & long_condition
        short_mask = cross_down_ema & short_condition
        # Exit conditions
        exit_long = cross_down_ema | ((indicators['bollinger']['middle'] - close) / indicators['bollinger']['middle'] > 0.2)
        exit_short = cross_up_ema | ((close - indicators['bollinger']['middle']) / indicators['bollinger']['middle'] > 0.2)
        # Apply exits
        exit_long_mask = np.zeros(n, dtype=bool)
        exit_short_mask = np.zeros(n, dtype=bool)
        prev_exit_long = np.roll(exit_long, 1)
        prev_exit_long[0] = False
        prev_exit_short = np.roll(exit_short, 1)
        prev_exit_short[0] = False
        exit_long_mask = exit_long & ~prev_exit_long
        exit_short_mask = exit_short & ~prev_exit_short
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
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