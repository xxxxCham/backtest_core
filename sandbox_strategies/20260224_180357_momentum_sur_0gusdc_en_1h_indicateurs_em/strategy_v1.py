from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=100,
                max_val=300,
                default=200,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        signals.iloc[:warmup] = 0.0
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        ema_50 = ema_fast
        ema_200 = ema_slow
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_200 = np.roll(ema_200, 1)
        prev_ema_50[0] = np.nan
        prev_ema_200[0] = np.nan
        cross_up_50_200 = (ema_50 > ema_200) & (prev_ema_50 <= prev_ema_200)
        cross_down_50_200 = (ema_50 < ema_200) & (prev_ema_50 >= prev_ema_200)
        bb_lower_prev = np.roll(indicators['bollinger']['lower'], 1)
        bb_lower_prev[0] = np.nan
        bb_contracting = indicators['bollinger']['lower'] < bb_lower_prev
        bb_deviation = np.abs((close - indicators['bollinger']['middle']) / indicators['bollinger']['middle'])
        vol_osc_mean = np.nan_to_num(pd.Series(vol_osc).rolling(10).mean().values)
        vol_osc_long_threshold = params.get("volume_oscillator_long", 26)
        vol_osc_short_threshold = params.get("volume_oscillator_short", 12)
        vol_osc_long_cond = (vol_osc > 0) & (vol_osc > vol_osc_mean)
        vol_osc_short_cond = (vol_osc < 0) & (vol_osc < vol_osc_mean)
        long_condition = cross_up_50_200 & bb_contracting & vol_osc_long_cond
        short_condition = cross_down_50_200 & bb_contracting & vol_osc_short_cond
        long_mask = long_condition
        short_mask = short_condition
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Exit condition
        prev_vol_osc = np.roll(vol_osc, 1)
        prev_vol_osc[0] = np.nan
        prev_vol_osc_2 = np.roll(vol_osc, 2)
        prev_vol_osc_2[0] = np.nan
        prev_vol_osc_2[1] = np.nan
        vol_osc_consecutive = (vol_osc < 0) & (prev_vol_osc < 0) & (prev_vol_osc_2 < 0)
        exit_condition = vol_osc_consecutive & (bb_deviation > 2.0)
        exit_long_mask = long_mask & exit_condition
        exit_short_mask = short_mask & exit_condition
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
