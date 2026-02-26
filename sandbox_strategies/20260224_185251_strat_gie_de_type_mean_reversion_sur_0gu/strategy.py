from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_inverse_mode')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'supertrend', 'atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'supertrend_period': 10,
         'tp_atr_mult': 2.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=2.0,
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

        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])

        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        indicators['vortex']['signal'] = np.nan_to_num(indicators['vortex']["signal"])

        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])

        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Define crossover helpers
        prev_vi_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vi_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_vortex_signal = np.roll(indicators['vortex']['signal'], 1)
        prev_vi_plus[0] = np.nan
        prev_vi_minus[0] = np.nan
        prev_vortex_signal[0] = np.nan

        cross_up_vortex = (indicators['vortex']['vi_plus'] > indicators['vortex']['signal']) & (prev_vi_plus <= prev_vortex_signal)
        cross_down_vortex = (indicators['vortex']['vi_plus'] < indicators['vortex']['signal']) & (prev_vi_plus >= prev_vortex_signal)

        # Touch band condition
        close_touch_upper = np.abs(close - upper_bb) < 1e-8
        close_touch_lower = np.abs(close - lower_bb) < 1e-8

        # Supertrend filter
        trend_up = direction == 1.0
        trend_down = direction == -1.0

        # Long entry: touch upper band, vortex crosses up, trend up
        long_entry = close_touch_upper & cross_up_vortex & trend_up

        # Short entry: touch lower band, vortex crosses down, trend down
        short_entry = close_touch_lower & cross_down_vortex & trend_down

        long_mask = long_entry
        short_mask = short_entry

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals