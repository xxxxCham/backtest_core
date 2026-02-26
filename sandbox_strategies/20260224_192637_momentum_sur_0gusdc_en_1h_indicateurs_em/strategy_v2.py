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
         'bollinger_std_dev': 2,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # Compute previous values for crossovers
        prev_ema = np.roll(ema, 1)
        prev_bb_middle = np.roll(indicators['bollinger']['middle'], 1)
        prev_vol_osc = np.roll(vol_osc, 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_close = np.roll(close, 1)

        # Initialize previous values
        prev_ema[0] = 0.0
        prev_bb_middle[0] = 0.0
        prev_vol_osc[0] = 0.0
        prev_st_direction[0] = 0.0
        prev_close[0] = 0.0

        # Entry conditions
        # Long entry: EMA crosses above BB middle band, volume oscillator positive, supertrend confirms uptrend
        long_entry_cross_up = (ema > indicators['bollinger']['middle']) & (prev_ema <= prev_bb_middle)
        long_entry_vol_pos = vol_osc > 0
        long_entry_trend_conf = st_direction < 1  # Supertrend direction is up (less than 1)

        long_mask = long_entry_cross_up & long_entry_vol_pos & long_entry_trend_conf

        # Short entry: EMA crosses below BB middle band, volume oscillator negative, supertrend confirms downtrend
        short_entry_cross_down = (ema < indicators['bollinger']['middle']) & (prev_ema >= prev_bb_middle)
        short_entry_vol_neg = vol_osc < 0
        short_entry_trend_conf = st_direction > 1  # Supertrend direction is down (greater than 1)

        short_mask = short_entry_cross_down & short_entry_vol_neg & short_entry_trend_conf

        # Exit conditions
        # Exit long: EMA crosses below BB middle band OR spread drops below 20%
        exit_long_ema_cross = (ema < indicators['bollinger']['middle']) & (prev_ema >= prev_bb_middle)
        spread_ratio = (indicators['bollinger']['upper'] - indicators['bollinger']['lower']) / indicators['bollinger']['middle']
        exit_long_spread = spread_ratio < 0.2
        exit_long = exit_long_ema_cross | exit_long_spread

        # Exit short: EMA crosses above BB middle band OR spread drops below 20%
        exit_short_ema_cross = (ema > indicators['bollinger']['middle']) & (prev_ema <= prev_bb_middle)
        exit_short_spread = spread_ratio < 0.2
        exit_short = exit_short_ema_cross | exit_short_spread

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set SL/TP levels for long entries
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        # Set SL/TP levels for short entries
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[:, "bb_stop_short"] = np.nan
            df.loc[:, "bb_tp_short"] = np.nan
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals