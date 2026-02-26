from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_5m_ema_bollinger_atr_scalp_v2')

    @property
    def required_indicators(self) -> List[str]:
        # ema is a dict with 'fast' and 'slow' sub‑keys, bollinger is a dict, atr is a plain array
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_threshold': 0.001,
            'bollinger_period': 20,
            'bollinger_std_dev': 2,
            'ema_fast_period': 9,
            'ema_slow_period': 21,
            'leverage': 1,
            'stop_atr_mult': 2.3,
            'tp_atr_mult': 4.6,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast_period': ParameterSpec(
                name='ema_fast_period',
                min_val=5,
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'ema_slow_period': ParameterSpec(
                name='ema_slow_period',
                min_val=15,
                max_val=35,
                default=21,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.5,
                max_val=3.0,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0005,
                max_val=0.005,
                default=0.001,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.6,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
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
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        """
        Generate long (+1) / short (-1) signals based on:
        - EMA fast > EMA slow (long) / EMA fast < EMA slow (short)
        - Price above Bollinger upper (long) / below Bollinger lower (short)
        - ATR above a configurable threshold
        """
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warm‑up protection
        warmup = int(params.get('warmup', 50))

        # ------------------------------------------------------------------
        # Extract price series
        close = df['close'].values

        # ------------------------------------------------------------------
        # Extract and clean indicators
        # EMA is a dict with 'fast' and 'slow' keys
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])

        # Bollinger bands are provided as a dict
        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb['upper'])
        bb_middle = np.nan_to_num(bb['middle'])
        bb_lower = np.nan_to_num(bb['lower'])

        # ATR is a plain array
        atr = np.nan_to_num(indicators['atr'])

        # ------------------------------------------------------------------
        # Parameters
        atr_thresh = float(params.get('atr_threshold', 0.001))
        stop_atr_mult = float(params.get('stop_atr_mult', 2.3))
        tp_atr_mult = float(params.get('tp_atr_mult', 4.6))

        # ------------------------------------------------------------------
        # Helper arrays for cross detection (previous bar values)
        prev_fast = np.roll(ema_fast, 1)
        prev_slow = np.roll(ema_slow, 1)
        prev_fast[0] = np.nan
        prev_slow[0] = np.nan

        # ------------------------------------------------------------------
        # Entry conditions
        entry_long = (ema_fast > ema_slow) & (close > bb_upper) & (atr > atr_thresh)
        entry_short = (ema_fast < ema_slow) & (close < bb_lower) & (atr > atr_thresh)

        # Apply masks
        long_mask = entry_long
        short_mask = entry_short

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ------------------------------------------------------------------
        # Risk management: ATR‑based stop‑loss / take‑profit columns
        # (stored back into df for downstream use)
        df['bb_stop_long'] = np.nan
        df['bb_tp_long'] = np.nan
        df['bb_stop_short'] = np.nan
        df['bb_tp_short'] = np.nan

        df.loc[long_mask, 'bb_stop_long'] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, 'bb_tp_long'] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, 'bb_stop_short'] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, 'bb_tp_short'] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # ------------------------------------------------------------------
        # Ensure no signals during warm‑up period
        signals.iloc[:warmup] = 0.0

        return signals