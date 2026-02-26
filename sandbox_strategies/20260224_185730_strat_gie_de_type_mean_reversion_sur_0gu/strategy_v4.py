from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'volume_oscillator', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volatility_gating_period': 20,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'volatility_gating_period': ParameterSpec(
                name='volatility_gating_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=2.0,
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
        bb = indicators['bollinger']
        close = df["close"].values
        ema_50 = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])

        # Compute moving average of volume oscillator
        vol_ma = np.nan_to_num(pd.Series(volume_oscillator).rolling(10).mean().values)

        # Boolean masks for touch bollinger bands
        lower_band = np.nan_to_num(bb["lower"])
        upper_band = np.nan_to_num(bb["upper"])
        touch_lower = (close == lower_band)
        touch_upper = (close == upper_band)

        # Volume oscillator crossing
        prev_vol = np.roll(volume_oscillator, 1)
        prev_vol[0] = np.nan
        vol_cross_up = (volume_oscillator > vol_ma) & (prev_vol <= vol_ma)
        vol_cross_down = (volume_oscillator < vol_ma) & (prev_vol >= vol_ma)

        # EMA 50 trend confirmation
        ema_diff = np.insert(np.diff(ema_50), 0, 0.0)
        ema_rising = (ema_diff > 0)
        ema_falling = (ema_diff < 0)

        # Long entry conditions
        long_entry = touch_lower & vol_cross_up & ema_rising

        # Short entry conditions
        short_entry = touch_upper & vol_cross_down & ema_falling

        # Set long and short masks
        long_mask = long_entry
        short_mask = short_entry

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)

        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals