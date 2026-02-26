from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='style_momentum_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=100,
                default=50,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])

        # Volume oscillator lagged values
        vol_osc_lag1 = np.roll(vol_osc, 1)
        vol_osc_lag2 = np.roll(vol_osc, 2)
        vol_osc_lag3 = np.roll(vol_osc, 3)
        vol_osc_lag1[0] = np.nan
        vol_osc_lag2[0] = vol_osc_lag2[1] = np.nan
        vol_osc_lag3[0] = vol_osc_lag3[1] = vol_osc_lag3[2] = np.nan

        # EMA crossover conditions
        ema_50 = ema
        close_array = close

        # Previous close and EMA for crossovers
        prev_close = np.roll(close_array, 1)
        prev_ema = np.roll(ema_50, 1)
        prev_close[0] = np.nan
        prev_ema[0] = np.nan

        # Long crossover
        ema_cross_up = (close_array > ema_50) & (prev_close <= prev_ema)
        # Short crossover
        ema_cross_down = (close_array < ema_50) & (prev_close >= prev_ema)

        # Inside band condition
        inside_band_long = (close_array > indicators['bollinger']['lower']) & (close_array < indicators['bollinger']['upper'])
        inside_band_short = (close_array < indicators['bollinger']['upper']) & (close_array > indicators['bollinger']['lower'])

        # Volume oscillator confirmation
        vol_up_3 = (vol_osc > vol_osc_lag3) & (vol_osc > vol_osc_lag2) & (vol_osc > vol_osc_lag1)

        # Long entry conditions
        long_entry = ema_cross_up & inside_band_long & vol_up_3

        # Short entry conditions
        short_entry = ema_cross_down & inside_band_short & vol_up_3

        # Exit conditions
        # Exit on upper band cross down
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        exit_up = (close_array < indicators['bollinger']['upper']) & (prev_bb_upper >= indicators['bollinger']['upper'])

        # Exit on RSI > 80 and volume oscillator decreasing
        # Remove RSI usage since it's not in required_indicators
        # Use only the available indicators
        exit_condition = exit_up

        # Apply long and short signals
        long_mask = long_entry
        short_mask = short_entry

        # Mark exits
        exit_long_mask = np.zeros(n, dtype=bool)
        exit_short_mask = np.zeros(n, dtype=bool)

        # For simplicity, apply exit condition on the next bar after entry
        exit_long_mask[1:] = exit_condition[:-1]
        exit_short_mask[1:] = exit_condition[:-1]

        # Ensure no double entry
        long_mask = long_mask & ~exit_long_mask
        short_mask = short_mask & ~exit_short_mask

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management: ATR-based SL/TP
        atr = np.nan_to_num(indicators['atr'])
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entries
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        # Short entries
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals