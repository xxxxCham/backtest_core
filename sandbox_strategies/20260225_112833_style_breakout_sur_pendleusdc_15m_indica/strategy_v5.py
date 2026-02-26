from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='style_breakout_pendleusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
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

        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Precompute previous values for crossover logic
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        prev_obv = np.roll(obv, 1)
        prev_volume_oscillator = np.roll(volume_oscillator, 1)

        prev_close[0] = np.nan
        prev_ema[0] = np.nan
        prev_obv[0] = np.nan
        prev_volume_oscillator[0] = np.nan

        # Entry long: close crosses above ema with OBV and volume_oscillator confirmation
        close_above_ema = (close > ema)
        prev_close_below_ema = (prev_close <= prev_ema)
        obv_up = (obv > prev_obv)
        volume_oscillator_up = (volume_oscillator > prev_volume_oscillator)

        long_entry = close_above_ema & prev_close_below_ema & obv_up & volume_oscillator_up
        long_mask = long_entry

        # Entry short: close crosses below ema with OBV and volume_oscillator confirmation
        close_below_ema = (close < ema)
        prev_close_above_ema = (prev_close >= prev_ema)
        obv_down = (obv < prev_obv)
        volume_oscillator_down = (volume_oscillator < prev_volume_oscillator)

        short_entry = close_below_ema & prev_close_above_ema & obv_down & volume_oscillator_down
        short_mask = short_entry

        # Exit long: close crosses below ema OR obv diverges
        exit_long = (close < ema) | (obv < prev_obv)
        long_mask = long_mask & ~exit_long

        # Exit short: close crosses above ema OR obv diverges
        exit_short = (close > ema) | (obv > prev_obv)
        short_mask = short_mask & ~exit_short

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

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
