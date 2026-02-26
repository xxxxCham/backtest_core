from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_vortex_volume_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'vortex', 'volume_oscillator', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_sma_period': 5,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
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
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])

        vx = indicators['vortex']
        vx_plus = np.nan_to_num(vx["vi_plus"])
        vx_minus = np.nan_to_num(vx["vi_minus"])
        vx_osc = np.nan_to_num(vx["oscillator"])

        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        vol_sma = np.nan_to_num(pd.Series(vol_osc).rolling(params["volume_sma_period"]).mean().values)

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # Prepare previous values for crossovers
        prev_kelt_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_kelt_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_vx_plus = np.roll(vx_plus, 1)
        prev_vx_minus = np.roll(vx_minus, 1)
        prev_vol_osc = np.roll(vol_osc, 1)
        prev_kelt_middle = np.roll(indicators['keltner']['middle'], 1)

        # Set first values to NaN for proper crossover detection
        prev_kelt_upper[0] = np.nan
        prev_kelt_lower[0] = np.nan
        prev_vx_plus[0] = np.nan
        prev_vx_minus[0] = np.nan
        prev_vol_osc[0] = np.nan
        prev_kelt_middle[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above keltner upper, vortex increasing, volume above SMA
        long_cross_up = (close > indicators['keltner']['upper']) & (prev_kelt_upper <= indicators['keltner']['upper'])
        vx_increasing = (vx_plus > prev_vx_plus) & (vx_minus < prev_vx_minus)
        vol_above_sma = (vol_osc > vol_sma)

        long_mask = long_cross_up & vx_increasing & vol_above_sma

        # Short entry: close crosses below keltner lower, vortex decreasing, volume below SMA
        short_cross_down = (close < indicators['keltner']['lower']) & (prev_kelt_lower >= indicators['keltner']['lower'])
        vx_decreasing = (vx_plus < prev_vx_plus) & (vx_minus > prev_vx_minus)
        vol_below_sma = (vol_osc < vol_sma)

        short_mask = short_cross_down & vx_decreasing & vol_below_sma

        # Exit conditions
        # Exit long if close crosses below middle band or RSI > 70
        rsi = np.nan_to_num(indicators['rsi'])
        exit_long = (close < indicators['keltner']['middle']) | (rsi > 70)
        exit_long_mask = np.zeros(n, dtype=bool)
        exit_long_mask[1:] = np.diff(exit_long) > 0
        exit_long_mask[0] = False

        # Exit short if close crosses above middle band or RSI > 70
        exit_short = (close > indicators['keltner']['middle']) | (rsi > 70)
        exit_short_mask = np.zeros(n, dtype=bool)
        exit_short_mask[1:] = np.diff(exit_short) > 0
        exit_short_mask[0] = False

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals