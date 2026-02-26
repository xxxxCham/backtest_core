from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_style_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
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
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])
        close = df["close"].values
        # Precompute previous volume oscillator values
        vol_prev = np.roll(volume_osc, 1)
        vol_prev[0] = np.nan
        vol_prev2 = np.roll(volume_osc, 2)
        vol_prev2[0] = vol_prev[1] = np.nan
        vol_prev3 = np.roll(volume_osc, 3)
        vol_prev3[0] = vol_prev[1] = vol_prev[2] = np.nan
        # EMA crossover conditions
        ema_prev = np.roll(ema, 1)
        ema_prev[0] = np.nan
        ema_cross_up = (ema > ema_prev) & (np.roll(ema, 1) <= ema_prev)
        ema_cross_down = (ema < ema_prev) & (np.roll(ema, 1) >= ema_prev)
        # Close inside Bollinger band
        close_inside_bb = (close > indicators['bollinger']['lower']) & (close < indicators['bollinger']['upper'])
        # Volume oscillator confirmation
        vol_increasing = (volume_osc > vol_prev) & (vol_prev > vol_prev2) & (vol_prev2 > vol_prev3)
        # Long entry
        long_entry = ema_cross_up & close_inside_bb & vol_increasing
        long_mask = long_entry
        # Short entry
        short_entry = ema_cross_down & close_inside_bb & vol_increasing
        short_mask = short_entry
        # Exit conditions
        # Exit long if price crosses below upper band or RSI > 80 with decreasing volume
        exit_long = (close < indicators['bollinger']['upper']) | ((rsi > 80) & (volume_osc < vol_prev))
        # Exit short if price crosses above lower band or RSI > 80 with decreasing volume
        exit_short = (close > indicators['bollinger']['lower']) | ((rsi > 80) & (volume_osc < vol_prev))
        # Apply long and short signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set ATR-based stop-loss and take-profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0
        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals