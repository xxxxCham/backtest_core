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
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'rsi', 'adx', 'supertrend']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_d = indicators['adx']
        adx = np.nan_to_num(adx_d["adx"])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        # EMA crossovers
        ema_20 = ema
        ema_50 = np.roll(ema, 50)
        ema_50[0:50] = np.nan
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_50[0] = np.nan
        ema_cross_up = (ema_50 > ema_20) & (prev_ema_50 <= ema_20)
        ema_cross_down = (ema_50 < ema_20) & (prev_ema_50 >= ema_20)
        # Close crossing bands
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        close_cross_up_bb_lower = (close > indicators['bollinger']['lower']) & (prev_close <= indicators['bollinger']['lower'])
        close_cross_down_bb_upper = (close < indicators['bollinger']['upper']) & (prev_close >= indicators['bollinger']['upper'])
        # Volume confirmation
        volume_long_cond = volume_oscillator > 0
        volume_short_cond = volume_oscillator < 0
        # ADX filter for inverse mode
        adx_filter = adx < 20
        # Entry long condition
        long_entry = ema_cross_up & close_cross_up_bb_lower & volume_long_cond & adx_filter
        long_mask[long_entry] = True
        # Entry short condition
        short_entry = ema_cross_down & close_cross_down_bb_upper & volume_short_cond & adx_filter
        short_mask[short_entry] = True
        # Exit conditions
        rsi_overbought = rsi > params["rsi_overbought"]
        exit_long = rsi_overbought | close_cross_down_bb_upper
        exit_short = rsi_overbought | close_cross_up_bb_lower
        # Apply exit signals
        signals[long_mask & exit_long] = 0.0
        signals[short_mask & exit_short] = 0.0
        # Set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Long entries
        entry_long_mask = (signals == 1.0)
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        # Short entries
        entry_short_mask = (signals == -1.0)
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
