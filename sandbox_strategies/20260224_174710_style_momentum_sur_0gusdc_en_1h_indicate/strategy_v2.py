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
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
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

        # Extract indicators
        ema_20 = np.nan_to_num(indicators['ema'])
        ema_50 = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        adx_d = indicators['adx']
        adx = np.nan_to_num(adx_d["adx"])

        # EMA crossovers
        ema_20 = ema_20[:n]
        ema_50 = ema_50[:n]
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_20 = np.roll(ema_20, 1)
        prev_ema_50[0] = np.nan
        prev_ema_20[0] = np.nan
        ema_cross_up = (ema_50 > ema_20) & (prev_ema_50 <= prev_ema_20)
        ema_cross_down = (ema_50 < ema_20) & (prev_ema_50 >= prev_ema_20)

        # Price crossing bands
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        price_cross_up = (close > indicators['bollinger']['lower']) & (prev_close <= indicators['bollinger']['lower'])
        price_cross_down = (close < indicators['bollinger']['upper']) & (prev_close >= indicators['bollinger']['upper'])

        # Volume oscillator
        volume_long = volume_osc > 0
        volume_short = volume_osc < 0

        # ADX filter
        adx_filter = adx > 25

        # Long entry conditions
        long_entry = ema_cross_up & price_cross_up & volume_long & adx_filter
        long_mask = long_entry

        # Short entry conditions
        short_entry = ema_cross_down & price_cross_down & volume_short & adx_filter
        short_mask = short_entry

        # Exit conditions
        exit_long = close > indicators['bollinger']['middle']
        exit_short = close < indicators['bollinger']['middle']

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit conditions
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

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
