from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_vortex_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'vortex', 'rsi', 'ema', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_atr_period': 14,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'rsi_overbought': 80,
         'rsi_oversold': 20,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_multiplier': ParameterSpec(
                name='keltner_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
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
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        vortex = indicators['vortex']
        vortex_plus = np.nan_to_num(indicators['vortex']["vi_plus"])
        vortex_minus = np.nan_to_num(indicators['vortex']["vi_minus"])
        indicators['vortex']['signal'] = np.nan_to_num(indicators['vortex']["signal"])
        rsi = np.nan_to_num(indicators['rsi'])
        ema = np.nan_to_num(indicators['ema'])
        close = df["close"].values
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # Compute crossovers for Vortex
        prev_vortex_plus = np.roll(vortex_plus, 1)
        prev_vortex_minus = np.roll(vortex_minus, 1)
        prev_vortex_plus[0] = np.nan
        prev_vortex_minus[0] = np.nan
        cross_up_vortex = (vortex_plus > indicators['vortex']['signal']) & (prev_vortex_plus <= indicators['vortex']['signal'])
        cross_down_vortex = (vortex_plus < indicators['vortex']['signal']) & (prev_vortex_plus >= indicators['vortex']['signal'])
        # Compute crossovers for EMA
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        cross_up_ema = (close > ema) & (prev_ema <= ema)
        cross_down_ema = (close < ema) & (prev_ema >= ema)
        # Entry conditions
        # Long entry: close crosses above indicators['keltner']['upper'] AND vortex.vortex < vortex.indicators['vortex']['signal'] AND rsi > 50 AND ema.close > ema.ema AND volume_oscillator > 0
        long_entry_condition = (
            (close > indicators['keltner']['upper']) &
            (vortex_plus < indicators['vortex']['signal']) &
            (rsi > 50) &
            (close > ema) &
            (volume_osc > 0)
        )
        long_mask = long_entry_condition
        # Short entry: close crosses below indicators['keltner']['lower'] AND vortex.vortex > vortex.indicators['vortex']['signal'] AND rsi < 50 AND ema.close < ema.ema AND volume_oscillator > 0
        short_entry_condition = (
            (close < indicators['keltner']['lower']) &
            (vortex_plus > indicators['vortex']['signal']) &
            (rsi < 50) &
            (close < ema) &
            (volume_osc > 0)
        )
        short_mask = short_entry_condition
        # Exit conditions
        # Exit long: rsi > 80 OR rsi < 20 OR vortex.vortex crosses above vortex.indicators['vortex']['signal']
        exit_long_condition = (
            (rsi > params["rsi_overbought"]) |
            (rsi < params["rsi_oversold"]) |
            cross_up_vortex
        )
        # Exit short: rsi > 80 OR rsi < 20 OR vortex.vortex crosses below vortex.indicators['vortex']['signal']
        exit_short_condition = (
            (rsi > params["rsi_overbought"]) |
            (rsi < params["rsi_oversold"]) |
            cross_down_vortex
        )
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Write SL/TP columns into df if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
