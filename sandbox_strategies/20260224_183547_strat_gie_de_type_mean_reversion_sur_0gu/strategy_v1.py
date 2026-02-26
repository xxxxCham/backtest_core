from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_aroon_vortex_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'aroon', 'vortex', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
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
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
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
        signals.iloc[:warmup] = 0.0
        close = df["close"].values
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        aroon = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        vortex = indicators['vortex']
        vortex_osc = np.nan_to_num(indicators['vortex']["oscillator"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_kelt_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_kelt_lower[0] = np.nan
        prev_kelt_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_kelt_upper[0] = np.nan
        prev_kelt_middle = np.roll(indicators['keltner']['middle'], 1)
        prev_kelt_middle[0] = np.nan
        prev_aroon_down = np.roll(indicators['aroon']['aroon_down'], 1)
        prev_aroon_down[0] = np.nan
        prev_aroon_up = np.roll(indicators['aroon']['aroon_up'], 1)
        prev_aroon_up[0] = np.nan
        prev_vortex_osc = np.roll(vortex_osc, 1)
        prev_vortex_osc[0] = np.nan
        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan
        cross_below_keltner = (close < indicators['keltner']['lower']) & (prev_close >= prev_kelt_lower)
        cross_above_keltner = (close > indicators['keltner']['upper']) & (prev_close <= prev_kelt_upper)
        aroon_down_below_30 = indicators['aroon']['aroon_down'] < 30
        aroon_up_below_30 = indicators['aroon']['aroon_up'] < 30
        vortex_below_12 = vortex_osc < 1.2
        long_condition = cross_below_keltner & aroon_down_below_30 & vortex_below_12
        short_condition = cross_above_keltner & aroon_up_below_30 & vortex_below_12
        long_mask = long_condition
        short_mask = short_condition
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        exit_long = (close > indicators['keltner']['middle']) | (rsi > params["rsi_overbought"])
        exit_short = (close < indicators['keltner']['middle']) | (rsi > params["rsi_overbought"])
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
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
