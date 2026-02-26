from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='atr_adx_ema_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 25,
         'atr_threshold': 1.5,
         'leverage': 1,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 3.0,
         'tp_range_atr_mult': 1.6,
         'tp_trend_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_trend_atr_mult': ParameterSpec(
                name='tp_trend_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_range_atr_mult': ParameterSpec(
                name='tp_range_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.6,
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

        close = np.nan_to_num(df["close"].values)
        ema = np.nan_to_num(indicators['ema'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        adx_thr = params.get("adx_threshold", 25)
        atr_thr = params.get("atr_threshold", 1.5)
        stop_mult = params.get("stop_atr_mult", 2.5)
        tp_trend_mult = params.get("tp_trend_atr_mult", 3.0)
        tp_range_mult = params.get("tp_range_atr_mult", 1.6)

        long_cond = (close > ema) & (adx > adx_thr) & (atr > atr_thr)
        short_cond = (close < ema) & (adx > adx_thr) & (atr > atr_thr)

        long_mask[long_cond] = True
        short_mask[short_cond] = True

        exit_long_cond = (close < ema) | (adx < 20)
        exit_short_cond = (close > ema) | (adx < 20)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_cond] = 0.0
        signals[exit_short_cond] = 0.0

        signals.iloc[:warmup] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = long_mask
        entry_short = short_mask

        tp_long = close + np.where(adx > adx_thr, tp_trend_mult, tp_range_mult) * atr
        tp_short = close - np.where(adx > adx_thr, tp_trend_mult, tp_range_mult) * atr

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = tp_long[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = tp_short[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
