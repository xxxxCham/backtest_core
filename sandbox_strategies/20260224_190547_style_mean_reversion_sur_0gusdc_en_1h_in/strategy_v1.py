from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_stochastic_williamsr_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'williams_r', 'volume_oscillator', 'donchian', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'donchian_period': 20,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=2,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_short': ParameterSpec(
                name='volume_oscillator_short',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_long': ParameterSpec(
                name='volume_oscillator_long',
                min_val=15,
                max_val=40,
                default=26,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        wr = np.nan_to_num(indicators['williams_r'])
        vo = np.nan_to_num(indicators['volume_oscillator'])
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        lower = np.nan_to_num(dc["lower"])
        bb = indicators['bollinger']
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        # compute divergence
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        prev_wr = np.roll(wr, 1)
        prev_wr[0] = np.nan
        prev_vo = np.roll(vo, 1)
        prev_vo[0] = np.nan
        # bearish divergence on volume oscillator
        bearish_vo_div = (vo < prev_vo) & (k > prev_k)
        # bullish divergence on volume oscillator
        bullish_vo_div = (vo > prev_vo) & (k < prev_k)
        # stochastic crosses below 20
        prev_k_20 = np.roll(k, 1)
        prev_k_20[0] = np.nan
        cross_below_20 = (k < 20) & (prev_k_20 >= 20)
        # stochastic crosses above 80
        prev_k_80 = np.roll(k, 1)
        prev_k_80[0] = np.nan
        cross_above_80 = (k > 80) & (prev_k_80 <= 80)
        # entry conditions
        long_mask = (cross_below_20 & bearish_vo_div & (close > upper))
        short_mask = (cross_above_80 & bullish_vo_div & (close < lower))
        # exit conditions
        exit_long = cross_above_80
        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # write SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # on long entry
        entry_long = (signals == 1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # on short entry
        entry_short = (signals == -1.0)
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
