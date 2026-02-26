from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aroon_keltner_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'keltner', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'aroon_period': 20,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 1.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
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
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=2.0,
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

        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        indicators['keltner']['upper'] = np.nan_to_num(indicators['keltner']["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])
        adx = np.nan_to_num(indicators['adx']["adx"])
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # Long condition
        long_condition = (indicators['aroon']['aroon_up'] < 30) & (close < indicators['keltner']['lower']) & (adx > 25)
        long_mask = long_condition

        # Short condition
        short_condition = (indicators['aroon']['aroon_down'] > 70) & (close > indicators['keltner']['upper']) & (adx > 25)
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based stop loss and take profit
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan

        entry_mask_long = (signals == 1.0)
        df.loc[entry_mask_long, "bb_stop_long"] = close[entry_mask_long] - stop_atr_mult * atr[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = close[entry_mask_long] + tp_atr_mult * atr[entry_mask_long]

        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_mask_short = (signals == -1.0)

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
