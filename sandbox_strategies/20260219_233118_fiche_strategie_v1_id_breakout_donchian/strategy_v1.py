from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.75, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.5,
                max_val=4.0,
                default=2.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
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
        def generate_signals(self, df, indicators, params):
                signals = pd.Series(0.0, index=df.index)
                n = len(df)

                # Skip warmup bars
                warmup = int(params.get("warmup", 50))
                signals.iloc[:warmup] = 0.0

                close = df["close"].values
                atr = np.nan_to_num(indicators['atr'])

                donchian = indicators['donchian']
                adx = indicators['adx']['adx']  # use only ADX value, not plus/minus DI
                stop_atr_mult = params.get("stop_atr_mult", 2.75)
                tp_atr_mult = params.get("tp_atr_mult", 3.0)

                # Long entry condition
                long_mask = (close > indicators['donchian']["upper"]) & (adx > 35)
                df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
                df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

                # Short entry condition (if applicable)
                # short_mask = ...

                # Exit conditions
                prev_close = np.roll(close, 1)
                prev_close[0] = np.nan
                exit_mask = ((prev_close < indicators['donchian']["middle"]) & (close >= indicators['donchian']["middle"])) | (adx < 15)

                # Long exits
                signals.loc[long_mask & exit_mask] = 0.0  # flatten long positions on signal bar
                signals.loc[(signals == 1.0) & exit_mask] = -1.0  # close any open long position

                # Short exits (if applicable)
                # signals.loc[short_mask & exit_mask] = 0.0  # flatten short positions on signal bar
                # signals.loc[(signals == -1.0) & exit_mask] = 1.0  # close any open short position

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
