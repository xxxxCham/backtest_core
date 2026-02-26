from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['amplitude_hunter', 'donchian', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'slippage': 0.002,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Implement ATR-based stop loss and take profit
                atr = np.nan_to_num(indicators['atr'])
                sl_level = df['close'] - params.get("leverage") * atr[0]  # use the first bar close price as reference for SL level
                tp_level = df['close'] + params.get("leverage") * atr[0]  

                long_mask = (signals == 1.0)    # boolean mask of being in a long position
                short_mask = (signals == -1.0)  # boolean mask of being in a short position

                signals[:params['warmup']] = 0.0   # placeholders for the first few bars, replace with actual data later

                # Implement LONG entry logic here
                if long_mask[0]:    # skip the placeholder values at the beginning of the series
                    warmup = params["warmup"]
                    signals[long_mask] = 1.0   # set all previous entries to be in a long position

                # Implement SHORT exit logic here
                if short_mask[-1]:    # skip the placeholder values at the end of the series
                    signals[~short_mask].astype(float)  # convert back from int64
                    signals[:params['warmup']] = 0.0   # placeholders for the first few bars of data, replace with actual data later

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
