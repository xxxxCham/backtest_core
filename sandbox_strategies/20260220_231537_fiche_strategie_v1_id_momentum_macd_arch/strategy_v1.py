from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
                n = len(df)

                # Initialize signals to zeros
                signals = np.zeros(n, dtype=np.float64)

                # Implement long/short logic based on indicators and parameters
                if "long_threshold" in params and "short_threshold" in params:
                    long_threshold, short_threshold = params["long_threshold"], params["short_threshold"]

                    close = df['close'].values
                    ema = pd.Series(indicators['ema'], index=df.index)
                    rsi = pd.Series(pd.stats.binning.bins_scalar(close, window=14), index=df.index)

                    long_mask = (rsi > close + (rsi - 25)) & (close < short_threshold)
                    signals[long_mask] = 2 # set signals to 2 when in long position based on your logic

                else:  # implement placeholder for missing parameters or throw exception if required params not found
                    raise Exception("Missing parameters 'long_threshold' and/or 'short_threshold'")

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
