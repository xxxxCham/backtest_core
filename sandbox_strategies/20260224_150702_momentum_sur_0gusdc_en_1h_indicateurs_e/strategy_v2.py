from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'stochastic']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'obv_threshold': 0.2,
         'sl_multiplier': 1.2,
         'stoch_k_cross_d_multiplier': 1.5,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_threshold': ParameterSpec(
                name='obv_threshold',
                min_val=0.01,
                max_val=0.8,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'stoch_k_cross_d_multiplier': ParameterSpec(
                name='stoch_k_cross_d_multiplier',
                min_val=1,
                max_val=3,
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
        if "bb_stop_long" in df and "bb_tp_long" in df:
            atr = np.nan_to_num(indicators['atr'])[params["warmup"]:]

            long_mask = (signals == 1) & (df['bb_stop_long'].notna()) & (df['bb_tp_long'].notna())
            sl_level = df.loc[long_mask, 'close'][params["warmup"]] - params["leverage"] * atr[:-params["warmup"]].mean()

            short_mask = (signals == -1) & (df['bb_stop_short'].notna()) & (df['bb_tp_short'].notna())
            tp_level = df.loc[short_mask, 'close'][params["warmup"]] + params["leverage"] * atr[:-params["warmup"]].mean()

            signals[(df['close'] >= sl_level)] = -1.0
            signals[(df['close'] <= tp_level)] = 1.0
        signals.iloc[:warmup] = 0.0
        return signals
