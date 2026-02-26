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
        return ['aroon', 'ema', 'atr', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'entry_fee': 0.001,
         'exit_fee': 0.001,
         'leverage': 2,
         'slippage': 0.001,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'entry_fee': ParameterSpec(
                name='entry_fee',
                min_val=0,
                max_val=0.01,
                default=0.001,
                param_type='float',
                step=0.1,
            ),
            'exit_fee': ParameterSpec(
                name='exit_fee',
                min_val=0,
                max_val=0.01,
                default=0.001,
                param_type='float',
                step=0.1,
            ),
            'slippage': ParameterSpec(
                name='slippage',
                min_val=0,
                max_val=0.01,
                default=0.001,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=5,
                max_val=50,
                default=50,
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
        import numpy as np
        from typing import Dict, Any
        import pandas as pd
        from strategies.base import StrategyBase

        class PhaseLockStrategy(StrategyBase):
            def __init__(self):
                super().__init__(name="Phase Lock")

            @property
            def required_indicators(self) -> List[str]:
                return ["AROON", "EMA","MACD"]

            @property
            def default_params(self) -> Dict[str, Any]:
                return {"leverage": 1} # assuming leverage = 1 for simplicity

            @property
            def parameter_specs(self) -> Dict[str, ParameterSpec]:
                pass # implement later

            def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:

                n = len(df)
                signals = np.zeros(n, dtype=np.float64) # Initialize signal series with zeros
                long_mask = np.zeros(n, dtype=bool) 
                short_mask = np.zeros(n, dtype=bool)  

                # compute atr and ATR based stop loss levels here if using a trailing stop strategy

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
