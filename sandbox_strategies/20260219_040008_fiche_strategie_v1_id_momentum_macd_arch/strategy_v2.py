from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE_v2')
    
    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'slippage': 0.5, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR stop-loss mult': ParameterSpec(
                name='ATR stop-loss mult',
                min_val=2.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'ATR take profit mult': ParameterSpec(
                name='ATR take profit mult',
                min_val=3.0,
                max_val=6.0,
                default=3.5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # Warmup protection
        signals.iloc[:self.default_params['warmup']] = 0.0
    
        atr = np.nan_to_num(indicators["atr"])
        close = df["close"].values
        sl_level = params.get("sl_level", 1) * atr - self.default_params["leverage"] * atr[0]
        tp_level = params.get("tp_level", 1) * atr + self.default_params["leverage"] * atr[-1]
    
        # Implement long/short logic here
        pass