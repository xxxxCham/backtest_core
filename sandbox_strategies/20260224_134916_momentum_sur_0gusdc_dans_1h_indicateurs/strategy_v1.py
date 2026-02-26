from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 5.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
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
                default=2,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=5.0,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params:Dict[str,Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            long_mask = np.zeros(n,dtype=bool)
            short_mask = np.zeros(n,dtype=bool)

            # implement explicit LONG / SHORT / FLAT logic
            warmup = int(params.get("warmup", 50))

            signals.iloc[:warmup] = 0.0

            if "bb_stop_long" in df and "bb_tp_long" in df:
                atr=indicators['atr']
                indicators['bollinger']['upper']=indicators['bollinger']["upper"]
                indicators['bollinger']['lower']=indicators['bollinger']["lower"]

                bb_stop_long = df['bb_stop_long'].values  # assuming the column name is "bb_stop_long" and it contains values
                bb_tp_long = df['bb_tp_long'].values    # similarly for bb_tp_long

                long_mask[np.where((df["close"] > (indicators['bollinger']['upper'][0] * atr) + 5))] = True
                short_mask[np.where(df["close"] < df['bb_stop_long'][0])] = True   # change the condition to your needs

            signals[(signals == 1).astype(bool)].dropna(inplace=True)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
