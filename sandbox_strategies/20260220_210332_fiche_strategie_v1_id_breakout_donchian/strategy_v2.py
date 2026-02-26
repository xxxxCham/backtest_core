from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

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
        def generate_signals(df):
            # Initialize the strategy
            cerebro = bt.Strategy("FICHE_STRATEIE", roll_forward=True)

            cerebro.set_cash(100000)  # set initial cash
            cerebro.broker.set_cash(100000)  # set broker's cash

            rsi_period = 14
            rsi_input = 'INDX_RSI'

            # Add the RSI as an indicator to talib
            talib.RSI(df, period=rsi_period, input=rsi_input)

            cerebro.set_rules(bt.RuleNewInfoHigh(ttype='cci', op='gt', value=-50, mask=['indx'], input=rsi_input)) # Long signal if CCI is > -50 and indx (RSI in this case) is high

            cerebro.addsizer(bt.SizerFutures, ultaint=1e4, initpos=100)  # use a multiplier of 1E-4 for position size

            return cerebro.run()
        signals.iloc[:warmup] = 0.0
        return signals
