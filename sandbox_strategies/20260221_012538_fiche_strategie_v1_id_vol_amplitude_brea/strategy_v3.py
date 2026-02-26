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
        def generate_signals(df, default_params):
            # Always include leverage=1 in default_params

            for i in df.index:
                # Check each asset or time period

                bollinger = {key: np.roll(val[0], 1) if key != 'donchian' else val for key, val in default_params['bollinger'].items()}
                indicators['donchian']['upper'] = bollinger['donchian'][0] # Donchian Bands Upper Band

                supertrend = {key: val[2] if key not in ('direction', 'supertrend') else True for key, val in default_params['supertrend'].items()} 
                adx = {key: val[1] if key != 'adx' else True for key, val in default_params['adx'].items()} # ADX Direction only

                stochastic = {key: val[-1] if key != 'stochastic' else False for key, val in default_params['stochastic'].items()} 

                atr = bollinger['atr'][i] # ATR value

                # Check conditions

                if ((df.loc[i,'amplitude_hunter:score'] > 0.6) & (indicators['donchian']['lower'] < close) & indicators['supertrend']['direction']<0):
                    signals[i] = -1.0   # Long position with Stop Loss below previous band and Target Price above Bollinger Upper Band
                elif ((df.loc[i,'amplitude_hunter:score'] > 0.6) & (indicators['donchian']['lower'] < close) & indicators['supertrend']['direction']>0):
                    signals[i] = -1.0   # Short position with Stop Loss below previous band and Target Price above Bollinger Upper Band
                elif ((df.loc[i,'amplitude_hunter:score'] > 0.6) & (indicators['donchian']['lower'] >= close) & indicators['supertrend']['direction']<0):
                    signals[i] = -1.0   # Long position with Stop Loss below previous band and Target Price above Bollinger Upper Band
                elif ((df.loc[i,'amplitude_hunter:score'] > 0.6) & (indicators['donchian']['lower'] >= close)):
                    signals[i] = 1.0    # Short position with Stop Loss below previous band and Target Price above Bollinger Upper Band

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
