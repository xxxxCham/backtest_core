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
        # Generate ONLY the body lines to insert inside generate_signals. Do NOT generate class/imports/function signature, or indicator values.
        def generate_signals(df, default_params={'leverage': 1}, long_intent='LONG', short_intent='SHORT'):
            indicators = {'rsi': df['close'].tolist(), 'ema': df['close'].tolist(), 'atr': df['high'] + df['low'] + df['close']} # TODO: add other indicators as needed

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)  # Always include leverage=1 in default_params

            for i, row in df.iterrows():
                rsi_val = indicators['rsi'][i] if np.isnan(indicators[f'{long_intent}_intent']['value']) else long_intent

                # TODO: Write the rest of generate_signals() function here with suitable conditions for LONG/SHORT intent, and appropriate actions based on available indicators
        signals.iloc[:warmup] = 0.0
        return signals
