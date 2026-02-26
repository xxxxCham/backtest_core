from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'roc']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATR_period': 14,
         'EMA_period': 5,
         'MACD_fast_length': 12,
         'MACD_slow_length': 26,
         'ROC_threshold': 20,
         'SLTR': 2.0,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'EMA_period': ParameterSpec(
                name='EMA_period',
                min_val=1,
                max_val=50,
                default=5,
                param_type='int',
                step=1,
            ),
            'MACD_fast_length': ParameterSpec(
                name='MACD_fast_length',
                min_val=10,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'MACD_slow_length': ParameterSpec(
                name='MACD_slow_length',
                min_val=20,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'ROC_threshold': ParameterSpec(
                name='ROC_threshold',
                min_val=20,
                max_val=80,
                default=20,
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
        def generate_signals(indicators):
            df = pd.DataFrame() # Assuming 'df' is the input DataFrame

            for i, row in df.iterrows():
                close = row[['close']]  # Assuming 'data' is a pandas Series/Dataframe with 'close' as the relevant column name

                ema_30 = indicators['ema'][i]['EMA 30']  # Calculate EMA using numpy array
                macd = indicators['macd'][i].macd()   # MACD calculation
                roc = indicators['roc'][i].roc()    # ROC calculation

                ema_crossed_long = np.where((ema_30 < close) & (close > ema_30), 1, 0).astype(np.float64)   # Long signal generation
                roc_above_threshold = np.where((roc - indicators['roc'][i].thresholds['signal']) >= indicators['roc'][i].thresholds[ 'threshold'], 1, 0).astype(np.float64)    # Short signal generation

                signals = ema_crossed_long & roc_above_threshold   # Combine the two signals to generate overall signal

                if len(signals):
                    signals = signals

            return df  # Assuming 'df' is DataFrame which we want to return as result.
        signals.iloc[:warmup] = 0.0
        return signals
