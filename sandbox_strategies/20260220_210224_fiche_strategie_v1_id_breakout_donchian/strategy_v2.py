from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=8.0,
                default=2.5,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self, df):
            # Assuming 'df' is a DataFrame with 'close' column
            n = len(df['close'])

            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # Implement explicit LONG / SHORT / FLAT logic here
            # For simplicity, we'll just use a simple RSI strategy: 
            # long when rsi > 70 and adx > 25, short otherwise

            close = df['close'].values

            rsi_value, _ = self.calculate_rsi(close)

            if rsi_value > 70:
                long_mask[:] = True

            elif rsi_value < 30:
                short_mask[:] = True

            else:
                long_mask[:] = False
                short_mask[:] = False

            # Calculate and apply stop loss levels based on adx and atr parameters
            if 'adx' in self.params and 'atr' in self.params:
               adx, _, _ = self.calculate_adx(df)
               upper_tp = df['close'] + self.params['tp_percent'] * df['high'].rolling(window=self.params['adx_period'], min_periods=1).mean() - \
                          df['open'] + self.params['stop_percent'] * df['low'].rolling(window=self.params['adx_period'], min_periods=1).mean()  # target levels for long and short positions
               lower_tp = df['close'] - self.params['tp_percent'] * df['high'].rolling(window=self.params['adx_period'], min_periods=1).mean() + \
                          df['open'] - self.params['stop_percent'] * df['low'].rolling(window=self.params['adx_period'], min_periods=1).mean()  # target levels for short positions
               long_mask[np.where((df['close'] > upper_tp) & (adx > self.params['min_adx']))] = True
               short_mask[np.where((df['close'] < lower_tp) & (adx >= self.params['min_adx']))] = True

            # Return the generated signals as a pandas Series
            return pd.Series(long_mask, index=range(n))
        signals.iloc[:warmup] = 0.0
        return signals
