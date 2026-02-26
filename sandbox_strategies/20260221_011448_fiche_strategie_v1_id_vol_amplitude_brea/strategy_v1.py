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
        def generate_signals(self, df):
            n = len(df)

            # Set up default parameters if not already defined in the algo or defaults
            leverage = self.params.get('leverage', 1)

            # Set up indicator dataframes for RSI, EMA and ATR
            rsi_dataframe = pd.DataFrame()
            ema_dataframe = pd.DataFrame()
            atr_dataframe = pd.DataFrame()

            # ... fill in the rest of your code here to compute these indicators using df data...

            # Set up default signals dataframe if not already defined in the algo or defaults
            signals_df = pd.DataFrame(index=df.index, dtype=float)

            # Calculate a simple moving average for comparison with RSI values
            sma = df['close'].rolling(window=self.params.get('sma\_window', 30)).mean()

            # ... fill in the rest of your code here to compute the signals based on the indicators and other conditions...

            # Add warmup protection by setting signals for a number of bars before the start date
            if not self.is_live:
                signals = signals_df[:self.params['warmup']]
            else:
                signals = signals_df[self.start_date():self.frame.datetime.max()]

            # Apply ATR-based risk management by adjusting signals based on volatility
            for symbol in df['symbol'].unique():
                atr_dataframe[symbol] = self._get_atr(df, symbol)

                close_prices = df[(df['symbol'] == symbol) & (df.index >= self.start_date())]['close'].values
                ema_values = pd.Series([self._ema(data=price[np.newaxis], window=self.params.get('sma\_window', 30)) for price in close_prices])

                signals[(df['symbol'] == symbol) & (df.index >= self.start_date())] = \
                    np.where((close_prices[-1]/ema_values[-1]-1 > self.params.get('rsi\_threshold', 0)) * \
                             ((self._get_atr(df, symbol) / close_prices[-1]) - self.params.get('atr\_ratio', 1)), 1., 0.)

            # ... fill in the rest of your code here to apply signals and other logic...
        signals.iloc[:warmup] = 0.0
        return signals
