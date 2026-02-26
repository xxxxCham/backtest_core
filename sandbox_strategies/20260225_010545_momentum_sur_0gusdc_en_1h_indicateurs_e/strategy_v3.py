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
        def generate_signals(df, indicators, params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Iterate over each bar in the dataframe
            for i, row in df.iterrows():
                close = row['close']

                # Compute Bollinger Bands parameters based on `params` and current bar
                bb_mult1, bb_mult2, _ = get_bollingerband_multipliers(i, params)
                bbl = compute_bb_length(row.high, row.low, close, i, params['period'])

                # Compute RSI based on current bar and previous `params` period
                rsi1, _ = compute_rsi(close, bb_mult2)

                # Adjusted Close values for ATR calculation
                adj_closes = adjust_for_atr(row.high, row.low, close, i, params['window'])

                # Compute the current bar's ATR using previous `params` window length
                atr1, _ = compute_atr(adj_closes, i)

                # Update signals based on RSI and Bollinger Band crossings
                if rsi1 > 50:  # Buy signal at or above 50%
                    signals[i] = 1.0  
                elif rsi1 < 50:  # Sell short below 50%
                    signals[i] = -1.0   

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
