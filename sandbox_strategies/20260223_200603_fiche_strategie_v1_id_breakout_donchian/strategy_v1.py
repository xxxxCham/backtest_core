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
        def generate_signals(self, df: np.ndarray, indicators: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
                # Initialize signals series with zeros
                signals = np.zeros(len(df), dtype=np.float64)

                n = len(df)
                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                # Implement explicit LONG / SHORT / FLAT logic
                # Compute ATR values for calculating SL and TP levels
                atr = indicators['atr'][:-1]  # Exclude the last value as it is NaN in the first bar
                n_bars = len(df) - 2   # Adjust for excluding the last two bars (one for each warmup period)
                k_sl, tp, k_tp = params['k_sl'], params['tp'], params['k_tp']

                # Calculate moving averages and upper/lower bands for entering long positions
                ema14 = indicators['ema'][n_bars:][-2:]  # Use the last two bars' EMA values to avoid NaN after warmup period
                ema30 = indicators['ema'][:-n_bars - 1:-1]  
                upper, lower = (ema14 + k_sl * atr) / 2.0, (ema30 + k_tp * atr) / 2.0

                # Calculate the moving average of close prices for entering short positions
                ema50 = indicators['ema'][:-n_bars - 1:-1]  
                sma50 = np.mean(df[params["warmup"]:][:, 'close']) * params['k_sma'] / (2**0.5)    # Adjust for applying SMA smoothing factor

                # Enter long positions based on Bollinger Bands
                signals[(lower - ema14)[:n] < df[:, 'close'][params["warmup"]:][:, 0]] = 1.0  

                # Enter short positions based on Bollinger Bands (using average of the last two bars)
                signals[((df[:, 'close'] > upper[:n]) & (sma50 <= lower - k_sl * atr)) | ((df[:, 'close'] < lower)[:n] >= sma50)] = 2  

                # Enter flat positions based on Bollinger Bands
                signals[(lower - ema14) > df[:,:]['close'][params["warmup"]:][:, 0]] = 0.5   

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
