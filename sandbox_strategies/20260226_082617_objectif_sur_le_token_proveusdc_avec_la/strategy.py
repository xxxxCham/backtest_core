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
        return ['roc', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index)  # initialize DataFrame with zeros

            # Apply ROC and EMA technical indicators on the dataframe
            roc_dataframe = calculate_roc(df['close'], df['window'])
            ema_dataframe = apply_ema(df['close'], df['window'])

            # Create boolean masks for long and short positions based on ROC data
            long_mask, short_mask = create_signals_masks(roc_dataframe)

            # If leverage is enabled, use ATR-based stop loss and take profit levels
            if params.get("leverage") == 1:
                sl_level = calculate_atr(df['close'], df['window']) * params["sl_mult"]
                tp_level = calculate_atr(df['close'], df['window']) * params["tp_mult"]

            # Generate buy signals based on ROC data and ATR-based SL/TP levels
            signals[long_mask] = np.where((roc_dataframe > ema_dataframe) & (sl_level != 0), -1, signals[long_mask])

            # If no buy signal generated, generate a sell signal based on ROC data and ATR-based SL/TP levels
            if np.sum(signals) == 0:
                signals[short_mask] = np.where((roc_dataframe < ema_dataframe) & (tp_level != 0), -1, signals[short_mask])

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
