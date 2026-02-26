from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_loss': ParameterSpec(
                name='stop_loss',
                min_val=0.9,
                max_val=2.0,
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'ATR_period': ParameterSpec(
                name='ATR_period',
                min_val=4,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
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
        def generate_signals(self, data):
            # Initialize DataFrame for Bollinger Bands calculations
            bars = pd.DataFrame()

            # Calculate BB Upper and Lower bands on 20 period MA
            bb14, bb28 = self._calculate_bollinger_bands(data)

            # Check if the data is valid before continuing
            if not (bb14 is None or bb28 is None):
                bars['BB_UPPER'] = data.close - bb28
                bars['BB_LOWER'] = data.close + bb28

            else:
                # If invalid, use 50 period MA for Bollinger Bands calculations and apply rules accordingly
                self._calculate_bollinger_bands(data)

            # Calculate rate of change on 14 periods (ROC)
            roc = self._compute_roc(data.close)

            # Check if the data is valid before continuing
            if not (roc is None):
                bars['ROI'] = abs((data[-1] - data[0]) / data[0]) * 100

            else:
                self._compute_roc(data)

            # Check if ROC values are available and generate signals accordingly
            if not (roc is None):
                # Define rules for long and short positions based on BB Upper, BB Lower, ROI
                self.on_enter_long = (bb28 > bb14 & data[-1] - data[0] < 0) | ((data[-1] - data[0]) / data[0] < bars['ROI'] & abs((roc * 100)) > 5)
                self.on_enter_short = (bb28 < bb14 & data[-1] - data[0] > 0) | ((data[-1] - data[0]) / data[0] > bars['ROI'] & abs((roc * 100)) > 5)
                self.on_order = (bb28 < bb14 & data[-1] - data[0] < 0) | ((data[-1] - data[0]) / data[0] > bars['ROI'] & abs((roc * 100)) <= 5)|(bb28 > bb14 & data[-1] - data[0] > 0)

                # Generate signals based on Bollinger Bands and ROC values
                for index, row in bars.iterrows():
                    if (row['BB_UPPER'] < data[index]) & (data[index + 1] >= bb28):
                        self.on_enter_long()

                    elif (row['BB_LOWER'] > data[index]) & (data[index + 1] <= bb28):
                        self.on_enter_short()

                # Check if orders are valid before executing them
        signals.iloc[:warmup] = 0.0
        return signals
