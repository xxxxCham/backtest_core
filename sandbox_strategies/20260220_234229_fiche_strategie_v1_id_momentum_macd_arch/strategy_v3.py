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
        def generate_signals(indicator, data, default_params={}, leverage=1):
            # Check if the inputs are valid
            assert isinstance(data, pd.Series)

            # Get available indicators
            available_indicators = ['rsi', 'ema', 'atr']
            assert indicator in available_indicators, f"Invalid indicator: {indicator}. Available indicators: {available_indicators}"

            # Prepare input data
            bars = data['close'].to_numpy()  # Assuming close price column is `close`
            bars.shape   # Check the shape of your array

            # Normalize inputs (if necessary)
            if 'rsi' in indicator:
                rsi_period, ema_fast_period, ema_slow_period = default_params[indicator]
                rsi_values = 100. - len(bars)*1./(1.+np.array([min(bars), max(bars)]))*rsi_period/ema_fast_period
            elif 'ema' in indicator:
                ema_fast_period, ema_slow_period = default_params[indicator]
                ema_values = np.mean(np.array([min(bars), max(bars)])) - 2.*(np.array([min(bars), max(bars)])-np.roll(np.array([min(bars), max(bars)]),1))/(ema_fast_period*len(bars)/sum(bars<bars[0]))
            elif 'atr' in indicator:
                atr_period = default_params[indicator]
                atr_values = np.std(np.array([min(bars), max(bars)]))-2.*np.mean((np.array([min(bars), max(bars)])-np.roll(np.array([min(bars), max(bars)]),1)))/(len(bars)*atr_period)
            else:
                raise ValueError("Unsupported indicator type")

            # Compute indicators based on input data and write signals to dataframe
            indicators['rsi'] = rsi_values if 'rsi' in default_params else (np.nan, np.nan)  # Assuming a DataFrame `df` with columns ['close', ...]
            df['ema_fast'], df['ema_slow'] = ema_fast_period*np.array([min(bars), max(bars)]), ema_slow_period*np.array([min(bars), max(bars)]) if 'ema' in default_params else (np.nan, np.nan)
            indicators['atr'], _ = atr_values if 'atr' in default_params else [0., 0.]  # Assuming a DataFrame `df` with columns ['close', ...]

            signals = pd.Series(0.0, index=bars.tolist())
            signals[np.where((bars > df['ema_fast'])|(bars < df['ema_slow'])) & (~ np.isnan(signals))] = 1. if bars >= indicators['rsi'][1] else -1. if bars <= indicators['rsi'][0] else 0
            signals[np.where((df['close'].shift() > df['bb_stop_long']) | (df['close'].shift() < df['bb_tp_long'])) & (~ np.isnan(signals))] = -1.0 if bars >= df['bb_tp_long'] else 1
            signals[np.where((df['close'].shift() > df['bb_stop_short']) | (df['close'].shift() < df['bb_tp_short'])) & (~ np.isnan(signals))] = -1.0 if bars <= df['bb_tp_short'] else 1

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
