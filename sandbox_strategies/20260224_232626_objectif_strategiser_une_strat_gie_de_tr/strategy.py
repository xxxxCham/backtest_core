from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='GUSDC_ComplexIndicators')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_period': 14,
         'stop_atr_mult': 1.8,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=2,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.6,
                max_val=3.8,
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.6,
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
        def generate_signals(df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # implement explicit LONG / SHORT / FLAT logic
            # warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            long_mask = np.zeros(len(df), dtype=bool)
            short_mask = np.zeros(len(df), dtype=bool)

            # ATR-based risk management SL & TP levels computation
            close = df["close"].values
            atr = indicators['atr']
            k_sl = params.get("k_sl", 1.5)
            k_tp = params.get("k_tp", 2.0)

            # compute ATR values for upper and lower bands (bollinger bands)
            bp, sp, _ = bollinger_bands(close, k=2*k_sl - 1)
            bs, ts, _ = bollinger_bands(close, k=2*k_tp + 1)

            # compute entry signals based on Bollinger Bands and ATR values
            long_signal = ((close > bp) & (indicators['bollinger'] < indicators['bollinger'])) | \
                          ((close < bs) & (indicators['bollinger'] > indicators['bollinger']))

            short_signal = ((close < sp) & (indicators['bollinger'] > indicators['bollinger'])) | \
                           ((close > ts) & (indicators['bollinger'] < indicators['bollinger']))

            # apply stop loss and take profit levels based on input parameters
            signals[long_mask] = 1.0 if long_signal else 0.0
            signals[short_mask] = -1.0 if short_signal else 0.0

            signals[(close < bs) & (indicators['bollinger'] > indicators['bollinger']) | \
                    (close > ts) & (indicators['bollinger'] < indicators['bollinger'])] *= -1.0 #if close is outside Bollinger Bands, change the signal direction to opposite

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
