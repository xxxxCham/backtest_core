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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            n = len(df)
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # implement explicit LONG / SHORT / FLAT logic
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            rsi = indicators['rsi']
            ema = indicators['ema']
            atr = indicators['atr']

            # RSI conditions - overbought/oversold thresholds at 70/30 respectively
            rsi_overbought = np.where(rsi > params['rsi']['overbought'], 1, 0)
            rsi_oversold = np.where(rsi < params['rsi']['oversold'], 1, 0)

            # EMA conditions - short term MA above long term MA (for both RSI and Bollinger Bands)
            ema_short = df["close"].ewm(span=params['ema']['span']).mean()
            ema_long = df["close"].rolling(window=params['ema']['window'], min_periods=params['ema']['min_periods']).mean()

            rsi_short = df["close"].apply(lambda x: 100 - 100 / (1 + x/max(x)))
            rsi_long = df["close"].rolling(window=params['rsi']['window'], min_periods=params['rsi']['min_periods']).apply(lambda x: 100 - 100 / (1 + x/max(x)))

            # Bollinger Bands conditions and signals for long positions
            bb = df["close"].rolling(window=params['boll']['window'], min_periods=params['boll']['min_periods']).mean()
            bol_upper, bol_middle, bol_lower = df["close"].rolling(window=params['boll']['window'], min_periods=params['boll']['min_periods']).apply(lambda x: params['boll'][f'{x.name}_bb_band'])

            bol_upper[np.isnan(bol_upper)] = 100
            bol_middle[np.isnan(bol_middle)] = df["close"].rolling(window=params['boll']['window'], min_periods=params['boll']['min_periods']).mean()

            bb_band_short = (df["close"] - bol_lower) / abs((df["close"] - bol_middle)) * 2
            signals[bb_band_short > params['boll']['upper_band_threshold']] = 1.0

            # Implementing long positions when RSI and BB conditions are met, and for overbought/oversold levels on RSI
            signals[(rsi == rsi_overbought) & (bb >= bol_middle)] = 1.0
            signals[long_mask] = signals[signals != 0].fillna(method='bfill')

            # Implementing short positions when RSI and BB conditions are met, and for oversold levels on RSI
            signals[(rsi == rsi_oversold) & (bb <= bol_middle)] = -1.0
            signals[short_mask] = signals[signals != 0].fillna(method='bfill')

            # Implementing flat positions when neither long/short conditions are met
            signals[(rsi == rsi_overbought) & (bb <= bol_middle)] *= 0.5
            signals[(rsi == rsi_oversold) & (bb >= bol_middle)] *= -1.2

            return signals[long_mask | short_mask] # Return only the long/short positions, discarding flat ones
        signals.iloc[:warmup] = 0.0
        return signals
