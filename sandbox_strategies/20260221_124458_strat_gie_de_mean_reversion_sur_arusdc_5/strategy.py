from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase_Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'stoch_rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
            # Assuming df['Close'] is the price series. Convert it into numpy array if not already one

            close = df['close'].values  # assuming 'close' exists in input dict with values

            indicators['bollinger']['upper'], indicators['bollinger']['middle'], indicators['bollinger']['lower'] = \
                indicators['bollinger'].get("upper"), indicators['bollinger'].get("middle"), indicators['bollinger'].get("lower")

            indicators['donchian']['upper'] = np.roll(indicators['donchian']['upper'], 1)
            indicators['donchian']['lower'] = np.roll(indicators['donchian']['lower'], -1) # Assuming 'donchian' exists in input dict with values

            adx, indicators['adx']['plus_di'], indicators['adx']['minus_di'] = indicators['adx'].get("adx"), indicators['adx'].get("plus_di"), indicators['adx'].get("minus_di")
            indicators['supertrend']['direction'] = indicators['supertrend'].get("direction") # Assuming 'supertrend' exists in input dict with values. If not, we can use indicators['supertrend']['up|down'] instead.

            indicators['stochastic']['stoch_k'], indicators['stochastic']['stoch_d'] = indicators['stochastic'].get("stoch_k"), indicators['stochastic'].get("stoch_d") # Assuming 'stochastic' exists in input dict with values

            atr = indicators['atr'].values  # assuming 'atr' already has values and it is a numpy array. If not, we need to convert the input indicator into one

            bb_stop_long = df["bb_low"].rolling(window=20).min() - atr * THRESHOLD  # Assuming 'bb_low' exists in input dict with values. This will be used for SL of long positions if ATR based stops are to be considered.

            bb_tp_long = df["bb_high"].rolling(window=20).max() + atr * THRESHOLD  # Assuming 'bb_high' exists in input dict with values. This will be used for TP of long positions if ATR based stops are to be considered.

            bb_stop_short = df["bb_high"].rolling(window=20).max() - atr * THRESHOLD  # Assuming 'bb_high' exists in input dict with values. This will be used for SL of short positions if ATR based stops are to be considered.

            bb_tp_short = df["bb_low"].rolling(window=20).min() + atr * THRESHOLD  # Assuming 'bb_low' exists in input dict with values. This will be used for TP of short positions if ATR based stops are to be considered.

            signals = pd.Series(np.zeros((len(close),)), index=df.index, dtype=np.float64)  # Assuming df['Close'] is the price series and 'Close' exists in input dict with values
        signals.iloc[:warmup] = 0.0
        return signals
