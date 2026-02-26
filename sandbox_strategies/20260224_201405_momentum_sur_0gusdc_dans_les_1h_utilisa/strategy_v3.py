from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
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
        # Define helper functions for Bollinger Bands, Donchian Channels, ADX, Supertrend and Stochastic indicators
        def bollinger_bands(df, indicator): 
            upper = np.roll(indicator['upper'], -1)
            middle = (indicator['lower'] + indicator['middle']) / 2
            lower = np.roll(indicator['lower'], -1)

            return df[["Close"]].merge(pd.DataFrame({"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower}), on="Close", how='left')

        def donchian_channels(df, indicator): 
            prev = np.roll(indicator['band'], -1)

            return df[["Close"]].merge(pd.DataFrame({"Donchian_Upper": [prev[-1]], "Donchian_Middle": (indicator['middle']), "Donchian_Lower": [prev[-2]]}), on="Close", how='left')

        def adx(df, indicator): 
            diff = abs(np.diff(df["Close"])) / df["Close"].shift() * indicator["plus_di"] + np.abs(np.diff(df["Close"])) / df["Close"].shift() * indicator["minus_di"]

            return pd.concat([pd.Series(indicator['adx']), diff], axis=1)

        def supertrend(df, indicator): 
            direction = np.sign(np.diff(df["Close"]))

            return pd.DataFrame({"Supertrend": [0] * len(df), "Direction": direction})

        def stochastic_stochKStdevD(df, indicator): 
            close = df[['Close']].values[-1:]
            high = np.array([np.max(close)]).reshape(-1)
            low = np.array([np.min(close)]).reshape(-1)

            return pd.concat([pd.Series(indicator["stoch_k"]), close, high, low], axis=1)
        signals.iloc[:warmup] = 0.0
        return signals
