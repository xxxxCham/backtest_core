from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATR stop mult': {'max': 2, 'min': 1.5},
         'ATR take profit mult': {'max': 4, 'min': 3},
         'StopLossMultiplier': '1.5x',
         'TakeProfitMultiplier': '4.0x',
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'StopLossMultiplier': ParameterSpec(
                name='StopLossMultiplier',
                min_val=0.5,
                max_val=4.0,
                default='1.5x',
                param_type='float',
                step=0.1,
            ),
            'TakeProfitMultiplier': ParameterSpec(
                name='TakeProfitMultiplier',
                min_val=0.5,
                max_val=4.0,
                default='4.0x',
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            # implement explicit LONG / SHORT / FLAT logic
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            if self.use_ATR:
                atr = np.nan_to_num(indicators["atr"])
                close = df["close"].values

                # ATR-based stop loss and take profit implementation (if applicable)
                entry_mask = (signals == 1.0)  

                signals[entry_mask] = close[entry_mask] - params.get("k_sl", 2.5 * atr[entry_mask]) # use k_sl as the stop loss multiplier if it is provided, otherwise default to 2.5*ATR
                targets = (close[entry_mask] + close[entry_mask] + params.get("tp", 10)) / 3 # calculate take profit levels using tp

                signals[entry_mask] = np.where((signals[entry_mask:].values > targets) | (signals[entry_mask-1:-1].diff().abs() < params.get("tolerance")), -1, 0) # check if price is above take profit levels and/or volatility is below tolerance
                signals = np.where(long_mask & entry_mask, 1, signals) # use long signal for entries within warmup period and short signal otherwise

            return signals
        return signals
