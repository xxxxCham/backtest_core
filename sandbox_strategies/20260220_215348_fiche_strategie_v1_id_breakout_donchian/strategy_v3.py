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

                # implement explicit LONG / SHORT logic 
                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                # warmup protection
                warmup = int(params.get("warmup", 50))
                signals.iloc[:warmup] = 0.0

                # ATR-based stop loss and take profit calculation (if applicable)
                atr = np.nan_to_num(indicators['atr'])
                close = df["close"].values

                entry_mask = signals == 1.0  # condition for entering a long position
                exit_mask = ((signals == -1.0) & (np.abs((df['close']-close[entry_mask])/(close[entry_mask]))>2)) | \
                            ((signals == 1.0) & (np.abs(close[exit_mask]-close[:-1][entry_mask])<2)) # condition for exiting a short position if no stop loss hit

                signals[(entry_mask)] = 1.0   # mark as long positions when entering the market
                signals[(exit_mask)] = -1.0    # mark as closed out of any open shorts

                sl_levels = close[entry_mask] - atr * n if exit_mask else close[entry_mask] + atr * n  # calculate stop loss levels
                tp_levels = close[entry_mask] + atr * n   # calculate take profit levels

                signals[(signals == 1.0) & (np.abs(close-sl_levels)<2)] = -1.0    # mark as short positions when entering the market
                signals[(signals == -1.0) & ((np.abs((df['close']-sl_levels)-close[:-1][entry_mask])<-2))] = 1.0   # mark as closed out of any open longs

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
