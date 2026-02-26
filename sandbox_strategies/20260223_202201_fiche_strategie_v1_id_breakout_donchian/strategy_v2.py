from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIEv1')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 30,
         'rsi_oversold': 70,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            # Implement explicit LONG / SHORT / FLAT logic here

            # Write SL/TP columns into df if using ATR-based risk management
            stop_loss_mult = params.get("stop_loss_mult", 1.5)
            take_profit_mult = params.get("take_profit_mult", 2.0)

            # Calculate ATR values for calculating SL and TP levels
            atr_periods = indicators['atr'].history(len(df))
            close = df["close"]

            # Compute stop loss and take profit levels using ATR-based risk management
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            sl_level = close - params["leverage"] * atr_periods[0] * stop_loss_mult
            tp_level = close + params["leverage"] * atr_periods[-1] * take_profit_mult

            signals[(close < sl_level) & (long_mask == False)] = 1.0 # Long entry signal when price falls below SL and there is no long position yet
            signals[(close > tp_level) & (short_mask == False)] = -1.0   # Short entry signal when price rises above TP and there is no short position yet

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
