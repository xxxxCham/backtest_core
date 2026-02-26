from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_improved')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.0,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0,
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
        donchian = np.nan_to_num(indicators['donchian']['middle'])  # Middle band of Donchian Channels
        adx = np.nan_to_num(indicators['adx']['adx'])  # Average Directional Index
        atr = np.nan_to_num(indicators['atr'])  # Average True Range
        close = df["close"].values

        # Long entries: close > Donchian upper band and ADX > 35
        long_mask = (close > donchian) & (adx > 35)
        signals[long_mask] = 1.0

        # Short entries: close < Donchian lower band and ADX > 35
        short_mask = (close < donchian) & (adx > 35)
        signals[short_mask] = -1.0

        # Exits: close crosses below/above middle of Donchian or ADX < 20
        exit_long_mask = ((np.roll(close, 1) <= donchian) & (close > donchian)) | (adx < 20)
        signals[exit_long_mask] = 0.0
        exit_short_mask = ((np.roll(close, 1) >= donchian) & (close < donchian)) | (adx < 20)
        signals[exit_short_mask] = 0.0

        # ATR-based stop loss and take profit
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))
        entry_price = df["close"].where(signals != 0).ffill()  # Entry price is previous close on entry signal bars (NaN elsewhere)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[long_mask, "bb_stop_long"] = entry_price[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = entry_price[short_mask] + stop_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
