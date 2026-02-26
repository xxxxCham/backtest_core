from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_macd_atr_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 1.0,
         'leverage': 1,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 5.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.9,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        exit_long_mask = np.zeros(n, dtype=bool)
        exit_short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])  # 1 for bullish, -1 for bearish

        macd_d = indicators['macd']
        indicators['macd']['macd'] = np.nan_to_num(macd_d["macd"])
        indicators['macd']['signal'] = np.nan_to_num(macd_d["signal"])

        atr_threshold = float(params.get("atr_threshold", 1.0))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.2))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.9))

        # Helper cross functions
        prev_macd_line = np.roll(indicators['macd']['macd'], 1)
        prev_macd_signal = np.roll(indicators['macd']['signal'], 1)
        prev_macd_line[0] = np.nan
        prev_macd_signal[0] = np.nan
        cross_up = (indicators['macd']['macd'] > indicators['macd']['signal']) & (prev_macd_line <= prev_macd_signal)
        cross_down = (indicators['macd']['macd'] < indicators['macd']['signal']) & (prev_macd_line >= prev_macd_signal)

        # Entry conditions
        long_mask = (direction > 0) & (indicators['macd']['macd'] > indicators['macd']['signal']) & (atr > atr_threshold)
        short_mask = (direction < 0) & (indicators['macd']['macd'] < indicators['macd']['signal']) & (atr > atr_threshold)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long_mask = (direction < 0) | cross_down
        exit_short_mask = (direction > 0) | cross_up

        # Ensure exits do not override entries on same bar
        exit_long_mask &= ~long_mask
        exit_short_mask &= ~short_mask

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
