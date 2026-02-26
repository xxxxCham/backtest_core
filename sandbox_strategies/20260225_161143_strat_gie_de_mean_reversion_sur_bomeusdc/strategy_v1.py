from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_rsi_obv_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['stoch_rsi', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stoch_rsi_overbought': 80,
         'stoch_rsi_oversold': 20,
         'stoch_rsi_period': 14,
         'stop_atr_mult': 2.3,
         'tp_atr_mult': 4.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_rsi_period': ParameterSpec(
                name='stoch_rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=4.5,
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

        # Wrap indicator arrays
        indicators['stoch_rsi']['k'] = np.nan_to_num(indicators['stoch_rsi']["k"])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # previous OBV for trend confirmation
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # Entry conditions
        oversold = params.get("stoch_rsi_oversold", 20)
        overbought = params.get("stoch_rsi_overbought", 80)
        long_mask = (indicators['stoch_rsi']['k'] < oversold) & (obv > prev_obv)
        short_mask = (indicators['stoch_rsi']['k'] > overbought) & (obv < prev_obv)

        # Exit condition: cross 50
        prev_k = np.roll(indicators['stoch_rsi']['k'], 1)
        prev_k[0] = np.nan
        cross_up = (indicators['stoch_rsi']['k'] > 50) & (prev_k <= 50)
        cross_down = (indicators['stoch_rsi']['k'] < 50) & (prev_k >= 50)
        exit_mask = cross_up | cross_down

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Exit to flat on cross
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 2.3)
        tp_mult = params.get("tp_atr_mult", 4.5)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
