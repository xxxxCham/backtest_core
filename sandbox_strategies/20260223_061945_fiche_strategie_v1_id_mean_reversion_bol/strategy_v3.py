from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 2.5,
         'warmup': 40}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
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
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.5,
                max_val=3.5,
                default=2.5,
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
        # extract indicator arrays
        close = np.nan_to_num(df["close"].values)
        bb = indicators['bollinger']
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1); prev_y = np.roll(y, 1)
            prev_x[0] = np.nan; prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1); prev_y = np.roll(y, 1)
            prev_x[0] = np.nan; prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return cross_up(x, y) | cross_down(x, y)

        # long entry: close below lower band, rsi below oversold, rsi crosses below oversold
        long_mask = (
            (close < indicators['bollinger']['lower'])
            & (rsi < params["rsi_oversold"])
            & cross_down(rsi, np.full_like(rsi, params["rsi_oversold"]))
        )

        # short entry: close above upper band, rsi above overbought, rsi crosses above overbought
        short_mask = (
            (close > indicators['bollinger']['upper'])
            & (rsi > params["rsi_overbought"])
            & cross_up(rsi, np.full_like(rsi, params["rsi_overbought"]))
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions: close crosses middle band OR rsi crosses 50
        exit_cross = cross_any(close, indicators['bollinger']['middle'])
        exit_rsi = cross_any(rsi, np.full_like(rsi, 50))
        exit_mask = exit_cross | exit_rsi

        # set flat where exit condition met and currently in a position
        # For simplicity, assume positions close immediately on exit signal
        # Here we do not maintain position state; just clear signals at exit bars
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute ATR-based SL/TP on entry bars
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = (
                close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            )
            df.loc[entry_long, "bb_tp_long"] = (
                close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
            )
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = (
                close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            )
            df.loc[entry_short, "bb_tp_short"] = (
                close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
            )
        signals.iloc[:warmup] = 0.0
        return signals
