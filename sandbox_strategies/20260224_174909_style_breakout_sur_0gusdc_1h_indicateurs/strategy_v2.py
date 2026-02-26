from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='style_breakout_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stochastic_overbought': 80,
         'stochastic_oversold': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volatility_threshold': 0.3,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.0,
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

        # Extract indicators
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        ema = np.nan_to_num(indicators['ema'])
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        open_ = df["open"].values

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Previous values for crossovers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above upper Bollinger, EMA rising, stochastic > 80, high volatility
        long_signal = (
            (close > upper) & (prev_close <= prev_upper) &
            (ema > prev_ema) &
            (k > params["stochastic_overbought"]) &
            (close > open_ + params["volatility_threshold"] * atr)
        )

        # Short entry: close crosses below lower Bollinger, EMA falling, stochastic < 20, high volatility
        short_signal = (
            (close < lower) & (prev_close >= prev_lower) &
            (ema < prev_ema) &
            (k < params["stochastic_oversold"]) &
            (close < open_ - params["volatility_threshold"] * atr)
        )

        long_mask = long_signal
        short_mask = short_signal

        # Exit conditions
        # Exit long: close crosses below middle Bollinger or stochastic < 20
        exit_long = (
            (close < middle) & (prev_close >= middle) |
            (k < params["stochastic_oversold"]) & (prev_k >= params["stochastic_oversold"])
        )

        # Exit short: close crosses above middle Bollinger or stochastic > 80
        exit_short = (
            (close > middle) & (prev_close <= middle) |
            (k > params["stochastic_overbought"]) & (prev_k <= params["stochastic_overbought"])
        )

        # Update exit masks
        exit_long_mask = exit_long
        exit_short_mask = exit_short

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        # For exit conditions, set to NaN or zero where needed
        # Already handled in mask logic
        signals.iloc[:warmup] = 0.0
        return signals