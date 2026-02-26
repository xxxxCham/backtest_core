from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_volume_breakout_ltcusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr', 'stochastic']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
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

        # Prepare indicators
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        stochastic = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(indicators['stochastic']["stoch_d"])

        # Close price
        close = df["close"].values

        # Previous values for crossovers
        prev_ema = np.roll(ema, 1)
        prev_obv = np.roll(obv, 1)
        prev_volume_oscillator = np.roll(volume_oscillator, 1)
        prev_close = np.roll(close, 1)
        prev_stoch_k = np.roll(indicators['stochastic']['stoch_k'], 1)
        prev_stoch_d = np.roll(indicators['stochastic']['stoch_d'], 1)

        # Set first values to NaN to avoid false crossovers
        prev_ema[0] = np.nan
        prev_obv[0] = np.nan
        prev_volume_oscillator[0] = np.nan
        prev_close[0] = np.nan
        prev_stoch_k[0] = np.nan
        prev_stoch_d[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above ema, obv increases, volume oscillator peaks, stochastic K > D
        long_entry = (close > ema) & (prev_close <= prev_ema) & (obv > prev_obv) & (volume_oscillator > prev_volume_oscillator) & (indicators['stochastic']['stoch_k'] > indicators['stochastic']['stoch_d']) & (prev_stoch_k <= prev_stoch_d)
        long_mask = long_entry

        # Short entry: close crosses below ema, obv decreases, volume oscillator peaks, stochastic K < D
        short_entry = (close < ema) & (prev_close >= prev_ema) & (obv < prev_obv) & (volume_oscillator < prev_volume_oscillator) & (indicators['stochastic']['stoch_k'] < indicators['stochastic']['stoch_d']) & (prev_stoch_k >= prev_stoch_d)
        short_mask = short_entry

        # Exit conditions
        # Exit long: close crosses below ema or volume oscillator turns negative
        long_exit = (close < ema) | (volume_oscillator < 0)
        long_mask = long_mask & ~long_exit

        # Exit short: close crosses above ema or volume oscillator turns positive
        short_exit = (close > ema) | (volume_oscillator > 0)
        short_mask = short_mask & ~short_exit

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Risk management
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        # Write SL/TP columns into df if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
