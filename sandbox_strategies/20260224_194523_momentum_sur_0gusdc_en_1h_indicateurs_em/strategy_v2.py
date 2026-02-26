from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_sur_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=100,
                max_val=300,
                default=200,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
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
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])

        # Compute EMA arrays
        fast_ema = ema_fast
        slow_ema = ema_slow

        # Compute crossover conditions
        prev_fast_ema = np.roll(fast_ema, 1)
        prev_slow_ema = np.roll(slow_ema, 1)
        prev_fast_ema[0] = np.nan
        prev_slow_ema[0] = np.nan

        cross_up = (fast_ema > slow_ema) & (prev_fast_ema <= prev_slow_ema)
        cross_down = (fast_ema < slow_ema) & (prev_fast_ema >= prev_slow_ema)

        # Entry conditions
        close = df["close"].values
        entry_long = cross_up & (close > indicators['bollinger']['lower']) & (volume_osc > 0)
        entry_short = cross_down & (close < indicators['bollinger']['upper']) & (volume_osc < 0)

        # Exit conditions
        exit_long = cross_down | (close > indicators['bollinger']['upper'])
        exit_short = cross_up | (close < indicators['bollinger']['lower'])

        # Apply entry signals
        long_mask = entry_long
        short_mask = entry_short

        # Apply exit signals
        exit_long_mask = exit_long
        exit_short_mask = exit_short

        # Apply long signals
        signals[long_mask] = 1.0
        # Apply short signals
        signals[short_mask] = -1.0

        # RSI filter for long entries (after entry)
        rsi_filter_long = rsi > params["rsi_overbought"]
        rsi_filter_short = rsi < params["rsi_oversold"]

        # Apply RSI filter to existing long entries
        long_entries = signals == 1.0
        rsi_long_mask = long_entries & rsi_filter_long
        # If RSI is overbought, we might want to reduce exposure or skip
        # Here we assume RSI filter is applied to confirm signal, not to cancel it
        # So we keep all long signals, but the RSI can be used for risk management

        # Write SL/TP columns if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # On long entries
        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # On short entries
        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
