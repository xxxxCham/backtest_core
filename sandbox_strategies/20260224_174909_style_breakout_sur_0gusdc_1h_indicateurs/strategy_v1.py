from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_style_ema_bollinger_stochastic')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 50,
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
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])
        # Prepare previous arrays for crossovers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_bb_lower[0] = np.nan
        # Long entry conditions
        # Close crosses above indicators['bollinger']['upper']
        cross_above_bb = (close > indicators['bollinger']['upper']) & (prev_close <= prev_bb_upper)
        # EMA is rising
        ema_rising = (ema > prev_ema)
        # Stochastic overbought
        stochastic_overbought = (indicators['stochastic']['stoch_k'] > params["stochastic_overbought"])
        # Volatility candle
        vol_candle = ((close - prev_close) / prev_close) > params["volatility_threshold"]
        # Long mask
        long_mask = cross_above_bb & ema_rising & stochastic_overbought & vol_candle
        # Short entry conditions
        # Close crosses below indicators['bollinger']['lower']
        cross_below_bb = (close < indicators['bollinger']['lower']) & (prev_close >= prev_bb_lower)
        # EMA is falling
        ema_falling = (ema < prev_ema)
        # Stochastic oversold
        stochastic_oversold = (indicators['stochastic']['stoch_k'] < params["stochastic_oversold"])
        # Volatility candle
        vol_candle_short = ((prev_close - close) / prev_close) > params["volatility_threshold"]
        # Short mask
        short_mask = cross_below_bb & ema_falling & stochastic_oversold & vol_candle_short
        # Exit conditions
        # Close crosses below indicators['bollinger']['middle']
        exit_long = (close < indicators['bollinger']['middle']) & (prev_close >= indicators['bollinger']['middle'])
        exit_short = (close > indicators['bollinger']['middle']) & (prev_close <= indicators['bollinger']['middle'])
        # Stochastic oversold/overbought for exit
        stoch_exit_long = (indicators['stochastic']['stoch_k'] < params["stochastic_oversold"])
        stoch_exit_short = (indicators['stochastic']['stoch_k'] > params["stochastic_overbought"])
        # Combine exit conditions
        exit_long_mask = exit_long | stoch_exit_long
        exit_short_mask = exit_short | stoch_exit_short
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Overwrite exit signals
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
