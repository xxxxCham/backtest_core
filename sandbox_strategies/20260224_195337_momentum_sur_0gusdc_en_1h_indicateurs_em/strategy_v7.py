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
        return ['ema', 'bollinger', 'volume_oscillator', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 5,
         'volume_oscillator_slow': 10,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=20,
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
                max_val=30,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])
        # EMA arrays
        ema_50 = ema_fast
        ema_200 = ema_slow
        # Volume oscillator check (3-period change)
        vol_roll = np.roll(volume_osc, 3)
        vol_roll[0:3] = np.nan
        vol_increase = (volume_osc > vol_roll) & (volume_osc > 0)
        vol_decrease = (volume_osc < vol_roll) & (volume_osc < 0)
        # EMA crossovers
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_200 = np.roll(ema_200, 1)
        prev_ema_50[0] = np.nan
        prev_ema_200[0] = np.nan
        cross_up_50_200 = (ema_50 > ema_200) & (prev_ema_50 <= prev_ema_200)
        cross_down_50_200 = (ema_50 < ema_200) & (prev_ema_50 >= prev_ema_200)
        # Price crossing bollinger bands
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_above_lower = (close > indicators['bollinger']['lower']) & (prev_close <= indicators['bollinger']['lower'])
        cross_below_upper = (close < indicators['bollinger']['upper']) & (prev_close >= indicators['bollinger']['upper'])
        # Stochastic confirmation (counter-consensus)
        prev_stoch_k = np.roll(indicators['stochastic']['stoch_k'], 1)
        prev_stoch_d = np.roll(indicators['stochastic']['stoch_d'], 1)
        prev_stoch_k[0] = np.nan
        prev_stoch_d[0] = np.nan
        stoch_cross_down = (indicators['stochastic']['stoch_k'] < indicators['stochastic']['stoch_d']) & (prev_stoch_k >= prev_stoch_d)
        # Long entry conditions
        long_entry = cross_up_50_200 & cross_above_lower & vol_increase
        long_mask[long_entry] = True
        # Short entry conditions
        short_entry = cross_down_50_200 & cross_below_upper & vol_decrease
        short_mask[short_entry] = True
        # Exit conditions
        exit_long = (close < indicators['bollinger']['middle']) | (ema_200 > close)
        exit_short = (close > indicators['bollinger']['middle']) | (ema_200 < close)
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
