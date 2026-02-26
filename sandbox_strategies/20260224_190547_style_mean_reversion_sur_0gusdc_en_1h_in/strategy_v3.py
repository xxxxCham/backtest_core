from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stochastic_volume_donchian_mean_reversion_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'volume_oscillator', 'donchian', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'donchian_period': 20,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_short': ParameterSpec(
                name='volume_oscillator_short',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        dc = indicators['donchian']
        indicators['donchian']['upper'] = np.nan_to_num(dc["upper"])
        indicators['donchian']['lower'] = np.nan_to_num(dc["lower"])
        bb = indicators['bollinger']
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Compute previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        prev_d = np.roll(d, 1)
        prev_d[0] = np.nan
        prev_vol_osc = np.roll(vol_osc, 1)
        prev_vol_osc[0] = np.nan
        prev_dc_lower = np.roll(indicators['donchian']['lower'], 1)
        prev_dc_lower[0] = np.nan
        prev_dc_upper = np.roll(indicators['donchian']['upper'], 1)
        prev_dc_upper[0] = np.nan

        # Long entry: k crosses below 20, volume oscillator negative, donchian lower crosses below close
        long_entry_k_cross = (k < 20) & (prev_k >= 20)
        long_entry_vol_neg = (vol_osc < 0)
        long_entry_dc_cross = (indicators['donchian']['lower'] < close) & (prev_dc_lower >= close)
        long_mask = long_entry_k_cross & long_entry_vol_neg & long_entry_dc_cross

        # Short entry: k crosses above 80, volume oscillator positive, donchian upper crosses above close
        short_entry_k_cross = (k > 80) & (prev_k <= 80)
        short_entry_vol_pos = (vol_osc > 0)
        short_entry_dc_cross = (indicators['donchian']['upper'] > close) & (prev_dc_upper <= close)
        short_mask = short_entry_k_cross & short_entry_vol_pos & short_entry_dc_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based risk management
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
