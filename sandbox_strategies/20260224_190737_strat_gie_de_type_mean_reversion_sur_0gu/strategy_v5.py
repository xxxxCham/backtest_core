from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_vortex_trend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
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
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
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
                min_val=2.0,
                max_val=4.5,
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

        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        oscillator = np.nan_to_num(indicators['vortex']["oscillator"])
        adx_d = indicators['adx']
        adx = np.nan_to_num(adx_d["adx"])

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Define previous values for crossovers
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_vortex_osc = np.roll(oscillator, 1)
        prev_vortex_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vortex_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_adx = np.roll(adx, 1)

        # Set first values to NaN for proper crossover detection
        prev_upper_bb[0] = np.nan
        prev_lower_bb[0] = np.nan
        prev_vortex_osc[0] = np.nan
        prev_vortex_plus[0] = np.nan
        prev_vortex_minus[0] = np.nan
        prev_adx[0] = np.nan

        # Entry conditions
        # Long entry: price crosses above lower band, vortex oscillator crosses above 0.5, ADX > 25
        cross_above_lower = (df["close"].values > lower_bb) & (prev_lower_bb <= lower_bb)
        vortex_cross_above_05 = (oscillator > 0.5) & (prev_vortex_osc <= 0.5)
        adx_above_25 = adx > 25

        long_entry = cross_above_lower & vortex_cross_above_05 & adx_above_25
        long_mask = long_entry

        # Short entry: price crosses below upper band, vortex oscillator crosses above 0.5, ADX > 25
        cross_below_upper = (df["close"].values < upper_bb) & (prev_upper_bb >= upper_bb)
        short_entry = cross_below_upper & vortex_cross_above_05 & adx_above_25
        short_mask = short_entry

        # Exit conditions
        # Exit long: price crosses above upper band OR vortex oscillator crosses below 0.5
        exit_long = (df["close"].values > upper_bb) | ((oscillator < 0.5) & (prev_vortex_osc >= 0.5))
        exit_short = (df["close"].values < lower_bb) | ((oscillator < 0.5) & (prev_vortex_osc >= 0.5))

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit signals to 0.0
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Risk management
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Calculate dynamic stop loss and take profit for longs
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        # Calculate dynamic stop loss and take profit for shorts
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals