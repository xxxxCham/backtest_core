from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='zecusdc_breakout_ichimoku_donchian_bollinger_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'donchian', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 3.5,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
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
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=3.5,
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
        # Boolean masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # Donchian bands
        dc = indicators['donchian']
        don_upper = np.nan_to_num(dc["upper"])
        don_middle = np.nan_to_num(dc["middle"])
        don_lower = np.nan_to_num(dc["lower"])

        # Bollinger bands
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])

        # Ichimoku components
        ich = indicators['ichimoku']
        tenkan = np.nan_to_num(ich["tenkan"])
        kijun = np.nan_to_num(ich["kijun"])
        senkou_a = np.nan_to_num(ich["senkou_a"])  # upper cloud
        senkou_b = np.nan_to_num(ich["senkou_b"])  # lower cloud

        # Long entry conditions
        long_cond = (
            (close > don_upper)
            & (close > indicators['bollinger']['upper'])
            & (tenkan > kijun)
            & (close > senkou_b)
        )
        long_mask[long_cond] = True

        # Short entry conditions
        short_cond = (
            (close < don_lower)
            & (close < indicators['bollinger']['lower'])
            & (tenkan < kijun)
            & (close < senkou_a)
        )
        short_mask[short_cond] = True

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        )

        # ATR-based SL/TP for short entries
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        )
        signals.iloc[:warmup] = 0.0
        return signals
