from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ffusdc_30m_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 2.0,
         'tp_atr_mult_trend': 4.2,
         'warmup': 50}

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
                max_val=3.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=2.0,
                max_val=6.0,
                default=4.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=3.0,
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

        # Wrap indicator arrays
        bb = indicators['bollinger']
        kelt = indicators['keltner']
        atr = np.nan_to_num(indicators['atr'])
        close = np.nan_to_num(df["close"].values)

        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])

        # Entry conditions
        long_mask = (close > indicators['bollinger']['upper']) & (close > indicators['keltner']['upper'])
        short_mask = (close < indicators['bollinger']['lower']) & (close < indicators['keltner']['lower'])

        # Exit conditions using cross detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down_mid = (close < indicators['keltner']['middle']) & (prev_close >= indicators['keltner']['middle'])
        cross_up_mid = (close > indicators['keltner']['middle']) & (prev_close <= indicators['keltner']['middle'])

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Calculate ATR-based SL/TP on entry bars
        stop_atr_mult = params.get("stop_atr_mult", 1.6)
        tp_trend_mult = params.get("tp_atr_mult_trend", 4.2)

        entry_long = long_mask
        entry_short = short_mask

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_trend_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_trend_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
