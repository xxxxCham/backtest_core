from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ffusdc_30m_regime_adaptive_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'keltner', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 2.0,
         'tp_atr_mult_trend': 4.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
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

        # unwrap indicators
        bb = indicators['bollinger']
        kelt = indicators['keltner']
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr_arr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (close > bb["upper"]) & (close > kelt["upper"]) & (adx_arr > 25)
        short_mask = (close < bb["lower"]) & (close < kelt["lower"]) & (adx_arr > 25)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry levels
        if long_mask.any():
            entry_close = close[long_mask]
            entry_atr = atr_arr[long_mask]
            entry_adx = adx_arr[long_mask]
            tp_mult = np.where(entry_adx > 25, params["tp_atr_mult_trend"], params["tp_atr_mult_range"])
            df.loc[long_mask, "bb_stop_long"] = entry_close - params["stop_atr_mult"] * entry_atr
            df.loc[long_mask, "bb_tp_long"] = entry_close + tp_mult * entry_atr

        # Short entry levels
        if short_mask.any():
            entry_close = close[short_mask]
            entry_atr = atr_arr[short_mask]
            entry_adx = adx_arr[short_mask]
            tp_mult = np.where(entry_adx > 25, params["tp_atr_mult_trend"], params["tp_atr_mult_range"])
            df.loc[short_mask, "bb_stop_short"] = entry_close + params["stop_atr_mult"] * entry_atr
            df.loc[short_mask, "bb_tp_short"] = entry_close - tp_mult * entry_atr
        signals.iloc[:warmup] = 0.0
        return signals
