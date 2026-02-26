from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.75, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=2.75,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        donchian = indicators['donchian']  # dict[str, ndarray]
        adx_dict = indicators['adx']  # dict[str, ndarray]
        atr = np.nan_to_num(indicators['atr'])  # ndarray
        close = df['close'].values
        stop_atr_mult = params.get('stop_atr_mult', 2.75)
        tp_atr_mult = params.get('tp_atr_mult', 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_mask = (close > indicators['donchian']["upper"]) & (adx_dict['adx'] > 35)
        short_mask = (close < indicators['donchian']["lower"]) & (adx_dict['adx'] > 35)
        exit_long_mask = ((close < indicators['donchian']["middle"]) | (adx_dict['adx'] < 15)) & (signals == 1.0)
        exit_short_mask = ((close > indicators['donchian']["middle"]) | (adx_dict['adx'] < 15)) & (signals == -1.0)

        # Write SL/TP columns into df if using ATR-based risk management
        df.loc[long_mask, 'bb_stop_long'] = close[long_mask]  - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, 'bb_tp_long']   = close[long_mask]  + tp_atr_mult * atr[long_mask]
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask | exit_short_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
