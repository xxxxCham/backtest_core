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
        return {'fees': 10,
         'leverage': 1,
         'no_lookahead': 'true',
         'only_registry_indicators': 'true',
         'slippage': 5,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.75,
                max_val=4.0,
                default=2.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Compute moving averages and Bollinger Bands
            ma_periods = np.array(params['ma_periods'], dtype=int) if 'ma_periods' in params else [20] * len(indicators) 
            boll_window = np.array(params['boll_window'], dtype=int) if 'boll_window' in params else [20]  

            ma = df['close'].rolling(ma_periods).mean()
            std_dev = df['close'].rolling(boll_window).std()
            upper, middle, lower = ma + 2 * std_dev, ma, ma - 2 * std_dev

            # Generate signals based on the Bollinger Bands
            long_mask = (signals == 1) & (df['close'] > lower)
            short_mask = (signals == 0) & (df['close'] < upper)

            # Apply leverage and ATR risk management if available  
            if 'leverage' in params:
                sl_level = middle - boll_window * params['atr'] / 100.0
                tp_level = middle + boll_window * params['atr'] / 100.0

                long_mask |= (signals == 1) & (df['close'] < sl_level)
                short_mask &= ~(signals == 0) & ((upper - df['close']) <= tp_level)

            # Update signals based on the above conditions
            signals[long_mask] = 2.0
            signals[short_mask] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
