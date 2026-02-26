from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fees': 0.01, 'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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

            # Fill the required parameters in default_params dictionary
            leverage = params['leverage'] if 'leverage' in params else 1
            warmup = int(params.get('warmup')) if 'warmup' in params else 50

            # Calculate ATR
            atr = indicators['atr']

            close = df["close"].values
            window = max(int(params['window']), 2)
            roll_factor = params.get('roll_factor') if 'roll_factor' in params else 1

            # Compute rolling EMA and direction EMA for long term trend
            ema_long = pd.Series(pd.Series(close).ewm(span=window, min_periods=window-1).mean(), index=df.index)
            direction_ema = np.sign(np.diff(close))*params['roll_factor']/float(window)*params.get('direction_ema') if 'DIRECTIONAL_GROWTH' in params else 0

            # Compute ADX based on 20 days data
            adx = indicators['adx'] if "adx" in indicators else 0
            diff = np.sign(np.diff(close))*params['leverage']/float(window)*params.get('adx') if 'ADX' in params and 'leverage' in params else 0

            entry_mask = ((direction_ema > 0) & (diff > adx)) |((direction_ema == 0) & diff >= adx) # Long Entry Condition
            exit_mask = ((direction_ema < 0) | direction_ema==0)&(abs(close)<=adx) # Exit Conditions

            signals[(signals.values==1)&entry_mask]=-1.0 # Buy Signal and Buy Price above ATR + Roll Factor * EMA
            signals[(signals.values==-1)&exit_mask]=adx+params['roll_factor']# Sell Signal and Sell Price below ATR - Roll Factor*EMA

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
