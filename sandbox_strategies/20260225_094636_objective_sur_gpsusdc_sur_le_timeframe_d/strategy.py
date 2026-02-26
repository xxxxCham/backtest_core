from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
        def generate_signals(df, long_intent=True, short_intent=True):
            # Define default parameters (leverage = 1)
            default_params = {'long': {}, 'short': {}}

            # Check for LONG or SHORT entry based on intents
            if long_intent:
                signals['long'] = np.where(df[indicators['rsi']] > df[indicators['rsi']].mean() + 2, 1.0, 0)

            # Define upper and lower Bollinger Bands based on previous period close price
            prev_close = np.roll(df['close'], -1)[:-1]
            indicators['donchian']['upper'] = df['close'] + (indicators['atr'] * 2).cumsum() / 2
            indicators['donchian']['lower'] = df['close'] - (indicators['atr'] * 2).cumsum() / 2

            # Check for LONG or SHORT entry based on momentum and risk control
            signals['long'] |= np.where(signals['long'] > 0, 1.0, 0) & \
                                np.where((df[indicators['rsi']] < df[indicators['rsi']]) | \
                                         (df[indicators['atr']] < DONCHIAN_LOWER), -1.0, signals['long'])

            # Check for SHORT entry based on momentum and risk control
            signals['short'] |= np.where(signals['short'] > 0, 1.0, 0) & \
                                 np.where((df[indicators['rsi']] > df[indicators['rsi']]) | \
                                          (df[indicators['atr']] > DONCHIAN_UPPER), -1.0, signals['short'])

            # Define stop-loss and take profit levels based on ATR-based SL/TP rules
            df['bb_stop_long'], df['bb_tp_long'] = bb_stoploss_takeprofit(df)
            df['bb_stop_short'], df['bb_tp_short'] = bb_stoploss_takeprofit(df, True)

            # Update signals with new stop-loss and take profit levels if present
            signals[long] |= np.where((df['close'] < df['bb_stop_long']) | (df['close'] > df['bb_tp_short']), 0.0, signals[long])

            # Return generated signals
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
