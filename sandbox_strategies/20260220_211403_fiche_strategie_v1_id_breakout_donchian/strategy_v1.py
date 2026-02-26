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
        def generate_signals(df):
            indicators = {
                'rsi': df['close'].rolling(21).RSI(),
                'ema': df.close.ewm(span=20, adjust=False).mean().rename('EMA'),
                'atr': ta.ATR(df),
                # Add other indicators here...
            }

            signals = pd.Series(np.zeros_like(df.index), index=df.index)
            default_params = dict()  # Define your default parameters here for ATR-based SL/TP

            long_intent = False
            short_intent = False

            prev_close = df['close'].shift(-1).fillna(method='bfill')

            for i, row in df.iterrows():
                # Long entry conditions: momentum is up and risk level is controlled
                if indicators[i]['rsi'] > 50 or (indicators[i - 1]['rsi'] < 50 and long_intent) \
                        and abs(row['close'] / prev_close - 1.0) <= default_params['atr']:
                    signals[i] = 1.0

                # Short entry conditions: momentum is down and risk level is controlled
                elif indicators[i]['rsi'] < 50 or (indicators[i - 1]['rsi'] > 50 and short_intent) \
                        and abs(row['close'] / prev_close - 1.0) <= default_params['atr']:
                    signals[i] = -1.0

                # Always check for confirmation of momentum indicators, Bollinger Bands, Donchian Channels, Supertrend direction and STOCHastics K & D:

                # Confirmation of RSI uptrend (if LONG intent) or downtrend (if SHORT intent):
                if long_intent and row['rsi'] > 50 \
                        or short_intent and row['rsi'] < 50:
                    signals[i] = 0.0

            # Calculate stop loss (SL) price levels based on ATR-based SL/TP:

            prev_upper, __, __ = ta.DONCHUAN(df).donchian()[:3][::-1].values  
            bb_stop_long, _, _ = calculate_sl_tp("bb_stop_long", df)
            bb_tp_long, _, _ = calculate_sl_tp("bb_tp_long", df)

            signals[signals < -bb_stop_long] = 0.0  
            signals[(prev_close / row['close'] > prev_upper[-1]) and (row['rsi'] <= 50)] = 0.0

            # Calculate take profit (TP) price levels based on ATR-based SL/TP:

            bb_stop_short, _, _ = calculate_sl_tp("bb_stop_short", df)
            bb_tp_short, _, _ = calculate_sl_tp("bb_tp_short", df)

            signals[signals > bb_tp_long] = 0.0  
            signals[(prev_close / row['close'] < prev_upper[-1]) and (row['rsi'] >= 50)] = 0.0

            # Confirmation of Bollinger Bands, Donchian Channels and STOCHastics K & D:
        signals.iloc[:warmup] = 0.0
        return signals
