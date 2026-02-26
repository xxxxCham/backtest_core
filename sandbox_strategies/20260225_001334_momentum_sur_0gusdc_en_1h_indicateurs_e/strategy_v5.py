from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_momentum_ATR_ADX')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 20,
         'atr_period': 14,
         'ema_period': 30,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=100,
                default=30,
                param_type='int',
                step=1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=200,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=40,
                default=14,
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
        def generate_signals(indicators):
            def _bollinger_bands(df, window=20, stds=[2]):
                upper = df['close'].rolling(window).mean() + (std * df.close)
                lower = df['close'].rolling(window).mean() - (std * df.close)
                return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

            def _donchian_bands(df):
                prev_highs = np.roll(np.array(df['high']), 1)[::-1]
                indicators['donchian']['upper'] = df['close'].rolling(2).max()
                return pd.DataFrame({'prev_uppers': prev_highs, 'donchian_upper': indicators['donchian']['upper']})

            def _adx_and_plusminus_di(df):
                adx = df['low'].diff().abs().rolling(window).mean() / df.low.diff().abs().rolling(window).mean()
                indicators['adx']['plus_di'], indicators['adx']['minus_di'] = df['high'] - df['close'], (df['open'] - df['low']) if 'close' not in df else 0
                return pd.DataFrame({'adx': adx, 'plus_di': indicators['adx']['plus_di'], 'minus_di': indicators['adx']['minus_di']})

            def _supertrend(df):
                super_trend = (np.array(df['high']) + np.array(df['low'])) / 2 * df['close'] > (0 if len(df) < 3 else df['super_trend'].rolling(window).mean())
                return pd.DataFrame({'supertrend': super_trend})

            def _stochastic(df):
                stoch = ((np.array(df['high']) + np.array(df['low'])) / 2 - (np.array(df['close']))) / ((0 if len(df) < 3 else df['super_trend'].rolling(window).mean()) * 100)
                return pd.DataFrame({'stoch_k': stoch, 'stoch_d': stoch})

            signals = pd.Series(np.nan, index=df.index, dtype=float)

            if any('ema' in i for i in indicators):
                ema = df['close'].ewm(span=indicators['ema'], adjust=False).mean() - (100 * df[i])
        signals.iloc[:warmup] = 0.0
        return signals
