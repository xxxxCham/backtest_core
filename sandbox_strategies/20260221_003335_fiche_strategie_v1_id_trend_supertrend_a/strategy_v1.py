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
        for _, row in df.iterrows():
            long = False
            short = False
            for indicator, value in indicators.items():
                if len(value) == 1 and 'rsi' in indicator: # RSI
                    arr = np.array([row[indicator]])
                    rsi_val = calculate_rsi(arr)[0]
                    long = row['close'] > (np.mean(arr) + 2 * np.std(arr)) if 'ma_type' not in indicators and 'long' in indicator else True
                elif len(value) == 1 and 'ema' in indicator: # EMA
                    arr = np.array([row[indicator]])
                    ema_val = calculate_ema(arr)[0]
                    long = row['close'] > (np.mean(arr) + 2 * np.std(arr)) if 'long' in indicator else True
                elif len(value) == 1 and 'atr' in indicator: # ATR
                    arr = np.array([row[indicator]])
                    atr_val = calculate_atr(arr)[0]
                    long = row['close'] > (np.mean(arr) + 2 * atr_val) if 'long' in indicator else True
                elif len(value) == 3 and all(['rsi' in i for i in value]): # Bollinger, Donchian breakout
                    arr1 = np.array([row[indicator]])
                    indicators['bollinger']['upper'] = calculate_bollinger_band(arr1)[0][2] if 'long' not in indicator else calculate_donchian_band(arr1)['prev_upper']
                    long = row['close'] > (indicators['bollinger']['upper'] + 2 * calculate_bollinger_band(arr1)[0][3]) if 'ma_type' not in indicators and 'long' in indicator else True
                elif len(value) == 4 and all(['plus_di', 'minus_di', 'adx'] in value): # SuperTrend, ADX & Plus/Minus DI
                    arr1 = np.array([row[indicator]])
                    indicators['stochastic']['stoch_k'] = calculate_stochastic(arr1)[0][0] if 'stoch_k' in indicator else 0
                    adx_val = calculate_adx(arr1, slow=True)[0]
                    indicators['adx']['plus_di'] = calculate_plus_minus_di(arr1)['+'] if len(value)>4 and ('+' in value[-3]) or ( '-' in value[-2]) else 0
                    indicators['adx']['minus_di'] = calculate_plus_minus_di(arr1)['-'] if len(value)>5 and ('-' in value[6]) or ( '+' in value[7] ) else 0
                    superTrendVal = indicators['stochastic']['stoch_k'] + adx_val - indicators['adx']['plus_di'] #superTrend direction, as per the indicator description.
                    long = np.logical_and(superTrendVal > calculate_adx(arr1)[0][3], row['close'] < (row['open'] - 2 * calculate_atr(arr1)[0])) if 'long' in indicator else True
                elif len(value) == 4 and all(['supertrend', 'direction'] in value): # SuperTrend, direction.
                    arr1 = np.array([row[indicator]])
                    indicators['stochastic']['stoch_k'] = calculate_stochastic(arr1)[0][0] if 'stoch_k' in indicator else 0
                    superTrendVal = indicators['stochastic']['stoch_k'] + row['close'] - calculate_adx(arr1, slow=True)[0][3] #direction as per the indicator description.
                    long = np.logical_and(superTrendVal > calculate_adx(arr1)[0][3], row['close'] < (row['open'] - 2 * calculate_atr(arr1)[0])) if 'long' in indicator else True
                elif len(value) == 4 and all(['plus_di', 'minus_di', 'adx'] in value): # SuperTrend, ADX & Plus/Minus DI
                    arr1 = np.array([row[indicator]])
                    indicators['stochastic']['stoch_k'] = calculate_stochastic(arr1)[0][0] if 'stoch_k' in indicator else 0
                    adx_val = calculate_adx(arr1, slow=True)[0]
                    indicators['adx']['plus_di'] = calculate_plus_minus_di(arr1)['+' if len(value)>4 and ('+' in value[-3]) or ( '-' in value[-2] ) else '-']  # direction as per the indicator description.
                    indicators['adx']['minus_di'] = calculate_plus_minus_di(arr1)['-' if len(value)>5 and ('-' in value[6]) or ( '+' in value[7] ) else '+']   #direction as per the indicator description.
                    superTrendVal = indicators['stochastic']['stoch_k'] + adx_val - indicators['adx']['plus_di']  #superTrend direction, as per the indicator description.
                    long = np.logical_and(superTrendVal > calculate_adx(arr1)[0][3], row['close'] < (row['open'] - 2 * calculate_atr(arr1)[0])) if 'long' in indicator else True
                elif len(value) == 4 and all(['supertrend', 'direction'] in value): # SuperTrend, direction.
                    arr1 = np.array([row[indicator]])
                    indicators['stochastic']['stoch_k'] = calculate_stochastic(arr1)[0][0] if 'stoch_k' in indicator else 0
                    superTrendVal = indicators['stochastic']['stoch_k'] + row['close'] - calculate_adx(arr1, slow=True)[0][3] #direction as per the indicator description.
                    long = np.logical_and(superTrendVal > calculate_adx(arr1)[0][3], row['close'] < (row['open'] - 2 * calculate_atr(arr1)[0])) if 'long' in indicator else True
            short = not long # if the bar is a long, then make it short.
        signals.iloc[:warmup] = 0.0
        return signals
