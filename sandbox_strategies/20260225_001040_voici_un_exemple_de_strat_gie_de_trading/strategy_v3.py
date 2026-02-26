from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Vortex_Snake_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_stddev': 2,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'fast_length': ParameterSpec(
                name='fast_length',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'slow_length': ParameterSpec(
                name='slow_length',
                min_val=5,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=5,
                max_val=30,
                default=20,
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
        def generate_signals(indicators, df):
            # Define default parameters for strategies in case they are not provided
            default_params = {
                'vortex': {'fast_length': 10, 'slow_length': 20},
                'bollinger': {'window': 20},
                'donchian': {'bars': 20},
                # ... add more strategies with their respective parameters here if needed
            }

            signals = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float64)

            for i in df.index:
                # Loop through each bar's data

                for strategy_name, params in default_params.items():
                    if strategy_name not in indicators or 'bollinger' not in indicators[strategy_name] or \
                       'donchian' not in indicators[strategy_name] or 'supertrend' not in indicators[strategy_name] or \
                       'adx' not in indicators[strategy_name] or 'stochastic' not in indicators[strategy_name]:
                        continue  # Skip this strategy if it is missing any of its required components

                    # Get Bollinger Band data
                    bollinger = indicators['bollinger'][strategy_name]['upper'] - \
                                indicators['bollinger'][strategy_name]['middle'] - \
                                indicators['bollinger'][strategy_name]['lower']

                    # Get Donchian Bands data
                    donchian = np.roll(indicators[strategy_name]['donchian'], 1)[:-params['bars']]

                    # Vortex crosses above Bollinger band and fast length is greater than slow length
                    vortex = (bollinger > df['close'][i]) & \
                             ((np.abs(df['high'].rolling(strategy_name['fast_length'], min_periods=1).max()) - \
                                      df['open'][i]).shift(-strategy_name['slow_length']) < 0)

                    # Donchian breakout should compare close vs previous band: prev_upper = np.roll(indicators['donchian']['upper'], 1)
                    indicators['donchian']['lower'] = donchian - df[strategy_name['donchian']]
                    indicators['donchian']['upper'] = donchian + df[strategy_name['donchian']]

                    # Supertrend data
                    supertrend = indicators[strategy_name]['supertrend'][0:250]  # Assume we take the first 250 bars for simplicity

                    # Stochastic data
                    indicators['stochastic']['stoch_k'] = df['high'].rolling(14, min_periods=1).apply(np.percentile, q=89)
                    indicators['stochastic']['stoch_d'] = df['low'].rolling(14, min_periods=1).apply(np.percentile, q=50)

                    # ATR-based SL/TP calculation for long and short positions (replace with your own logic if needed)
                    indicators['bollinger']['lower'], indicators['bollinger']['upper'] = df['close'].rolling('20 bar').apply(lambda x: np.mean([np.min(x[:len(x)-1]),  # Calculate the rolling average over the last 3 days only for close data
                                                                              np.max((x[-4:-8:-1] + x[-(len(x)+1):-9]))],
                                                                        ).tolist())
                    bb_stop_long = df['open'][i].expanding().apply(lambda x: max([bb_lower,  # Use the lower Bollinger Band for SL if ATR is below zero
                                                                              (np.mean((x[-4:-8:-1] + x[-(len(x)+1):-9])) - np.std((x[-4:-8:-1]) * 2))]),  # Use upper BB for TP when above it on a daily closing basis
                        ).tolist()
                    bb_tp_long = df['close'][i].expanding().apply(lambda x: min([bb_upper,  # Use the upper Bollinger Band for TP if ATR is above zero
                                                                              (np.mean((x[-4:-8:-1] + x[-(len(x)+1):-9])) - np.std((x[-4:-8:-1]) * 2))]),  # Use lower BB for SL when below it on a daily closing basis
                        ).tolist()
                    bb_stop_short = df['open'][i].expanding().apply(lambda x: max([bb_lower,  # Use the lower Bollinger Band for SL if ATR is above zero
                                                                              (np.mean((x[-4:-8:-1] + x[-(len(x)+1):-9])) - np.std((x[-4:-8:-1]) * 2))]),  # Use upper BB for TP when below it on a daily closing basis
                        ).tolist()
                    bb_tp_short = df['close'][i].expanding().apply(lambda x: min([bb_upper,  # Use the upper Bollinger Band for TP if ATR is above zero
                                                                              (np.mean((x[-4:-8:-1] + x[-(len(x)+1):-9])) - np.std((x[-4:-8:-1]) * 2))]),  # Use lower BB for SL when below it on a daily closing basis
                        ).tolist()

                    # Calculate the signals using your strategy logic here
        signals.iloc[:warmup] = 0.0
        return signals
