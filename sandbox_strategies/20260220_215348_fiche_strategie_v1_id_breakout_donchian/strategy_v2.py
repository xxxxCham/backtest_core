from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

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
        def generate_signals(indicators):
            # Import necessary libraries
            import numpy as np

            # Initialize empty lists to store signals for LONG, SHORT, and ADJUSTED signals
            long_signals = []
            short_signals = []
            adjusted_signals = []

            # Loop through each bar in the DataFrame df
            for i, row in df.iterrows():
                # Define Bollinger Bands values
                middle_band = np.mean(row['close'].rolling(window=20).std())
                lower_band = np.mean(row['close'].rolling(window=20).std() * 2) - row['atr']
                upper_band = np.mean(row['close'].rolling(window=20).std()) + row['atr']

                # Define Donchian Channels values (no need for previous band, as ATR is used instead)
                indicators['donchian']['upper'] = np.roll(donchian_highs, 1)[:-1]
                donchian_lows = np.roll(donchian_lows, 1)[:-1]

                # Define SuperTrend values (direction is 'buy'/'sell')
                super_trend_signal = row['super_trend'] * np.sign(row['close'].diff())

                # Calculate Stochastic K and D values
                fast_k, slow_d = stoch_k_d(row['close'], row['high'], row['low'])

                # Initialize EMA14 and EMA20 values (no need for 'ema' keys)
                ema14, ema20 = simple_moving_avg([row['close']], window=14), simple_moving_avg([row['close']], window=20)

                # Calculate ATR-based SL/TP values (no need to convert them into boolean masks)
                bb_tp, bb_sl = calculate_take_profit(row, 'close', row['atr'], 1.0), calculate_stop_loss(row, 'close', row['atr'])

                # Define signals for each indicator and adjust ATR values accordingly
                if np.mean(row['close'].rolling(window=20).std()) > upper_band:
                    signal = 1.0
                elif np.mean(row['close'].rolling(window=20).std() * 2) - row['atr'] < lower_band:
                    signal = -1.0
                else:
                    signal = 0.0

                # Append signals to the respective lists
                long_signals.append(signal if row['close'].diff().cumsum()[i] > 0 and not np.isnan(row['close']).all() else None)
                short_signals.append(signal if row['close'].diff().cumsum()[i] < 0 and not np.isnan(row['close']).all() else None)

                # Calculate ADX values (no need to convert it into 'plus_di' or 'minus_di')
                adx = calculate_adx([np.where(donchian_highs[:-1] > donchian_lows[0])[0], np.where((donchian_highs[-1:] < donchian_lows)[0]), np.where((row['close'] == row['close'].max()) & (row['open'] != row['open'].min()))]).sum() / df.shape[0]

                # Adjust ADX values to compare with 'plus_di' and 'minus_di' inputs (no need for direct comparison)
                if adx > 25:
                    signal = -1.0
                elif adx < -(25):
                    signal = 1.0

                # Calculate RSI values using simple moving average crossover strategy (no need to compare with previus band or calculate 'rsi' key)
                rsi_val, _ = relative_strength_index([row['close']], window=14)[0]

                # Check if close price is above the simple moving average of the last 20 candles (no need for direct comparison with previous band or 'stoch_k' and 'stoch_d' keys)
                signal = 1.0 if row['close'] > ema20[-1] else -1.0

                # Calculate TA signals based on input indicators, adjust ATR SL/TP values accordingly (no need for 'SL', 'TP', or direct comparison with boolean masks)
                if not np.isnan(row['atr']):
                    signal = 1.0 if row['close'].diff().cumsum()[i] > bb_tp[i].stop else -1.0

                # Append signals to the respective lists
                adjusted_signals.append(signal)

            # Return a list of generated signals with None values replaced by 0.0 for missing data (no need for True/False or specific assignment syntax)
            return [val if val is not None else 0.0 for sublist in zip(long_signals, short_signals, adjusted_signals) for val in sublist]
        signals.iloc[:warmup] = 0.0
        return signals
