from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

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
        # This function generates buy and sell signals based on the available indicators
        def generate_signals(indicators):
            df = pd.DataFrame()   # Initialize an empty DataFrame to store data for each bar (entry, exit)

            # Loop over every row in the DataFrame
            for i, row in enumerate(df.iterrows()):
                bars = list(row[1].values())  # Get all values of this row as a Python list

                if len(bars) < 2:   # Skip rows with less than two data points (not enough to calculate indicators)
                    continue

                close_price, _, _, _, prev = bars[:5]    # Extract relevant data for the current bar
                lower_bollinger, middle_bollinger, upper_bollinger  = indicators['bollinger']['lower'|'middle'|'upper'](close_price)   # Get Bollinger band values

                rsi, indicators['donchian']['upper'], indicators['donchian']['lower'], _ = [indicators[i](close_price)[0] for i in ['rsi', 'donchian_band__upper','donchian_band__lower']]   # Get RSI and Donchian band values
                adx, indicators['adx']['plus_di'], indicators['adx']['minus_di']  = indicators['adx']['adx'|'plus_di'|'minus_di'](close_price)    # Get ADX values

                indicators['stochastic']['stoch_k'], indicators['stochastic']['stoch_d']   = [indicators[i](close_price)[0] for i in ['stochastic','stoch_k', 'stoch_d']]  # Get Stochastic K&D values
                supertrend    = indicators['supertrend']['supertrend'][()]     # Assume Supertrend is a constant value (no need to calculate)

                atr   = indicators['atr'](close_price)[0]      # Calculate ATR for the current bar using previous close price
                prev_upper = np.roll(indicators['donchian']['upper'], 1)[:]            # Roll Donchian upper band and reset it to a new list (for next iteration)

                if not any([bollinger >= bollinger[0] for bollinger in [lower_bollinger, middle_bollinger, upper_bollinger]]):   # Check if price is outside Bollinger bands
                    continue

                if rsi < 40:    # Long signal (price is oversold)
                    signals.at[i] = -1.0     # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if close_price > upper_bollinger and abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check the Long Intraday Breakout rule (Close price is above Bollinger Upper Band AND previous Donchian Upper Band was broken)
                    signals.at[i] = 1.0    # Set the exit signal to be 1.0 for long positions (buying to sell at a profit)

                if close_price < lower_bollinger and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is below Bollinger Lower Band AND previous Donchian Lower Band was broken)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if adx > 25:   # Check Long Momentum rule (ADX is above 25)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for long positions (-1.0 means short position, 1.0 means sell signal)

                if indicators['stochastic']['stoch_k'] > indicators['donchian']['upper'] and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check Stochastic K%D rule (K is above D AND close price is below previous Donchian Upper Band)
                    signals.at[i] = 1.0    # Set the exit signal to be 1.0 for long positions (buying to sell at a profit)

                if indicators['stochastic']['stoch_d'] > indicators['donchian']['lower'] and abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check Stochastic K%D rule (K is above D AND close price is below previous Donchian Upper Band)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if supertrend > 0 and close_price < indicators['donchian']['upper']:   # Check Supertrend rule (Supertrend is up AND price below Donchian Upper Band)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for long positions (-1.0 means short position, 1.0 means sell signal)

                if close_price < lower_bollinger and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is below Bollinger Lower Band AND previous Donchian Lower Band was broken)
                    signals.at[i] = 1.0    # Set the exit signal to be 1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if atr < close_price and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check ATR rule (ATR is below Close price AND close price is above Donchian Lower Band)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for long positions (-1.0 means short position, 1.0 means sell signal)

                if atr > close_price and abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check ATR rule (ATR is above Close price AND close price is below Donchian Upper Band)
                    signals.at[i] = 1.0    # Set the exit signal to be 1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check Donchian Breakout rule (Close price is above or below previous Donchian Upper/Lower Bands)
                    signals.at[i] = 1.0    # Set the exit signal to be -1.0 for long positions (-1.0 means short position, 1.0 means sell signal)

                if close_price < indicators['donchian']['lower'] and abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is below Bollinger Lower Band AND previous Donchian Upper Band was broken)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if close_price > indicators['donchian']['upper'] and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is above Bollinger Upper Band AND previous Donchian Lower Band was broken)
                    signals.at[i] = -1.0    # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if close_price > indicators['donchian']['upper'] and abs(indicators['donchian']['lower']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is above Bollinger Upper Band AND previous Donchian Lower Band was broken)
                    signals.at[i] = 1.0    # Set the exit signal to be -1.0 for short positions (-1.0 means long position, 1.0 means sell signal)

                if close_price < indicators['donchian']['lower'] and abs(indicators['donchian']['upper']-close_price)<abs(prev_upper-close_price):   # Check the Short Intraday Breakout rule (Close price is below Bollinger Lower Band AND previous Donchian Lower Band was broken)
                    signals.at[i] = -1.0    # Set the exit signal to be
        signals.iloc[:warmup] = 0.0
        return signals
