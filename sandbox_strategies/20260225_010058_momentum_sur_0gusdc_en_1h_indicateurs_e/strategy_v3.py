from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic']

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
        def generate_signals(indicators, default_params):
            # Define default parameters
            df = pd.DataFrame()  # Assuming 'df' is a DataFrame with open/high/low/close prices
            leverage = 1

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i, indicator in enumerate(indicators):
                if 'ema' == indicator:
                    ema_length = default_params['ema']['length']

                    # Calculate EMA values
                    df[f"EMA_{i+1}"] = calculate_eMA(df.close, ema_length)

                elif 'stochastic' == indicator:
                    stoch_k_length = default_params['stochastic']['k_period']

                    # Calculate Stochastic K values
                    df[f"STOCH_{i+1}"] = calculate_stochK(df.close, stoch_k_length)

                elif 'bollinger' == indicator:
                    upper, middle, lower = default_params['bollinger']

                    # Calculate Bollinger Bands values (upper, middle, and lower bands)
                    df[f"BB_{i+1}"] = calculate_bollinger(df.close, upper, middle, lower)

                elif 'donchian' == indicator:
                    donchian_length = default_params['donchian']['length']

                    # Calculate Donchian Bands values (previous and current bands)
                    df[f"DB_{i+1}"] = calculate_donchian(df.close, donchian_length)

                elif 'adx' == indicator:
                    adx_length = default_params['adx']['length']

                    # Calculate ADX values (adx, indicators['adx']['plus_di'], indicators['adx']['minus_di'])
                    df[f"ADX_{i+1}"] = calculate_adx(df.high, df.low, adx_length)

                elif 'supertrend' == indicator:
                    supertrend_length = default_params['supertrend']['length']

                    # Calculate Supertrend values (supertrend, direction)
                    df[f"SUPERTREND_{i+1}"] = calculate_supertrend(df.close, supertrend_length, indicator=indicator[-1])

                elif 'stochrsi' == indicator:
                    stochrsi_length = default_params['stochastic']['k_period'] + \
                                      default_params['rsi']['n_periods']  # Use K and N periods for RSI

                    # Calculate StochRSI values (stochK, RSI)
                    df[f"STOCHRSI_{i+1}"] = calculate_stochastic(df.close, stochrsi_length)

                elif 'ema' == indicator:
                    ema_length = default_params['ema']['length']

                    # Calculate EMA values
                    df[f"EMA_{i+1}"] = calculate_eMA(df.close, ema_length)

                elif 'atr' == indicator:
                    atr_n = default_params['atr']['n_periods']  # Use N periods for ATR

                    # Calculate ATR values (upper and lower bands)
                    df[f"ATR_{i+1}"] = calculate_atr(df.high, df.low, atr_n)

                elif 'rsi' == indicator:
                    rsi_length = default_params['rsi']['n_periods']  # Use N periods for RSI

                    # Calculate RSI values (RSI)
                    df[f"RSI_{i+1}"] = calculate_stochastic(df.close, rsi_length)

                elif 'ema' == indicator:
                    ema_length = default_params['ema']['length']

                    # Calculate EMA values
        signals.iloc[:warmup] = 0.0
        return signals
