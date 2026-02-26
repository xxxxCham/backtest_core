from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='AgressiveCryptoTrader')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATR_PipSize': 10,
         'MACD_Fastperiod': 12,
         'MACD_Signalperiod': 9,
         'MACD_Slowperiod': 26,
         'RSI_Period': 14,
         'leverage': 2,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'RSI_Period': ParameterSpec(
                name='RSI_Period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stoploss_atr_multiple': ParameterSpec(
                name='stoploss_atr_multiple',
                min_val=0.5,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(self, bars):
                macd = self.MACD()  # calculate MACD indicator using technical library

                if len(macd) < 10:  
                    return     

                short_window, long_window, signal_window = macd[-3:]  

                crossover_crossed_above = cross_over(short_window, long_window)    # check if MACD is above zero (uptrend)
                crossover_crossed_below = cross_under(short_window, long_window)  # check if MACD is below zero (downtrend)  

                self.info("MACD: %s" % macd[-1])     

                if crossover_crossed_above and signal_window[0] > 0:  
                    position = self.getPosition()  # get the current position

                    if np.isnan(position):      
                        buyPrice = self.broker.getBestAsk(self.ES)      # buy at 1.5x the ask price with a stop loss

                        self.buy(size=1, limit=buyPrice * (1+0.5))     # buy at 1.5x the ask price with a stop loss

                    else:  
                        sellPrice = self.broker.getBestBid(self.ES)    # buy at 30% of the bid price with a take profit

                        self.sellToMarket(size=position, limit=sellPrice * (1-0.2))     # sell at 30% of the ask price with a take profit

                elif crossover_crossed_below and signal_window[0] < 0:  
                    position = self.getPosition()  

                    if np.isnan(position):      
                        sellPrice = self.broker.getBestBid(self.ES)     # buy at 30% of the bid price with a take profit

                        self.buyToMarket(size=position, limit=sellPrice * (1-0.2))    # sell at 70% of the ask price with a take profit
        signals.iloc[:warmup] = 0.0
        return signals
