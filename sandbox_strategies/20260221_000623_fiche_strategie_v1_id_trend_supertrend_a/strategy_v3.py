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
        def generate_signals(indicators, entry):
            # Entrée long si momentum haussier confirmé et risque contrôlé. 
            if 'momentum' in indicators and 'rsi' in indicators['momentum'] and all([np.abs(indicators['momentum']) > 0]):
                signals = np.where((df[entry] < df[signal_cols]['bb_stop_long']) & (df[entry] >= df[signal_cols]['bb_tp_long']), -1.0, signals)
            # Entrée short si momentum baissier confirmé et risque contrôlé. 
            elif 'momentum' in indicators and 'rsi' in indicators['momentum'] and all([np.abs(indicators['momentum']) < 0]):
                signals = np.where((df[entry] > df[signal_cols]['bb_stop_short']) & (df[entry] <= df[signal_cols]['bb_tp_short']), -1.0, signals)
            # Supertrend up: long signal when price above upper band; short signal when price below lower band
            elif 'supertrend' in indicators and 'direction' in indicators['supertrend'] and all([indicators['supertrend']['direction'] == 'up']):
                signals = np.where((df[entry] > df[signal_cols]['bb_stop_long']) & (df[entry] <= df[signal_cols]['bb_tp_long']), 1.0, signals)
            # Supertrend down: short signal when price above upper band; long signal when price below lower band
            elif 'supertrend' in indicators and 'direction' in indicators['supertrend'] and all([indicators['supertrend']['direction'] == 'down']):
                signals = np.where((df[entry] < df[signal_cols]['bb_stop_short']) & (df[entry] >= df[signal_cols]['bb_tp_short']), -1.0, signals)
            # Momentum: buy when above 21 EMA; sell when below 21 EMA
            elif 'momentum' in indicators and 'ema' in indicators['momentum'] and all([np.abs(indicators['momentum']) > np.abs(df[signal_cols]['ema_21'])]):
                signals = np.where((df[entry] < df[signal_cols]['ema_21']) & (df[entry] >= df[signal_cols]['eod_rsi']), 1.0, signals)
            # RSI: buy when above 30; sell when below 70
            elif 'rsi' in indicators and all([np.abs(indicators['rsi']) > np.abs(df[signal_cols]['rsi_overbought'])]) and all([np.abs(indicators['rsi']) < np.abs(df[signal_cols]['rsi_oversold'])]):
                signals = np.where((df[entry] >= df[signal_cols]['rsi_overbought']) & (df[entry] <= df[signal_cols]['rsi_oversold']), 1.0, signals)
            # ATR: buy when above ATR + StdDev; sell when below ATR - StdDev
            elif 'atr' in indicators and all([np.abs(indicators['atr']) > np.abs(df[signal_cols]['atr_buy'])]):
                signals = np.where((df[entry] >= df[signal_cols]['atr_sell']) & (df[entry] <= df[signal_cols]['atr_buy']), 1.0, signals)
            else:
                # No signal if no momentum or RSI indicators are present
                signals = np.where((np.abs(indicators['momentum']) == 0), 0, signals)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
