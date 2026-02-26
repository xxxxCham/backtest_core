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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            # implement explicit LONG / SHORT / FLAT logic
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            warmup = int(params.get("warmup", 50))

            # use ATR-based risk management for SL/TP levels
            atr = indicators['atr']
            close = df["close"].values
            entry_mask = (signals == 1.0)  

            self.bb_stop_long = []
            self.bb_tp_long = []

            # use ATR-based risk management for SL/TP levels
            if params.get("leverage", 3) > 2:
                leverage = float(params.get("leverage"))

                # set initial stop loss and take profit level
                self.bb_stop_long.append((close[entry_mask] - atr).mean())
                self.bb_tp_long.append(((close[entry_mask]+atr*2) if close[entry_mask]<(close[entry_mask]+atr*2) else (close[entry_mask]-atr)).mean())
            # use percent from the k_sl param for SL/TP levels
            else:
                stop_pct = float(params.get("stop_percent", 0)) / 100

                self.bb_stop_long.append((close[entry_mask] - atr).mean()*stop_pct)
                self.bb_tp_long.append(((close[entry_mask]+atr *2 ) if close[entry_mask]<(close[entry_mask]+atr * 2) else (close[entry_mask]-atr)).mean())

            # implement drawdown limit for risk management
        signals.iloc[:warmup] = 0.0
        return signals
