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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Indicateur RSI
        rsi_values = indicators["rsi"]
        overbought = rsi_values > 70
        oversold = rsi_values < 30

        # Momentum haussier
        haussier_momentum = (rsi_values > 50) & (overbought)

        # Indicateur ATR
        atr_values = indicators["atr"]

        # Calcul du risque relatif
        risk_ratio = atr_values / df['close']

        # Seuils de risque contrôlé
        controlled_risk = risk_ratio < 0.005

        # Combinaison pour long
        long_entry = haussier_momentum & controlled_risk

        # Signaux baissiers
        baissier_momentum = (rsi_values < 50) & (oversold)
        baissier_entry = baissier_momentum & controlled_risk

        # Application des signaux
        signals[long_entry] = 1.0
        signals[baissier_entry] = -1.0
        return signals
