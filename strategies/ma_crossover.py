"""
Backtest Core - MA Crossover Strategy
=====================================

Stratégie classique de croisement de moyennes mobiles simples (SMA).
Signal d'achat quand SMA rapide croise SMA lente vers le haut.
Signal de vente quand SMA rapide croise SMA lente vers le bas.

Indicateurs requis:
- SMA: Simple Moving Average (calculé internement)
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, StrategyResult, register_strategy
from utils.parameters import ParameterSpec


@register_strategy("ma_crossover")
class MACrossoverStrategy(StrategyBase):
    """
    Stratégie de croisement de moyennes mobiles simples (SMA).

    Signaux:
        - LONG (+1): SMA rapide croise SMA lente vers le haut (golden cross)
        - SHORT (-1): SMA rapide croise SMA lente vers le bas (death cross)

    Paramètres:
        - fast_period: Période SMA rapide (défaut: 10)
        - slow_period: Période SMA lente (défaut: 30)
        - leverage: Multiplicateur de position (défaut: 1)
    """

    def __init__(self, name: str = "MACrossover"):
        super().__init__(name)

    @property
    def required_indicators(self) -> List[str]:
        """Aucun indicateur externe requis (calcul interne)."""
        return []

    @property
    def default_params(self) -> Dict[str, Any]:
        """Paramètres par défaut."""
        return {
            "fast_period": 10,
            "slow_period": 30,
            "leverage": 1,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Spécifications des paramètres pour l'UI et l'optimisation."""
        return {
            "fast_period": ParameterSpec(
                name="fast_period",
                min_val=5,
                max_val=50,
                default=10,
                param_type=int,
                description="Période SMA rapide"
            ),
            "slow_period": ParameterSpec(
                name="slow_period",
                min_val=20,
                max_val=200,
                default=30,
                param_type=int,
                description="Période SMA lente"
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=10,
                default=1,
                param_type=int,
                description="Levier de trading"
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Génère les signaux de trading basés sur le croisement SMA.

        Args:
            df: DataFrame OHLCV
            indicators: Dictionnaire d'indicateurs (non utilisé ici)
            params: Paramètres de la stratégie

        Returns:
            Series de signaux (-1, 0, +1)
        """
        signals = pd.Series(0.0, index=df.index)

        fast_period = int(params.get("fast_period", 10))
        slow_period = int(params.get("slow_period", 30))

        # Vérification paramètres
        if fast_period >= slow_period:
            return signals

        close = df["close"]

        # Calculer les SMAs
        sma_fast = close.rolling(window=fast_period, min_periods=fast_period).mean()
        sma_slow = close.rolling(window=slow_period, min_periods=slow_period).mean()

        # Détecter les croisements
        fast_above = sma_fast > sma_slow
        fast_above_shifted = fast_above.shift(1)
        fast_above_prev = fast_above_shifted.where(fast_above_shifted.notna(), False)

        # Golden Cross: SMA rapide passe au-dessus de SMA lente
        golden_cross = fast_above & fast_above_prev.eq(False)

        # Death Cross: SMA rapide passe en-dessous de SMA lente
        death_cross = fast_above.eq(False) & fast_above_prev

        signals[golden_cross] = 1.0
        signals[death_cross] = -1.0

        return signals

    def describe(self) -> str:
        """Description de la stratégie."""
        return (
            "MA Crossover Strategy: Génère des signaux sur les croisements "
            "de moyennes mobiles simples. "
            "Achat sur golden cross, vente sur death cross."
        )

    def run(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any] = None
    ) -> StrategyResult:
        """
        Exécute la stratégie.
        """
        if params is None:
            params = self.default_params

        signals = self.generate_signals(df, indicators, params)

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()

        self._last_result = StrategyResult(
            signals=signals,
            params_used=params,
            metadata={
                "strategy": self.name,
                "total_signals": n_long + n_short,
                "long_signals": int(n_long),
                "short_signals": int(n_short),
                "fast_period": params.get("fast_period", 10),
                "slow_period": params.get("slow_period", 30),
            }
        )

        return self._last_result
