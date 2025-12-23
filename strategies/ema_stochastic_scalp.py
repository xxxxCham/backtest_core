"""
Backtest Core - EMA Stochastic Scalp Strategy
==============================================

Stratégie de scalping crypto combinant:
- EMA 50/100 pour la direction de tendance
- Stochastic pour le timing d'entrée (survente/surachat)

Win rate attendu: 65-80% selon documentation
Optimale sur BTC/USDT, ETH/USDT avec timeframe 1m-5m.

Règles d'Entrée LONG:
1. EMA 50 > EMA 100 (tendance haussière)
2. Stochastic %K < oversold (survente) ou croise %D vers le haut

Règles d'Entrée SHORT:
1. EMA 50 < EMA 100 (tendance baissière)
2. Stochastic %K > overbought (surachat) ou croise %D vers le bas
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, StrategyResult, register_strategy
from utils.parameters import ParameterSpec


@register_strategy("ema_stochastic_scalp")
class EMAStochasticScalpStrategy(StrategyBase):
    """
    Stratégie EMA + Stochastic pour scalping.

    Signaux:
        - LONG (+1): EMA50 > EMA100 ET Stochastic en survente/croise haut
        - SHORT (-1): EMA50 < EMA100 ET Stochastic en surachat/croise bas

    Paramètres:
        - fast_ema: Période EMA rapide (défaut: 50)
        - slow_ema: Période EMA lente (défaut: 100)
        - stoch_k: Période Stochastic %K (défaut: 14)
        - stoch_d: Période Stochastic %D (défaut: 3)
        - stoch_oversold: Seuil survente (défaut: 20)
        - stoch_overbought: Seuil surachat (défaut: 80)
        - leverage: Multiplicateur de position (défaut: 10)
    """

    def __init__(self, name: str = "EMAStochScalp"):
        super().__init__(name)

    @property
    def required_indicators(self) -> List[str]:
        """Indicateurs requis par la stratégie."""
        return ["stochastic"]

    @property
    def default_params(self) -> Dict[str, Any]:
        """Paramètres par défaut."""
        return {
            "fast_ema": 50,
            "slow_ema": 100,
            "stoch_k": 14,
            "stoch_d": 3,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "leverage": 10,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Spécifications des paramètres pour l'UI et l'optimisation."""
        return {
            "fast_ema": ParameterSpec(
                name="fast_ema",
                min_val=20,
                max_val=100,
                default=50,
                param_type="int",
                description="Période EMA rapide"
            ),
            "slow_ema": ParameterSpec(
                name="slow_ema",
                min_val=50,
                max_val=200,
                default=100,
                param_type="int",
                description="Période EMA lente"
            ),
            "stoch_k": ParameterSpec(
                name="stoch_k",
                min_val=5,
                max_val=21,
                default=14,
                param_type="int",
                description="Période Stochastic %K"
            ),
            "stoch_d": ParameterSpec(
                name="stoch_d",
                min_val=2,
                max_val=9,
                default=3,
                param_type="int",
                description="Période Stochastic %D"
            ),
            "stoch_oversold": ParameterSpec(
                name="stoch_oversold",
                min_val=10,
                max_val=30,
                default=20,
                param_type="int",
                description="Seuil survente Stochastic"
            ),
            "stoch_overbought": ParameterSpec(
                name="stoch_overbought",
                min_val=70,
                max_val=90,
                default=80,
                param_type="int",
                description="Seuil surachat Stochastic"
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=20,
                default=10,
                param_type="int",
                description="Levier de trading (scalping)"
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Génère les signaux de trading combinant EMA et Stochastic.

        Args:
            df: DataFrame OHLCV
            indicators: Dictionnaire contenant 'stochastic' (tuple k, d)
            params: Paramètres de la stratégie

        Returns:
            Series de signaux (-1, 0, +1)
        """
        signals = pd.Series(0.0, index=df.index)

        fast_ema_period = int(params.get("fast_ema", 50))
        slow_ema_period = int(params.get("slow_ema", 100))
        oversold = params.get("stoch_oversold", 20)
        overbought = params.get("stoch_overbought", 80)

        close = df["close"]

        # Calculer les EMAs
        ema_fast = close.ewm(span=fast_ema_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_ema_period, adjust=False).mean()

        # Tendance
        uptrend = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow

        # Récupérer le Stochastic
        if "stochastic" in indicators and indicators["stochastic"] is not None:
            stoch_data = indicators["stochastic"]
            if isinstance(stoch_data, tuple) and len(stoch_data) >= 2:
                stoch_k = stoch_data[0]
                stoch_d = stoch_data[1]
            else:
                return signals
        else:
            # Calcul interne si non fourni
            stoch_k, stoch_d = self._calculate_stochastic(
                df, int(params.get("stoch_k", 14)), int(params.get("stoch_d", 3))
            )

        # Convertir en Series
        if isinstance(stoch_k, np.ndarray):
            stoch_k = pd.Series(stoch_k, index=df.index)
        if isinstance(stoch_d, np.ndarray):
            stoch_d = pd.Series(stoch_d, index=df.index)

        stoch_k_prev = stoch_k.shift(1)
        stoch_d_prev = stoch_d.shift(1)

        # Signal LONG: uptrend + stochastic en survente ou croise vers le haut
        stoch_crosses_up = (stoch_k_prev < stoch_d_prev) & (stoch_k >= stoch_d)
        stoch_oversold_cond = stoch_k < oversold

        long_signal = uptrend & (stoch_crosses_up | stoch_oversold_cond)

        # Signal SHORT: downtrend + stochastic en surachat ou croise vers le bas
        stoch_crosses_down = (stoch_k_prev > stoch_d_prev) & (stoch_k <= stoch_d)
        stoch_overbought_cond = stoch_k > overbought

        short_signal = downtrend & (stoch_crosses_down | stoch_overbought_cond)

        signals[long_signal] = 1.0
        signals[short_signal] = -1.0

        return signals

    def get_indicator_params(
        self,
        indicator_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mappe les parametres de la strategie vers les indicateurs."""
        if indicator_name == "stochastic":
            return {
                "k_period": int(params.get("stoch_k", 14)),
                "d_period": int(params.get("stoch_d", 3)),
                "smooth_k": 3,
            }
        return super().get_indicator_params(indicator_name, params)

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int, d_period: int
    ) -> tuple:
        """Calcul interne du Stochastic si non fourni."""
        k_period = int(k_period)
        d_period = int(d_period)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Highest high et lowest low sur k_period
        highest_high = high.rolling(window=k_period).max()
        lowest_low = low.rolling(window=k_period).min()

        # %K
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # %D = SMA de %K
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k.values, stoch_d.values

    def describe(self) -> str:
        """Description de la stratégie."""
        return (
            "EMA Stochastic Scalp Strategy: Stratégie de scalping combinant "
            "la tendance (EMA 50/100) et le timing (Stochastic). "
            "Achat en survente + uptrend, vente en surachat + downtrend."
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
                "fast_ema": params.get("fast_ema", 50),
                "slow_ema": params.get("slow_ema", 100),
            }
        )

        return self._last_result
