"""
Backtest Core - Bollinger Bands + ATR Strategy
==============================================

Stratégie de mean-reversion basée sur les Bandes de Bollinger
avec filtre de volatilité ATR.

Logique:
- LONG: Prix touche/dépasse la bande inférieure (oversold)
- SHORT: Prix touche/dépasse la bande supérieure (overbought)
- Filtre ATR: Ne trade que si volatilité > seuil (évite marchés plats)
- Stop-loss: Basé sur multiple de l'ATR
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import SAFE_RANGES_PRESET, ParameterSpec, Preset

from .base import StrategyBase, register_strategy


@register_strategy("bollinger_atr")
class BollingerATRStrategy(StrategyBase):
    """
    Stratégie Bollinger Bands + ATR (Mean Reversion).

    Cette stratégie est le "moteur de Baptiste" - la stratégie de référence
    utilisée pour valider le nouveau moteur de backtest.

    Paramètres:
        entry_z: Seuil d'entrée en Z-score (défaut: 2.0)
        k_sl: Multiplicateur ATR pour stop-loss (défaut: 1.5)
        atr_percentile: Percentile ATR pour filtre volatilité (défaut: 30)
        leverage: Levier de trading (défaut: 3)

    Signaux:
        +1 (Long): close <= lower_band ET ATR > seuil
        -1 (Short): close >= upper_band ET ATR > seuil
        0: Sinon
    """

    def __init__(self):
        super().__init__(name="BollingerATR")

    @property
    def required_indicators(self) -> List[str]:
        """Indicateurs requis: Bollinger Bands et ATR."""
        return ["bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        """Paramètres par défaut de la stratégie."""
        return {
            # Bollinger
            "bb_period": 20,
            "bb_std": 2.0,
            # ATR
            "atr_period": 14,
            "atr_percentile": 30,  # Filtre: ATR doit être > ce percentile
            # Trading
            "entry_z": 2.0,  # Z-score pour entrée (touch band = 2 std)
            "k_sl": 1.5,     # Stop loss = k_sl * ATR
            "leverage": 3,
            # Frais (pour référence)
            "fees_bps": 10,
            "slippage_bps": 5
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Spécifications détaillées des paramètres pour UI/optimisation."""
        return {
            "bb_period": ParameterSpec(
                name="bb_period",
                min_val=10, max_val=50, default=20,
                param_type="int",
                description="Période des Bandes de Bollinger"
            ),
            "bb_std": ParameterSpec(
                name="bb_std",
                min_val=1.5, max_val=3.0, default=2.0,
                param_type="float",
                description="Écarts-types pour les bandes"
            ),
            "entry_z": ParameterSpec(
                name="entry_z",
                min_val=1.0, max_val=3.0, default=2.0,
                param_type="float",
                description="Seuil z-score pour entree"
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=7, max_val=21, default=14,
                param_type="int",
                description="Période de l'ATR"
            ),
            "atr_percentile": ParameterSpec(
                name="atr_percentile",
                min_val=10, max_val=60, default=30,
                param_type="int",
                description="Percentile volatilite minimum (ATR)"
            ),
            "k_sl": ParameterSpec(
                name="k_sl",
                min_val=1.0, max_val=3.0, default=1.5,
                param_type="float",
                description="Multiplicateur ATR pour stop-loss"
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1, max_val=10, default=3,
                param_type="int",
                description="Levier de trading"
            ),
        }

    def get_preset(self) -> Optional[Preset]:
        """Retourne le preset Safe Ranges associé."""
        return SAFE_RANGES_PRESET

    def get_indicator_params(
        self,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mappe les parametres de la strategie vers les indicateurs."""
        if indicator_name == "bollinger":
            return {
                "period": int(params.get("bb_period", 20)),
                "std_dev": float(params.get("bb_std", 2.0)),
            }
        if indicator_name == "atr":
            return {"period": int(params.get("atr_period", 14))}
        return super().get_indicator_params(indicator_name, params)

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Génère les signaux de trading Bollinger + ATR.

        Args:
            df: DataFrame OHLCV
            indicators: {"bollinger": (upper, middle, lower), "atr": atr_array}
            params: Paramètres de stratégie

        Returns:
            pd.Series de signaux (+1, -1, 0)
        """
        # Initialiser signaux à 0 (hold)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64, name="signals")

        # Extraire Bollinger Bands
        if "bollinger" not in indicators or indicators["bollinger"] is None:
            return signals

        bb_result = indicators["bollinger"]
        if not isinstance(bb_result, tuple) or len(bb_result) < 3:
            return signals

        upper, middle, lower = bb_result[:3]

        # Convertir en Series si nécessaire
        if not isinstance(upper, pd.Series):
            upper = pd.Series(np.asarray(upper), index=df.index)
        if not isinstance(lower, pd.Series):
            lower = pd.Series(np.asarray(lower), index=df.index)
        if not isinstance(middle, pd.Series):
            middle = pd.Series(np.asarray(middle), index=df.index)

        close = df["close"]

        # === Signaux Bollinger (Mean Reversion) ===
        bb_std = float(params.get("bb_std", 2.0))
        if bb_std <= 0:
            bb_std = 2.0
        entry_z = float(params.get("entry_z", bb_std))
        sigma = (upper - middle) / bb_std
        entry_upper = middle + (sigma * entry_z)
        entry_lower = middle - (sigma * entry_z)

        # Long: prix <= seuil bas (oversold)
        long_condition = close <= entry_lower

        # Short: prix >= seuil haut (overbought)
        short_condition = close >= entry_upper

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        # === Filtre ATR (volatilité) ===
        if "atr" in indicators and indicators["atr"] is not None:
            atr_values = indicators["atr"]

            if not isinstance(atr_values, pd.Series):
                atr_values = pd.Series(np.asarray(atr_values), index=df.index)

            # Seuil de volatilité (percentile)
            atr_percentile = params.get("atr_percentile", 30)
            atr_threshold = atr_values.quantile(atr_percentile / 100.0)

            # Filtre: annuler signaux si volatilité trop faible
            low_volatility = atr_values <= atr_threshold
            signals[low_volatility] = 0.0

        # === Éviter signaux consécutifs identiques ===
        # Ne garder que les changements de signal
        signals_diff = signals.diff()
        # Premier signal conservé, ensuite seulement les changements
        signals_clean = signals.copy()
        signals_clean[1:] = np.where(signals_diff[1:] != 0, signals[1:], 0)

        return signals_clean

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        atr_value: float,
        params: Dict[str, Any]
    ) -> float:
        """
        Calcule la taille de position basée sur le risque ATR.

        Position sizing: risk_amount / (k_sl * ATR)

        Args:
            capital: Capital disponible
            entry_price: Prix d'entrée
            atr_value: Valeur ATR actuelle
            params: Paramètres de stratégie

        Returns:
            Quantité à trader
        """
        leverage = params.get("leverage", 3)
        k_sl = params.get("k_sl", 1.5)
        risk_pct = params.get("risk_pct", 0.02)  # 2% du capital par trade

        # Risque en valeur absolue
        risk_amount = capital * risk_pct

        # Distance du stop en prix
        stop_distance = k_sl * atr_value

        # Quantité basée sur le risque
        if stop_distance > 0:
            quantity = risk_amount / stop_distance
        else:
            quantity = 0

        # Limiter par le leverage disponible
        max_quantity = (capital * leverage) / entry_price

        return min(quantity, max_quantity)

    def get_stop_loss(
        self,
        entry_price: float,
        atr_value: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calcule le niveau de stop-loss.

        Args:
            entry_price: Prix d'entrée
            atr_value: Valeur ATR
            side: "long" ou "short"
            params: Paramètres

        Returns:
            Prix du stop-loss
        """
        k_sl = params.get("k_sl", 1.5)
        distance = k_sl * atr_value

        if side == "long":
            return entry_price - distance
        else:
            return entry_price + distance


__all__ = ["BollingerATRStrategy"]
