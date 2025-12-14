"""
Backtest Core - Commodity Channel Index (CCI)
=============================================

Le CCI mesure l'écart du prix par rapport à sa moyenne, normalisé.
Développé par Donald Lambert (1980).

Interprétation:
- CCI > +100 : Surachat
- CCI < -100 : Survente
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CCISettings:
    """Paramètres CCI."""
    period: int = 20
    factor: float = 0.015  # Facteur de normalisation standard


def cci(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 20,
    factor: float = 0.015,
) -> np.ndarray:
    """
    Calcule le Commodity Channel Index (CCI).

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        period: Période de calcul (défaut: 20)
        factor: Facteur de normalisation (défaut: 0.015)

    Returns:
        Valeurs CCI (généralement entre -200 et +200)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Typical Price
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)

    # SMA du Typical Price
    tp_sma = tp_series.rolling(window=period).mean().values

    # Deviation
    deviation = typical_price - tp_sma

    # Mean Absolute Deviation
    def rolling_mad(arr, window):
        result = np.empty_like(arr)
        result[:window - 1] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(np.abs(arr[i - window + 1:i + 1] - tp_sma[i]))
        return result

    mean_dev = rolling_mad(typical_price, period)

    # CCI
    cci_values = np.where(mean_dev != 0, deviation / (factor * mean_dev), 0.0)

    return cci_values


__all__ = ["cci", "CCISettings"]
