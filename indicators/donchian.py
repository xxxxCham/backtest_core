"""
Backtest Core - Donchian Channel
================================

Canal formé par les plus hauts et plus bas sur une période.
Utilisé pour breakout trading.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class DonchianSettings:
    """Paramètres Donchian Channel."""
    period: int = 20


def donchian_channel(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule Donchian Channel.

    Args:
        high: Prix hauts
        low: Prix bas
        period: Période du canal (défaut: 20)

    Returns:
        Tuple (upper, middle, lower)
    """
    if isinstance(high, pd.Series):
        high_series = high
    else:
        high_series = pd.Series(high)

    if isinstance(low, pd.Series):
        low_series = low
    else:
        low_series = pd.Series(low)

    upper = high_series.rolling(window=period).max().values
    lower = low_series.rolling(window=period).min().values
    middle = (upper + lower) / 2.0

    return upper, middle, lower


__all__ = ["donchian_channel", "DonchianSettings"]
