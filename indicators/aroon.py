"""
Backtest Core - Aroon Indicator
===============================

Aroon mesure le temps écoulé depuis le plus haut/plus bas sur une période.
Développé par Tushar Chande (1995).

Interprétation:
- Aroon Up > 70 et Aroon Down < 30 : Forte tendance haussière
- Aroon Down > 70 et Aroon Up < 30 : Forte tendance baissière
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class AroonSettings:
    """Paramètres Aroon."""
    period: int = 14


def aroon(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule Aroon Up/Down.

    Args:
        high: Prix hauts
        low: Prix bas
        period: Période (défaut: 14)

    Returns:
        Tuple (aroon_up, aroon_down) en pourcentage (0-100)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values

    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)

    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]

        bars_since_high = period - 1 - int(np.argmax(window_high))
        bars_since_low = period - 1 - int(np.argmin(window_low))

        aroon_up[i] = 100.0 * (period - bars_since_high) / period
        aroon_down[i] = 100.0 * (period - bars_since_low) / period

    return aroon_up, aroon_down


__all__ = ["aroon", "AroonSettings"]
