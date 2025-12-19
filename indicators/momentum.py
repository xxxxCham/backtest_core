"""
Backtest Core - Momentum Indicator
==================================

Momentum mesure le taux de changement absolu du prix sur une période.
Simple mais efficace pour détecter accélération/décélération.

Formula: Momentum = Close - Close[n periods ago]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomentumSettings:
    """Paramètres Momentum."""
    period: int = 14


def momentum(
    close: pd.Series | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calcule Momentum.

    Args:
        close: Prix de clôture
        period: Période (défaut: 14)

    Returns:
        Différence de prix sur la période
    """
    if isinstance(close, pd.Series):
        close = close.values

    momentum_values = close - np.roll(close, period)
    momentum_values[:period] = np.nan

    return momentum_values


__all__ = ["momentum", "MomentumSettings"]
