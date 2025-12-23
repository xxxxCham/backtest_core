"""
Backtest Core - Pi Cycle Indicator
==================================

Commonly used with BTC tops: SMA(111) crosses above 2 * SMA(350).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.ema import sma
from indicators.registry import register_indicator


@dataclass
class PiCycleSettings:
    """Settings for Pi Cycle."""

    short_period: int = 111
    long_period: int = 350
    long_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.short_period < 1:
            raise ValueError("short_period must be >= 1")
        if self.long_period < 1:
            raise ValueError("long_period must be >= 1")
        if self.long_period <= self.short_period:
            raise ValueError("long_period must be > short_period")
        if self.long_multiplier <= 0:
            raise ValueError("long_multiplier must be > 0")


def pi_cycle(
    close: pd.Series | np.ndarray,
    short_period: int = 111,
    long_period: int = 350,
    long_multiplier: float = 2.0,
    settings: PiCycleSettings | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute Pi Cycle moving averages and cross signal.

    Returns:
        Dict with short_ma, long_ma, and signal (+1/-1/0)
    """
    if settings is not None:
        short_period = settings.short_period
        long_period = settings.long_period
        long_multiplier = settings.long_multiplier

    if isinstance(close, pd.Series):
        close = close.values

    short_ma = sma(close, int(short_period))
    long_ma = sma(close, int(long_period)) * float(long_multiplier)

    signal = np.zeros(len(close), dtype=np.float64)
    valid = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    if np.any(valid):
        above = short_ma > long_ma
        above_prev = np.roll(above, 1)
        above_prev[0] = above[0]
        cross_up = valid & above & ~above_prev
        cross_down = valid & ~above & above_prev
        signal[cross_up] = 1.0
        signal[cross_down] = -1.0

    return {
        "short_ma": short_ma,
        "long_ma": long_ma,
        "signal": signal,
    }


def calculate_pi_cycle(df: pd.DataFrame, **params) -> dict[str, np.ndarray]:
    """
    Wrapper for registry calculation.

    Params:
        short_period: Short SMA period (default: 111)
        long_period: Long SMA period (default: 350)
        long_multiplier: Long MA multiplier (default: 2.0)
    """
    return pi_cycle(
        df["close"],
        short_period=int(params.get("short_period", 111)),
        long_period=int(params.get("long_period", 350)),
        long_multiplier=float(params.get("long_multiplier", 2.0)),
    )


register_indicator(
    "pi_cycle",
    calculate_pi_cycle,
    settings_class=PiCycleSettings,
    required_columns=("close",),
    description="Pi Cycle - SMA(111) vs 2 * SMA(350)",
)


__all__ = [
    "pi_cycle",
    "calculate_pi_cycle",
    "PiCycleSettings",
]
