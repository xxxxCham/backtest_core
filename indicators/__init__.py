"""
Backtest Core - Indicators Package
==================================

Indicateurs techniques vectorisés avec NumPy.
"""

from .adx import adx, calculate_adx
from .atr import ATRSettings, atr
from .bollinger import BollingerSettings, bollinger_bands
from .ema import EMASettings, ema, sma
from .macd import macd, macd_signal
from .registry import calculate_indicator, list_indicators
from .rsi import RSISettings, rsi
from .stochastic import stochastic, stochastic_signal

# Indicateurs ajoutés 12/12/2025
from .vwap import vwap, VWAPSettings
from .donchian import donchian_channel, DonchianSettings
from .cci import cci, CCISettings
from .keltner import keltner_channel, KeltnerSettings
from .mfi import mfi, MFISettings
from .williams_r import williams_r, WilliamsRSettings
from .momentum import momentum, MomentumSettings
from .obv import obv, OBVSettings
from .roc import roc, ROCSettings
from .aroon import aroon, AroonSettings
from .supertrend import supertrend, SuperTrendSettings

# Indicateurs Phase 2 (13/12/2025)
from .ichimoku import ichimoku, ichimoku_signal, calculate_ichimoku
from .psar import parabolic_sar, psar_signal, calculate_psar
from .stoch_rsi import stochastic_rsi, stoch_rsi_signal, calculate_stoch_rsi
from .vortex import vortex, vortex_signal, calculate_vortex

# Additional indicators
from .volume_oscillator import (
    volume_oscillator,
    calculate_volume_oscillator,
    VolumeOscillatorSettings,
)
from .standard_deviation import (
    standard_deviation,
    calculate_standard_deviation,
    StandardDeviationSettings,
)
from .fibonacci import (
    fibonacci_levels,
    calculate_fibonacci_levels,
    FibonacciSettings,
)
from .pivot_points import (
    pivot_points,
    calculate_pivot_points,
    PivotPointsSettings,
)
from .onchain_smoothing import (
    onchain_smoothing,
    calculate_onchain_smoothing,
    OnchainSmoothingSettings,
)
from .fear_greed import (
    fear_greed_index,
    calculate_fear_greed,
    FearGreedSettings,
)
from .pi_cycle import (
    pi_cycle,
    calculate_pi_cycle,
    PiCycleSettings,
)
from .amplitude_hunter import (
    amplitude_hunter,
    calculate_amplitude_hunter,
    AmplitudeHunterSettings,
)

__all__ = [
    # Indicateurs de base
    "bollinger_bands",
    "BollingerSettings",
    "atr",
    "ATRSettings",
    "rsi",
    "RSISettings",
    "ema",
    "sma",
    "EMASettings",
    "macd",
    "macd_signal",
    "adx",
    "calculate_adx",
    "stochastic",
    "stochastic_signal",
    # Indicateurs 12/12/2025
    "vwap",
    "VWAPSettings",
    "donchian_channel",
    "DonchianSettings",
    "cci",
    "CCISettings",
    "keltner_channel",
    "KeltnerSettings",
    "mfi",
    "MFISettings",
    "williams_r",
    "WilliamsRSettings",
    "momentum",
    "MomentumSettings",
    "obv",
    "OBVSettings",
    "roc",
    "ROCSettings",
    "aroon",
    "AroonSettings",
    "supertrend",
    "SuperTrendSettings",
    # Phase 2 (13/12/2025)
    "ichimoku",
    "ichimoku_signal",
    "calculate_ichimoku",
    "parabolic_sar",
    "psar_signal",
    "calculate_psar",
    "stochastic_rsi",
    "stoch_rsi_signal",
    "calculate_stoch_rsi",
    "vortex",
    "vortex_signal",
    "calculate_vortex",
    # Additional indicators
    "volume_oscillator",
    "calculate_volume_oscillator",
    "VolumeOscillatorSettings",
    "standard_deviation",
    "calculate_standard_deviation",
    "StandardDeviationSettings",
    "fibonacci_levels",
    "calculate_fibonacci_levels",
    "FibonacciSettings",
    "pivot_points",
    "calculate_pivot_points",
    "PivotPointsSettings",
    "onchain_smoothing",
    "calculate_onchain_smoothing",
    "OnchainSmoothingSettings",
    "fear_greed_index",
    "calculate_fear_greed",
    "FearGreedSettings",
    "pi_cycle",
    "calculate_pi_cycle",
    "PiCycleSettings",
    "amplitude_hunter",
    "calculate_amplitude_hunter",
    "AmplitudeHunterSettings",
    # Registre
    "calculate_indicator",
    "list_indicators",
]
