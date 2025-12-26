"""
Backtest Core - Indicator Registry
==================================

Registre centralisé des indicateurs pour faciliter l'accès et le calcul.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from .adx import adx, calculate_adx
from .atr import ATRSettings, atr

# Imports relatifs des indicateurs
from .bollinger import BollingerSettings, bollinger_bands
from .ema import EMASettings, ema, sma
from .macd import calculate_macd, macd
from .rsi import RSISettings, rsi
from .stochastic import stochastic

# Nouveaux indicateurs
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


@dataclass
class IndicatorInfo:
    """Métadonnées d'un indicateur."""

    name: str
    function: Callable
    settings_class: Optional[type]
    required_columns: Tuple[str, ...]
    description: str


# Registre global des indicateurs
_INDICATOR_REGISTRY: Dict[str, IndicatorInfo] = {}


def register_indicator(
    name: str,
    function: Callable,
    settings_class: Optional[type] = None,
    required_columns: Tuple[str, ...] = ("close",),
    description: str = ""
) -> None:
    """Enregistre un nouvel indicateur."""
    _INDICATOR_REGISTRY[name.lower()] = IndicatorInfo(
        name=name,
        function=function,
        settings_class=settings_class,
        required_columns=required_columns,
        description=description
    )


def get_indicator(name: str) -> Optional[IndicatorInfo]:
    """Récupère les infos d'un indicateur."""
    return _INDICATOR_REGISTRY.get(name.lower())


def list_indicators() -> list[str]:
    """Liste tous les indicateurs disponibles."""
    return list(_INDICATOR_REGISTRY.keys())


def calculate_indicator(
    name: str,
    df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Calcule un indicateur par son nom.

    Args:
        name: Nom de l'indicateur (bollinger, atr, rsi, ema, sma)
        df: DataFrame OHLCV
        params: Paramètres de l'indicateur

    Returns:
        Résultat du calcul (array ou tuple selon indicateur)

    Raises:
        ValueError: Si indicateur inconnu ou colonnes manquantes
    """
    name = name.lower()
    info = get_indicator(name)

    if info is None:
        available = ", ".join(list_indicators())
        raise ValueError(f"Indicateur inconnu: '{name}'. Disponibles: {available}")

    # Vérifier colonnes requises
    missing = [col for col in info.required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {name}: {missing}")

    # Préparer les arguments
    params = params or {}

    # Appeler la fonction appropriée
    if name == "bollinger":
        return bollinger_bands(
            df["close"],
            period=int(params.get("period", 20)),
            std_dev=float(params.get("std_dev", 2.0))
        )

    elif name == "atr":
        return atr(
            df["high"],
            df["low"],
            df["close"],
            period=int(params.get("period", 14)),
            method=params.get("method", "ema")
        )

    elif name == "rsi":
        return rsi(
            df["close"],
            period=int(params.get("period", 14))
        )

    elif name == "ema":
        return ema(
            df["close"],
            period=int(params.get("period", 20))
        )

    elif name == "sma":
        return sma(
            df["close"],
            period=int(params.get("period", 20))
        )

    elif name == "macd":
        macd_line, signal_line, histogram = macd(
            df["close"],
            fast_period=int(params.get("fast_period", 12)),
            slow_period=int(params.get("slow_period", 26)),
            signal_period=int(params.get("signal_period", 9))
        )
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    elif name == "adx":
        adx_val, plus_di, minus_di = adx(
            df["high"],
            df["low"],
            df["close"],
            period=int(params.get("period", 14))
        )
        return {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di}

    elif name == "stochastic":
        stoch_k, stoch_d = stochastic(
            df["high"],
            df["low"],
            df["close"],
            k_period=int(params.get("k_period", 14)),
            d_period=int(params.get("d_period", 3)),
            smooth_k=int(params.get("smooth_k", 3))
        )
        return (stoch_k, stoch_d)

    elif name == "vwap":
        return vwap(
            df["high"], df["low"], df["close"], df["volume"],
            period=params.get("period", None)
        )

    elif name == "donchian":
        upper, middle, lower = donchian_channel(
            df["high"], df["low"],
            period=int(params.get("period", 20))
        )
        return {"upper": upper, "middle": middle, "lower": lower}

    elif name == "cci":
        return cci(
            df["high"], df["low"], df["close"],
            period=int(params.get("period", 20))
        )

    elif name == "keltner":
        middle, upper, lower = keltner_channel(
            df["high"], df["low"], df["close"],
            ema_period=int(params.get("ema_period", 20)),
            atr_period=int(params.get("atr_period", 10)),
            atr_multiplier=float(params.get("atr_multiplier", 2.0))
        )
        return {"middle": middle, "upper": upper, "lower": lower}

    elif name == "mfi":
        return mfi(
            df["high"], df["low"], df["close"], df["volume"],
            period=int(params.get("period", 14))
        )

    elif name == "williams_r":
        return williams_r(
            df["high"], df["low"], df["close"],
            period=int(params.get("period", 14))
        )

    elif name == "momentum":
        return momentum(
            df["close"],
            period=int(params.get("period", 14))
        )

    elif name == "obv":
        return obv(df["close"], df["volume"])

    elif name == "roc":
        return roc(
            df["close"],
            period=int(params.get("period", 12))
        )

    elif name == "aroon":
        aroon_up, aroon_down = aroon(
            df["high"], df["low"],
            period=int(params.get("period", 14))
        )
        return {"aroon_up": aroon_up, "aroon_down": aroon_down}

    elif name == "supertrend":
        st_values, st_direction = supertrend(
            df["high"], df["low"], df["close"],
            atr_period=int(params.get("atr_period", 10)),
            multiplier=float(params.get("multiplier", 3.0))
        )
        return {"supertrend": st_values, "direction": st_direction}

    else:
        # Appel générique
        return info.function(df, **params)


# Enregistrement des indicateurs de base
register_indicator(
    name="bollinger",
    function=bollinger_bands,
    settings_class=BollingerSettings,
    required_columns=("close",),
    description="Bandes de Bollinger - Mean reversion indicator"
)

register_indicator(
    name="atr",
    function=atr,
    settings_class=ATRSettings,
    required_columns=("high", "low", "close"),
    description="Average True Range - Volatility indicator"
)

register_indicator(
    name="rsi",
    function=rsi,
    settings_class=RSISettings,
    required_columns=("close",),
    description="Relative Strength Index - Momentum oscillator"
)

register_indicator(
    name="ema",
    function=ema,
    settings_class=EMASettings,
    required_columns=("close",),
    description="Exponential Moving Average"
)

register_indicator(
    name="sma",
    function=sma,
    settings_class=None,
    required_columns=("close",),
    description="Simple Moving Average"
)

register_indicator(
    name="macd",
    function=calculate_macd,
    settings_class=None,
    required_columns=("close",),
    description="Moving Average Convergence Divergence - Momentum indicator"
)

register_indicator(
    name="adx",
    function=calculate_adx,
    settings_class=None,
    required_columns=("high", "low", "close"),
    description="Average Directional Index - Trend strength indicator"
)

register_indicator(
    name="stochastic",
    function=stochastic,
    settings_class=None,
    required_columns=("high", "low", "close"),
    description="Stochastic Oscillator - Momentum indicator for overbought/oversold"
)

# Nouveaux indicateurs

register_indicator(
    name="vwap",
    function=vwap,
    settings_class=VWAPSettings,
    required_columns=("high", "low", "close", "volume"),
    description="Volume Weighted Average Price"
)

register_indicator(
    name="donchian",
    function=donchian_channel,
    settings_class=DonchianSettings,
    required_columns=("high", "low"),
    description="Donchian Channel - Breakout indicator"
)

register_indicator(
    name="cci",
    function=cci,
    settings_class=CCISettings,
    required_columns=("high", "low", "close"),
    description="Commodity Channel Index - Momentum oscillator"
)

register_indicator(
    name="keltner",
    function=keltner_channel,
    settings_class=KeltnerSettings,
    required_columns=("high", "low", "close"),
    description="Keltner Channel - Volatility channel based on EMA and ATR"
)

register_indicator(
    name="mfi",
    function=mfi,
    settings_class=MFISettings,
    required_columns=("high", "low", "close", "volume"),
    description="Money Flow Index - Volume-weighted RSI"
)

register_indicator(
    name="williams_r",
    function=williams_r,
    settings_class=WilliamsRSettings,
    required_columns=("high", "low", "close"),
    description="Williams %R - Momentum oscillator"
)

register_indicator(
    name="momentum",
    function=momentum,
    settings_class=MomentumSettings,
    required_columns=("close",),
    description="Momentum - Absolute price change over period"
)

register_indicator(
    name="obv",
    function=obv,
    settings_class=OBVSettings,
    required_columns=("close", "volume"),
    description="On-Balance Volume - Cumulative volume flow"
)

register_indicator(
    name="roc",
    function=roc,
    settings_class=ROCSettings,
    required_columns=("close",),
    description="Rate of Change - Percentage price change"
)

register_indicator(
    name="aroon",
    function=aroon,
    settings_class=AroonSettings,
    required_columns=("high", "low"),
    description="Aroon Indicator - Trend identification"
)

register_indicator(
    name="supertrend",
    function=supertrend,
    settings_class=SuperTrendSettings,
    required_columns=("high", "low", "close"),
    description="SuperTrend - ATR-based trend follower"
)

# Late imports for indicators that self-register to avoid circular imports.
from . import amplitude_hunter  # noqa: F401
from . import fear_greed  # noqa: F401
from . import fibonacci  # noqa: F401
from . import ichimoku  # noqa: F401
from . import onchain_smoothing  # noqa: F401
from . import pi_cycle  # noqa: F401
from . import pivot_points  # noqa: F401
from . import psar  # noqa: F401
from . import standard_deviation  # noqa: F401
from . import volume_oscillator  # noqa: F401
from . import vortex  # noqa: F401


class IndicatorRegistry:
    """
    Classe wrapper pour le registre d'indicateurs.

    Permet une utilisation orientée objet et potentiellement
    le caching des résultats dans le futur.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = False

    def enable_cache(self) -> None:
        """Active le cache des calculs."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Désactive et vide le cache."""
        self._cache_enabled = False
        self._cache.clear()

    def clear_cache(self) -> None:
        """Vide le cache."""
        self._cache.clear()

    def calculate(
        self,
        name: str,
        df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Calcule un indicateur, avec mise en cache optionnelle.
        """
        cache_key = f"{name}_{hash(df.index[0])}_{hash(df.index[-1])}_{str(params)}"

        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        result = calculate_indicator(name, df, params)

        if self._cache_enabled:
            self._cache[cache_key] = result

        return result

    def calculate_multiple(
        self,
        df: pd.DataFrame,
        indicator_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcule plusieurs indicateurs d'un coup.

        Args:
            df: DataFrame OHLCV
            indicator_configs: Dict {nom_indicateur: {params}}

        Returns:
            Dict {nom_indicateur: résultat}
        """
        results = {}
        for name, params in indicator_configs.items():
            results[name] = self.calculate(name, df, params)
        return results

    @staticmethod
    def list_available() -> list[str]:
        """Liste les indicateurs disponibles."""
        return list_indicators()

    @staticmethod
    def get_info(name: str) -> Optional[IndicatorInfo]:
        """Récupère les infos d'un indicateur."""
        return get_indicator(name)


__all__ = [
    "IndicatorRegistry",
    "calculate_indicator",
    "list_indicators",
    "register_indicator",
    "get_indicator"
]
