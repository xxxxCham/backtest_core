"""
Backtest Core - Data Package
============================

Chargement et gestion des données OHLCV.
"""

from .loader import discover_available_data, load_ohlcv

__all__ = ["load_ohlcv", "discover_available_data"]
