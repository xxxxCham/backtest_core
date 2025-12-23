"""
Indicator ranges loader
=======================

Reads config/indicator_ranges.toml for UI parameter definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib


_INDICATOR_RANGES_CACHE: Optional[Dict[str, Any]] = None


def load_indicator_ranges(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load indicator ranges from TOML.

    Args:
        path: Optional custom path to the TOML file.

    Returns:
        Nested dict from TOML (indicator -> param -> spec).
    """
    global _INDICATOR_RANGES_CACHE

    use_cache = path is None
    if _INDICATOR_RANGES_CACHE is not None and use_cache:
        return _INDICATOR_RANGES_CACHE

    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "indicator_ranges.toml"

    if not path.exists():
        data: Dict[str, Any] = {}
    else:
        with path.open("rb") as handle:
            data = tomllib.load(handle)

    if use_cache:
        _INDICATOR_RANGES_CACHE = data

    return data


def get_indicator_param_specs(
    indicator_name: str,
    ranges: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Return parameter specs for a single indicator.
    """
    if ranges is None:
        ranges = load_indicator_ranges()

    return ranges.get(indicator_name.lower(), {})


__all__ = ["load_indicator_ranges", "get_indicator_param_specs"]
