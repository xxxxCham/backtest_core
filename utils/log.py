"""
Backtest Core - Logging Simplifié
=================================

Configuration du logging simple et efficace.
"""

import logging
import sys
from typing import Optional

# Cache des loggers pour éviter duplication
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtient un logger configuré pour le module spécifié.

    Args:
        name: Nom du module (utilise __name__ généralement)

    Returns:
        Logger configuré avec format standard
    """
    if name is None:
        name = "backtest_core"

    # Retourner le logger en cache si existe
    if name in _loggers:
        return _loggers[name]

    # Créer nouveau logger
    logger = logging.getLogger(name)

    # Éviter duplication si déjà configuré
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Handler console
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Format simple et lisible
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Éviter propagation vers le root logger
        logger.propagate = False

    _loggers[name] = logger
    return logger


def set_level(level: str) -> None:
    """
    Change le niveau de log global.

    Args:
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    for logger in _loggers.values():
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)


__all__ = ["get_logger", "set_level"]
