"""
Backtest Core - Test Configuration
==================================

Configuration pytest pour résoudre les imports correctement.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter la racine du projet au path pour les imports
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_ohlcv():
    """Génère des données OHLCV de test."""
    np.random.seed(42)
    n = 100

    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.exponential(1000, n),
    })

    df.index = pd.date_range("2024-01-01", periods=n, freq="1h")
    return df
