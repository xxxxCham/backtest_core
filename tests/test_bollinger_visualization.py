"""
Script de test pour vérifier l'amélioration des diagrammes Bollinger.
Compare l'ancien vs le nouveau générateur de données synthétiques.
"""

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache
def _load_bollinger_bands():
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    module = importlib.import_module("indicators.bollinger")
    return getattr(module, "bollinger_bands")


def old_synthetic_price(n: int = 160) -> tuple:
    """Ancienne version (trop lisse)."""
    x = np.arange(n)
    base = 100 + 4 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = 0.9 * np.sin(np.linspace(0, 11 * np.pi, n))
    price = base + noise
    price_series = pd.Series(price)
    return x, price, price_series


def new_synthetic_price(n: int = 160, volatility: float = 2.5) -> tuple:
    """Nouvelle version (plus réaliste)."""
    np.random.seed(42)
    x = np.arange(n)

    base = 100 + 4 * np.sin(np.linspace(0, 4 * np.pi, n))
    mid_freq = 0.9 * np.sin(np.linspace(0, 11 * np.pi, n))
    random_walk = np.random.randn(n).cumsum() * 0.3
    shocks = np.random.randn(n) * volatility

    price = base + mid_freq + random_walk + shocks
    price_series = pd.Series(price)
    return x, price, price_series


def test_comparison():
    """Compare l'impact des paramètres sur les deux générateurs."""
    bollinger_bands = _load_bollinger_bands()
    n = 300
    bb_period = 42
    bb_std = 2.25

    # Générer les deux versions
    x_old, price_old, ps_old = old_synthetic_price(n)
    x_new, price_new, ps_new = new_synthetic_price(n)

    # Calculer les bandes
    upper_old, middle_old, lower_old = bollinger_bands(ps_old, bb_period, bb_std)
    upper_new, middle_new, lower_new = bollinger_bands(ps_new, bb_period, bb_std)

    # Analyser les largeurs
    idx_valid = bb_period - 1

    width_old = np.nanmean(upper_old[idx_valid:] - lower_old[idx_valid:])
    width_new = np.nanmean(upper_new[idx_valid:] - lower_new[idx_valid:])

    std_old = np.nanstd(price_old[idx_valid:])
    std_new = np.nanstd(price_new[idx_valid:])

    print("=" * 70)
    print("COMPARAISON ANCIEN vs NOUVEAU GÉNÉRATEUR")
    print("=" * 70)
    print("\n📊 Paramètres:")
    print(f"  • Nombre de points: {n}")
    print(f"  • Période Bollinger: {bb_period}")
    print(f"  • Écart-type multiplier: {bb_std}")

    print("\n📈 ANCIEN générateur (trop lisse):")
    print(f"  • Volatilité des prix: {std_old:.2f}")
    print(f"  • Largeur moyenne bandes: {width_old:.2f}")
    print(f"  • Ratio largeur/prix: {width_old/100:.1%}")

    print("\n📈 NOUVEAU générateur (réaliste):")
    print(f"  • Volatilité des prix: {std_new:.2f}")
    print(f"  • Largeur moyenne bandes: {width_new:.2f}")
    print(f"  • Ratio largeur/prix: {width_new/100:.1%}")

    print("\n✅ AMÉLIORATION:")
    print(f"  • Volatilité augmentée: {std_new/std_old:.1f}x")
    print(f"  • Largeur bandes augmentée: {width_new/width_old:.1f}x")

    if width_new > width_old * 2:
        print(f"\n  🎯 SUCCÈS! Les bandes sont maintenant {width_new/width_old:.1f}x plus larges")
        print("     L'impact d'une période élevée (42) est VISIBLE sur le graphique")
    else:
        print(f"\n  ⚠️  Amélioration modérée ({width_new/width_old:.1f}x)")

    # Test avec période standard pour comparaison
    upper_std, middle_std, lower_std = bollinger_bands(ps_new, 20, bb_std)
    width_std = np.nanmean(upper_std[19:] - lower_std[19:])

    print("\n📊 Comparaison avec période STANDARD (20):")
    print(f"  • Largeur bandes (période=20): {width_std:.2f}")
    print(f"  • Largeur bandes (période=42): {width_new:.2f}")
    print(f"  • Ratio: {width_new/width_std:.2f}x")

    if width_new > width_std * 1.3:
        print(f"\n  ✅ Une période de 42 produit des bandes {width_new/width_std:.1f}x plus larges!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_comparison()
