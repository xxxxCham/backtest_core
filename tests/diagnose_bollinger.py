"""
Script de diagnostic pour les bandes de Bollinger.
Vérifie que le calcul et l'affichage sont corrects.
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


def diagnose():
    bollinger_bands = _load_bollinger_bands()
    # Créer des données synthétiques
    np.random.seed(42)
    n = 100
    close_prices = 100 + np.random.randn(n).cumsum() * 0.5
    close_series = pd.Series(close_prices)

    # Paramètres de test (valeurs de l'UI)
    period = 42
    std_dev = 2.25

    # Calculer les bandes
    upper, middle, lower = bollinger_bands(close_series, period, std_dev)

    # Analyser les dernières valeurs (après warmup)
    last_close = close_prices[-1]
    last_middle = middle[-1]
    last_upper = upper[-1]
    last_lower = lower[-1]
    last_std = (last_upper - last_middle) / std_dev

    print("=" * 60)
    print("DIAGNOSTIC BANDES DE BOLLINGER")
    print("=" * 60)
    print("\n📊 Paramètres:")
    print(f"  • Période (bb_period): {period}")
    print(f"  • Écart-type (bb_std): {std_dev}")
    print(f"  • Nombre de données: {n}")

    print("\n💹 Dernières valeurs:")
    print(f"  • Close: {last_close:.2f}")
    print(f"  • Middle (SMA): {last_middle:.2f}")
    print(f"  • Écart-type σ: {last_std:.2f}")
    print(f"  • Upper (middle + {std_dev}σ): {last_upper:.2f}")
    print(f"  • Lower (middle - {std_dev}σ): {last_lower:.2f}")

    print("\n📏 Distances:")
    print(f"  • Distance Upper-Close: {(last_upper - last_close):.2f} ({(last_upper - last_close)/last_close*100:.1f}%)")
    print(f"  • Distance Close-Lower: {(last_close - last_lower):.2f} ({(last_close - last_lower)/last_close*100:.1f}%)")
    print(f"  • Largeur totale: {(last_upper - last_lower):.2f} ({(last_upper - last_lower)/last_middle*100:.1f}%)")
    print(f"  • Distance Upper-Middle: {(last_upper - last_middle):.2f}")
    print(f"  • Distance Middle-Lower: {(last_middle - last_lower):.2f}")

    # Vérifier la cohérence
    print("\n✅ Vérifications:")
    upper_check = np.isclose(last_upper, last_middle + std_dev * last_std, rtol=1e-5)
    lower_check = np.isclose(last_lower, last_middle - std_dev * last_std, rtol=1e-5)
    symmetric_check = np.isclose(last_upper - last_middle, last_middle - last_lower, rtol=1e-5)

    print(f"  • Formule Upper correcte: {upper_check}")
    print(f"  • Formule Lower correcte: {lower_check}")
    print(f"  • Symétrie: {symmetric_check}")

    # Test avec une période élevée sur une série réelle
    print(f"\n📈 Impact d'une PÉRIODE ÉLEVÉE ({period}):")
    print(f"  • Avec période = 42, la SMA lisse sur {period} bougies")
    print(f"  • L'écart-type est calculé sur {period} bougies")
    print("  • Plus la période est élevée, plus:")
    print("    - La SMA est lisse (suit moins les variations)")
    print("    - L'écart-type capture la volatilité sur longue durée")
    print("    - Les bandes sont PLUS LARGES (car σ augmente)")

    # Comparer avec période standard
    period_std = 20
    upper_std, middle_std, lower_std = bollinger_bands(close_series, period_std, std_dev)
    last_std_20 = (upper_std[-1] - middle_std[-1]) / std_dev

    print("\n🔄 Comparaison avec période standard (20):")
    print(f"  • Écart-type σ (période=20): {last_std_20:.2f}")
    print(f"  • Écart-type σ (période={period}): {last_std:.2f}")
    print(f"  • Ratio: {last_std / last_std_20:.2f}x")
    print(f"  • Largeur bandes (période=20): {(upper_std[-1] - lower_std[-1]):.2f}")
    print(f"  • Largeur bandes (période={period}): {(last_upper - last_lower):.2f}")

    print("\n" + "=" * 60)
    print("🔍 CONCLUSION:")
    print("=" * 60)

    if last_std > last_std_20 * 1.5:
        print("⚠️  PÉRIODE ÉLEVÉE détectée:")
        print(f"   Avec bb_period={period}, l'écart-type est {last_std/last_std_20:.1f}x plus élevé")
        print("   que la période standard (20). Cela ÉLARGIT les bandes.")
        print("   \n   👉 Sur le graphique, les bandes DEVRAIENT être PLUS ÉLOIGNÉES")
        print("      du prix qu'avec période=20.")
    else:
        print("✅ Période dans la norme, bandes normales")

    print("\n💡 Si le graphique ne montre PAS de bandes larges:")
    print("   1. Vérifier que les paramètres bb_period et bb_std sont bien passés")
    print("   2. Vérifier le code d'affichage dans ui/components/charts.py")
    print("   3. Vérifier que c'est une visualisation avec VRAIES DONNÉES")
    print("      et pas un diagramme SYMBOLIQUE")


if __name__ == "__main__":
    diagnose()
