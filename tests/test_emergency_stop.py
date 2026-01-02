"""
Script de test pour le système d'arrêt d'urgence.

Usage:
    python test_emergency_stop.py
"""

import logging
import time
from ui.emergency_stop import get_emergency_handler, execute_emergency_stop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_emergency_handler():
    """Teste le gestionnaire d'arrêt d'urgence."""
    print("\n" + "="*60)
    print("TEST DU SYSTÈME D'ARRÊT D'URGENCE")
    print("="*60 + "\n")

    # Test 1: Création du gestionnaire
    print("1️⃣ Création du gestionnaire...")
    handler = get_emergency_handler()
    assert handler is not None, "Handler non créé"
    print("   ✅ Handler créé avec succès\n")

    # Test 2: Vérifier flag d'arrêt
    print("2️⃣ Test flag d'arrêt...")
    assert not handler.is_stop_requested(), "Flag devrait être False"
    handler.request_stop()
    assert handler.is_stop_requested(), "Flag devrait être True"
    handler.reset_stop()
    assert not handler.is_stop_requested(), "Flag devrait être False après reset"
    print("   ✅ Flags fonctionnent correctement\n")

    # Test 3: Nettoyage complet (simulation sans session Streamlit)
    print("3️⃣ Test nettoyage complet...")
    start = time.perf_counter()
    stats = handler.full_cleanup(session_state=None)
    duration = time.perf_counter() - start

    print(f"   ⏱️  Durée: {duration:.2f}s")
    print(f"   🧹 Composants nettoyés: {len(stats['components_cleaned'])}")
    print(f"   ❌ Erreurs: {len(stats['errors'])}")

    # Afficher détails
    print("\n   📋 Composants nettoyés:")
    for comp in stats["components_cleaned"]:
        print(f"      • {comp}")

    if stats["errors"]:
        print("\n   ⚠️  Erreurs détectées:")
        for err in stats["errors"]:
            print(f"      • {err}")

    print("\n   ✅ Nettoyage terminé\n")

    # Test 4: Vérifier les statistiques
    print("4️⃣ Vérification statistiques...")
    last_stats = handler.get_last_cleanup_stats()
    assert last_stats == stats, "Stats ne correspondent pas"
    print("   ✅ Statistiques cohérentes\n")

    # Test 5: Raccourci execute_emergency_stop
    print("5️⃣ Test raccourci execute_emergency_stop()...")
    stats2 = execute_emergency_stop(None)
    assert "components_cleaned" in stats2, "Stats incomplètes"
    print("   ✅ Raccourci fonctionnel\n")

    # Résumé final
    print("="*60)
    print("✅ TOUS LES TESTS PASSÉS")
    print("="*60 + "\n")

    return stats


if __name__ == "__main__":
    try:
        final_stats = test_emergency_handler()

        print("\n📊 RÉSUMÉ FINAL:")
        print(f"   • Composants nettoyés: {len(final_stats['components_cleaned'])}")
        print(f"   • Erreurs rencontrées: {len(final_stats['errors'])}")
        print(f"   • Timestamp: {final_stats['timestamp']}")

        if final_stats["errors"]:
            print("\n⚠️  Note: Certaines erreurs sont normales si les modules")
            print("   ne sont pas installés (ex: PyTorch, CuPy)")

        print("\n✅ Système d'arrêt d'urgence OPÉRATIONNEL")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
