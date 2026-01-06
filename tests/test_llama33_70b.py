#!/usr/bin/env python3
"""
Script de test et validation pour Llama-3.3-70B-Instruct.

Vérifie:
1. Modèle disponible dans la configuration
2. Sélection correcte pour les rôles Critic/Validator
3. Inférence fonctionnelle
4. Distribution GPU et utilisation mémoire
5. Intégration avec GPUMemoryManager

Usage:
    python tools/test_llama33_70b.py
"""

import os
import sys
import time

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_model_in_config():
    """Vérifie que Llama-3.3-70B est dans la configuration."""
    try:
        from agents.model_config import KNOWN_MODELS

        print("\n📋 Vérification de la configuration...")

        # Vérifier les deux variantes
        models_to_check = [
            "llama3.3:70b-instruct-q4_K_M",
            "llama3.3-70b-optimized"
        ]

        found = []
        for model_name in models_to_check:
            if model_name in KNOWN_MODELS:
                info = KNOWN_MODELS[model_name]
                print(f"✅ {model_name} trouvé dans KNOWN_MODELS")
                print(f"   Catégorie: {info.category.value}")
                print(f"   Description: {info.description}")
                print(f"   Recommandé pour: {', '.join(info.recommended_for)}")
                print(f"   Temps moyen: {info.avg_response_time_s}s")
                found.append(model_name)
            else:
                print(f"❌ {model_name} NON trouvé dans KNOWN_MODELS")

        return len(found) > 0

    except Exception as e:
        print(f"❌ Erreur lors de la vérification de la config: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_role_assignment():
    """Vérifie que le modèle est assigné aux bons rôles."""
    try:
        from agents.model_config import RoleModelConfig

        print("\n🎯 Vérification de l'assignation par rôle...")

        config = RoleModelConfig()

        # Vérifier Critic
        critic_models = config.critic.models
        if "llama3.3-70b-optimized" in critic_models:
            print("✅ llama3.3-70b-optimized dans Critic.models")
            print(f"   Allow heavy après itération: {config.critic.allow_heavy_after_iteration}")
        else:
            print("❌ llama3.3-70b-optimized PAS dans Critic.models")
            return False

        # Vérifier Validator
        validator_models = config.validator.models
        if "llama3.3-70b-optimized" in validator_models:
            print("✅ llama3.3-70b-optimized dans Validator.models")
            print(f"   Allow heavy après itération: {config.validator.allow_heavy_after_iteration}")
        else:
            print("❌ llama3.3-70b-optimized PAS dans Validator.models")
            return False

        return True

    except Exception as e:
        print(f"❌ Erreur lors de la vérification des rôles: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_selection():
    """Teste la sélection du modèle via get_model()."""
    try:
        from agents.model_config import RoleModelConfig

        print("\n🔍 Test de sélection du modèle...")

        config = RoleModelConfig()

        # Vérifier si le modèle est installé
        installed = config.get_installed_models()
        print(f"   Modèles installés: {len(installed)}")

        is_installed = "llama3.3-70b-optimized" in installed
        if is_installed:
            print("✅ llama3.3-70b-optimized est installé")
        else:
            print("⚠️  llama3.3-70b-optimized n'est pas encore installé")
            print("   Lancez: python tools/setup_llama33_70b.py")
            return False

        # Tester sélection pour Critic (iteration=3 pour autoriser HEAVY)
        model_critic = config.get_model("critic", iteration=3, allow_heavy=True)
        print(f"   Sélection Critic (iter=3, heavy=True): {model_critic}")

        # Tester sélection pour Validator (iteration=4 pour autoriser HEAVY)
        model_validator = config.get_model("validator", iteration=4, allow_heavy=True)
        print(f"   Sélection Validator (iter=4, heavy=True): {model_validator}")

        return True

    except Exception as e:
        print(f"❌ Erreur lors du test de sélection: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ollama_availability():
    """Vérifie qu'Ollama est actif et que le modèle est disponible."""
    try:
        from agents.ollama_manager import is_ollama_available, list_ollama_models

        print("\n🔌 Vérification d'Ollama...")

        if not is_ollama_available():
            print("❌ Ollama n'est pas actif")
            print("   Lancez: ollama serve")
            return False

        print("✅ Ollama actif")

        models = list_ollama_models()
        print(f"   {len(models)} modèle(s) installé(s)")

        # Chercher les variantes Llama 3.3
        llama33_models = [m for m in models if "llama3.3" in m.lower()]
        if llama33_models:
            print("✅ Modèles Llama 3.3 trouvés:")
            for m in llama33_models:
                print(f"   - {m}")
            return True
        else:
            print("⚠️  Aucun modèle Llama 3.3 trouvé")
            print("   Lancez: python tools/setup_llama33_70b.py")
            return False

    except Exception as e:
        print(f"❌ Erreur lors de la vérification Ollama: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Teste une inférence simple avec le modèle."""
    try:
        from agents.llm_client import LLMConfig, create_llm_client

        print("\n🧪 Test d'inférence...")

        # Créer le client LLM
        config = LLMConfig(
            model="llama3.3-70b-optimized",
            provider="ollama",
            temperature=0.7
        )

        client = create_llm_client(config)

        if not client.is_available():
            print("❌ Le client LLM n'est pas disponible")
            return False

        print("✅ Client LLM créé")

        # Test avec un prompt simple
        prompt = "Explain the Sharpe ratio in one sentence."
        print(f"   Prompt: {prompt}")

        start = time.perf_counter()

        try:
            response = client.simple_chat(prompt)
            elapsed = time.perf_counter() - start

            if response and hasattr(response, 'content') and response.content:
                print(f"✅ Réponse reçue ({elapsed:.1f}s)")
                print(f"   Réponse: {response.content[:200]}...")
                print(f"   Longueur: {len(response.content)} caractères")
                return True
            else:
                print("❌ Réponse vide ou invalide")
                return False

        except TimeoutError:
            print("⏱️  Timeout - Le modèle prend trop de temps")
            return False

    except Exception as e:
        print(f"❌ Erreur lors du test d'inférence: {e}")
        import traceback
        traceback.print_exc()
        return False


def monitor_gpu_usage():
    """Affiche l'utilisation GPU actuelle."""
    try:
        from performance.gpu import get_gpu_info

        print("\n📊 Monitoring GPU...")

        info = get_gpu_info()

        if not info.get("cupy_available", False):
            print("⚠️  CuPy non disponible - Monitoring GPU limité")
            return True

        print(f"✅ GPU détecté: {info.get('cupy_device_name', 'Unknown')}")
        print(f"   VRAM totale: {info.get('cupy_memory_total_gb', 0):.2f} GB")
        print(f"   VRAM libre: {info.get('cupy_memory_free_gb', 0):.2f} GB")
        print(f"   VRAM utilisée: {info.get('cupy_memory_total_gb', 0) - info.get('cupy_memory_free_gb', 0):.2f} GB")

        return True

    except Exception as e:
        print(f"⚠️  Impossible de monitorer le GPU: {e}")
        return True  # Non-bloquant


def test_gpu_memory_manager():
    """Teste le GPUMemoryManager avec le modèle."""
    try:
        from agents.ollama_manager import GPUMemoryManager

        print("\n🔄 Test GPUMemoryManager...")

        manager = GPUMemoryManager(
            model_name="llama3.3-70b-optimized",
            verbose=True
        )

        # Test is_model_loaded
        is_loaded = manager.is_model_loaded()
        print(f"   Modèle chargé: {is_loaded}")

        # Test unload/reload
        print("   Test unload...")
        state = manager.unload()
        print(f"   État: was_loaded={state.was_loaded}, unload_time={state.unload_time_ms:.0f}ms")

        time.sleep(2)  # Pause

        print("   Test reload...")
        success = manager.reload(state)
        print(f"   Reload: {'✅ Succès' if success else '❌ Échec'}")

        stats = manager.get_stats()
        print(f"   Stats: {stats}")

        return success

    except Exception as e:
        print(f"⚠️  Erreur GPUMemoryManager: {e}")
        import traceback
        traceback.print_exc()
        return True  # Non-bloquant


def print_recommendations():
    """Affiche des recommandations d'utilisation."""
    print("\n" + "="*60)
    print("💡 RECOMMANDATIONS D'UTILISATION")
    print("="*60)

    print("\n📌 Utilisation dans un backtest:")
    print("   - Le modèle sera automatiquement sélectionné pour Critic (iter>=2)")
    print("   - Et pour Validator (iter>=3)")
    print("   - Utilisez allow_heavy=True pour forcer l'utilisation dès iter=1")

    print("\n⚡ Performance:")
    print("   - Temps de réponse attendu: ~5 min pour analyses complexes")
    print("   - Distribution automatique sur 2 GPUs via Ollama")
    print("   - Offloading RAM pour les layers lourds")

    print("\n🔧 Optimisations:")
    print("   - Utilisez gpu_compute_context() pour libérer la VRAM avant backtests")
    print("   - Déchargez avec GPUMemoryManager si besoin de VRAM")
    print("   - Variables d'environnement optionnelles:")
    print("     export CUDA_VISIBLE_DEVICES=0,1")
    print("     export OLLAMA_NUM_PARALLEL=1")


def main():
    """Exécute tous les tests."""
    print("="*60)
    print("🧪 TEST LLAMA-3.3-70B-INSTRUCT")
    print("="*60)

    results = {}

    # Test 1: Configuration
    results["config"] = check_model_in_config()

    # Test 2: Rôles
    results["roles"] = check_role_assignment()

    # Test 3: Sélection
    results["selection"] = check_model_selection()

    # Test 4: Ollama disponibilité
    results["ollama"] = check_ollama_availability()

    # Test 5: Inférence (si Ollama OK)
    if results["ollama"]:
        results["inference"] = test_inference()
    else:
        results["inference"] = None
        print("\n⏭️  Skip test d'inférence (Ollama non disponible)")

    # Test 6: Monitoring GPU
    results["gpu"] = monitor_gpu_usage()

    # Test 7: GPUMemoryManager (si modèle installé)
    if results["ollama"]:
        results["gpu_manager"] = test_gpu_memory_manager()
    else:
        results["gpu_manager"] = None
        print("\n⏭️  Skip test GPUMemoryManager (modèle non installé)")

    # Résumé
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}: PASS")
        elif result is False:
            print(f"❌ {test_name}: FAIL")
        else:
            print(f"⏭️  {test_name}: SKIP")

    print(f"\n📊 Total: {passed}/{total} réussis, {failed} échecs, {skipped} skip")

    # Recommandations
    if passed >= 4:  # Au moins config, roles, selection, gpu
        print_recommendations()

    print("\n" + "="*60)

    if failed == 0:
        print("✅ TOUS LES TESTS PASSÉS")
        print("="*60)
        return True
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("="*60)
        print("\n💡 Actions recommandées:")
        if not results["config"]:
            print("   - Vérifiez que model_config.py a été modifié correctement")
        if not results["ollama"]:
            print("   - Installez le modèle: python tools/setup_llama33_70b.py")
        if not results["inference"]:
            print("   - Vérifiez les logs Ollama pour plus de détails")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
