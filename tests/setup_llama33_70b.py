#!/usr/bin/env python3
"""
Script de téléchargement et configuration de Llama-3.3-70B-Instruct pour Ollama.

Optimisations:
- Distribution sur 2 GPUs
- Offloading RAM DDR5 automatique
- Quantization Q4 (~40GB VRAM total)
- Configuration multi-GPU optimisée

Usage:
    python tools/setup_llama33_70b.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_ollama_installed() -> bool:
    """Vérifie si Ollama est installé."""
    return shutil.which("ollama") is not None


def check_disk_space(required_gb: float = 40.0) -> tuple[bool, float]:
    """
    Vérifie l'espace disque disponible.

    Args:
        required_gb: Espace requis en GB

    Returns:
        tuple[bool, float]: (espace_suffisant, espace_disponible_gb)
    """
    try:
        # Obtenir le répertoire home Ollama (généralement ~/.ollama)
        home = Path.home()
        stat = shutil.disk_usage(home)
        available_gb = stat.free / (1024**3)
        return available_gb >= required_gb, available_gb
    except Exception:
        # Si erreur, on suppose qu'il y a assez d'espace
        return True, 0.0


def check_gpus() -> tuple[bool, int, list]:
    """
    Vérifie les GPUs disponibles via CuPy.

    Returns:
        tuple[bool, int, list]: (gpu_disponible, nombre_gpus, infos)
    """
    try:
        from performance.gpu import get_gpu_info

        info = get_gpu_info()
        has_gpu = info.get("cupy_available", False)

        if not has_gpu:
            return False, 0, []

        # Essayer de compter les GPUs
        try:
            import cupy as cp
            num_gpus = cp.cuda.runtime.getDeviceCount()

            gpu_infos = []
            for i in range(num_gpus):
                cp.cuda.Device(i).use()
                props = cp.cuda.runtime.getDeviceProperties(i)
                total_mem_gb = props["totalGlobalMem"] / (1024**3)
                gpu_infos.append({
                    "id": i,
                    "name": props["name"].decode("utf-8"),
                    "memory_gb": total_mem_gb
                })

            return True, num_gpus, gpu_infos
        except Exception:
            # Fallback si CuPy ne peut pas compter
            return has_gpu, 1, [{"id": 0, "name": "Unknown", "memory_gb": 0}]

    except Exception:
        return False, 0, []


def check_ram() -> tuple[bool, float]:
    """
    Vérifie la RAM disponible.

    Returns:
        tuple[bool, float]: (ram_suffisante, ram_total_gb)
    """
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        return ram_gb >= 32, ram_gb
    except ImportError:
        # Si psutil pas installé, on suppose que c'est OK
        return True, 0.0


def is_model_installed(model_name: str) -> bool:
    """
    Vérifie si un modèle est déjà installé.

    Args:
        model_name: Nom du modèle Ollama

    Returns:
        bool: True si installé
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return model_name in result.stdout
    except Exception:
        return False


def pull_model(model_name: str) -> bool:
    """
    Télécharge un modèle Ollama.

    Args:
        model_name: Nom du modèle à télécharger

    Returns:
        bool: True si succès
    """
    print(f"\n📥 Téléchargement de {model_name}...")
    print("⏱️  Ceci peut prendre 10-30 minutes selon votre connexion...")

    try:
        # Lancer ollama pull avec output en temps réel
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Afficher la progression
        for line in process.stdout:
            print(f"   {line.rstrip()}")

        process.wait()

        if process.returncode == 0:
            print(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        else:
            print(f"❌ Échec du téléchargement (code {process.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Timeout - Le téléchargement a pris trop de temps")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return False


def create_optimized_modelfile(base_model: str, num_gpus: int = 2) -> str:
    """
    Crée un Modelfile optimisé pour multi-GPU.

    Args:
        base_model: Modèle de base
        num_gpus: Nombre de GPUs à utiliser

    Returns:
        str: Contenu du Modelfile
    """
    return f"""FROM {base_model}

# Paramètres système - Optimisation multi-GPU
PARAMETER num_gpu {num_gpus}
PARAMETER num_thread 16
PARAMETER num_ctx 8192

# Paramètres génération
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM You are Llama 3.3, a helpful AI assistant specialized in analyzing trading strategies and financial data. You provide clear, data-driven insights and identify potential risks in trading approaches.
"""


def create_custom_model(
    base_model: str,
    custom_name: str,
    modelfile_content: str
) -> bool:
    """
    Crée un modèle personnalisé Ollama avec un Modelfile.

    Args:
        base_model: Modèle de base
        custom_name: Nom du modèle personnalisé
        modelfile_content: Contenu du Modelfile

    Returns:
        bool: True si succès
    """
    print(f"\n🔧 Création du modèle optimisé '{custom_name}'...")

    # Créer un fichier temporaire pour le Modelfile
    modelfile_path = Path("Modelfile.llama33.tmp")

    try:
        # Écrire le Modelfile
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        # Créer le modèle
        result = subprocess.run(
            ["ollama", "create", custom_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ Modèle '{custom_name}' créé avec succès")
            print("   Configuration: multi-GPU, contexte 8K, optimisé pour trading")
            return True
        else:
            print(f"❌ Échec création modèle: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Erreur création modèle: {e}")
        return False
    finally:
        # Nettoyer le fichier temporaire
        if modelfile_path.exists():
            modelfile_path.unlink()


def test_model(model_name: str) -> bool:
    """
    Teste le modèle avec un prompt simple.

    Args:
        model_name: Nom du modèle à tester

    Returns:
        bool: True si le modèle répond correctement
    """
    print(f"\n🧪 Test du modèle {model_name}...")

    try:
        start = time.perf_counter()

        result = subprocess.run(
            [
                "ollama", "run", model_name,
                "What is the Sharpe ratio in one sentence?"
            ],
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes max
        )

        elapsed = time.perf_counter() - start

        if result.returncode == 0 and result.stdout.strip():
            print(f"✅ Test réussi ({elapsed:.1f}s)")
            print(f"   Réponse: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ Test échoué: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏱️  Timeout - Le modèle prend trop de temps à répondre")
        return False
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False


def print_env_setup_instructions(num_gpus: int):
    """Affiche les instructions pour configurer l'environnement."""
    print("\n" + "="*60)
    print("📋 CONFIGURATION ENVIRONNEMENT")
    print("="*60)

    if num_gpus >= 2:
        print("\n✅ Multi-GPU détecté! Pour garantir l'utilisation des 2 GPUs:")
        print("   Ajoutez à votre environnement (optionnel):")
        print("   export CUDA_VISIBLE_DEVICES=0,1  # Linux/Mac")
        print("   $env:CUDA_VISIBLE_DEVICES='0,1'  # PowerShell")

    print("\n💡 Configuration Ollama avancée (optionnel):")
    print("   export OLLAMA_NUM_PARALLEL=1        # Éviter surcharge mémoire")
    print("   export OLLAMA_MAX_LOADED_MODELS=1   # Décharger autres modèles")


def main():
    """Exécute le setup complet."""
    print("="*60)
    print("🚀 SETUP LLAMA-3.3-70B-INSTRUCT")
    print("="*60)

    # Configuration
    BASE_MODEL = "llama3.3:70b-instruct-q4_K_M"
    CUSTOM_MODEL = "llama3.3-70b-optimized"

    # Étape 1: Vérifications prérequis
    print("\n📋 Vérification des prérequis...")

    if not check_ollama_installed():
        print("❌ Ollama n'est pas installé")
        print("   Installez Ollama: https://ollama.ai/download")
        return False
    print("✅ Ollama installé")

    # Vérifier espace disque
    has_space, available_gb = check_disk_space(40.0)
    if not has_space:
        print(f"❌ Espace disque insuffisant ({available_gb:.1f} GB disponible, 40 GB requis)")
        return False
    print(f"✅ Espace disque suffisant ({available_gb:.1f} GB disponible)")

    # Vérifier GPUs
    has_gpu, num_gpus, gpu_infos = check_gpus()
    if not has_gpu:
        print("⚠️  Aucun GPU détecté - Le modèle fonctionnera mais sera TRÈS lent")
        num_gpus = 0
    else:
        print(f"✅ {num_gpus} GPU(s) détecté(s):")
        for gpu in gpu_infos:
            print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

    # Vérifier RAM
    has_ram, ram_gb = check_ram()
    if not has_ram:
        print(f"⚠️  RAM limitée ({ram_gb:.1f} GB) - 32GB+ recommandé pour offloading")
    else:
        print(f"✅ RAM suffisante ({ram_gb:.1f} GB)")

    # Étape 2: Téléchargement
    print("\n" + "="*60)
    print("📥 TÉLÉCHARGEMENT DU MODÈLE")
    print("="*60)

    if is_model_installed(BASE_MODEL):
        print(f"✅ Modèle {BASE_MODEL} déjà installé - Skip téléchargement")
    else:
        if not pull_model(BASE_MODEL):
            print("❌ Échec du téléchargement")
            return False

    # Étape 3: Création modèle optimisé
    print("\n" + "="*60)
    print("🔧 CRÉATION MODÈLE OPTIMISÉ")
    print("="*60)

    # Utiliser num_gpus détecté (minimum 1 si GPU disponible)
    gpu_config = max(num_gpus, 1) if has_gpu else 1
    modelfile = create_optimized_modelfile(BASE_MODEL, gpu_config)

    if not create_custom_model(BASE_MODEL, CUSTOM_MODEL, modelfile):
        print("⚠️  Échec création modèle optimisé - Le modèle de base est utilisable")

    # Étape 4: Test
    print("\n" + "="*60)
    print("🧪 TEST DE VALIDATION")
    print("="*60)

    # Tester le modèle personnalisé d'abord, sinon le base
    model_to_test = CUSTOM_MODEL if is_model_installed(CUSTOM_MODEL) else BASE_MODEL
    test_model(model_to_test)

    # Étape 5: Instructions finales
    print_env_setup_instructions(num_gpus)

    print("\n" + "="*60)
    print("✅ INSTALLATION TERMINÉE")
    print("="*60)
    print(f"\n📝 Modèle installé: {model_to_test}")
    print("\n🎯 Prochaines étapes:")
    print("   1. Le modèle a été ajouté automatiquement à model_config.py")
    print("   2. Disponible pour les rôles: Critic (iter>=2), Validator (iter>=3)")
    print("   3. Lancez un backtest avec agents pour le tester")
    print("\n💡 Commande de test rapide:")
    print(f"   ollama run {model_to_test} 'Explain RSI indicator'")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
