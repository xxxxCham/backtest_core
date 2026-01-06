#!/usr/bin/env python3
"""
Script pour configurer Ollama en mode multi-GPU et vérifier l'utilisation GPU.

Purpose:
- Configurer les variables d'environnement pour multi-GPU Ollama
- Créer un Modelfile optimisé pour Llama 3.3 70B avec répartition GPU
- Vérifier que le GPU est utilisé pendant l'inférence
- Monitorer l'utilisation VRAM en temps réel

Usage:
    python tools/configure_ollama_multigpu.py [--test]

Options:
    --test    Lancer un test d'inférence après configuration
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_nvidia_smi():
    """Vérifier si nvidia-smi est disponible."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ nvidia-smi disponible")
        print("\n📊 GPUs détectés:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvidia-smi non trouvé - CUDA n'est peut-être pas installé")
        return False


def get_gpu_count():
    """Retourne le nombre de GPUs CUDA disponibles (ignore iGPU)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Filtrer uniquement les GPUs NVIDIA (ignore AMD iGPU)
        lines = result.stdout.strip().split('\n')
        cuda_gpus = [line for line in lines if line and 'NVIDIA' in line]
        return len(cuda_gpus)
    except Exception:
        return 0


def create_modelfile_multigpu(model_base: str = "llama3.3:70b-instruct-q4_K_M", num_gpu: int = 2):
    """
    Crée un Modelfile optimisé pour multi-GPU.

    Args:
        model_base: Modèle de base Ollama
        num_gpu: Nombre de GPUs à utiliser
    """
    modelfile_content = f'''FROM {model_base}

# Configuration multi-GPU
PARAMETER num_gpu {num_gpu}

# Optimisations mémoire
PARAMETER num_thread 8
PARAMETER num_ctx 8192

# Température et sampling
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# Répétition
PARAMETER repeat_penalty 1.1

# System prompt optimisé pour analyse financière
SYSTEM You are an expert financial analyst and trading strategist. Analyze data precisely and provide actionable insights.
'''

    modelfile_path = Path("Modelfile.llama33-multigpu")
    modelfile_path.write_text(modelfile_content)

    print(f"\n📝 Modelfile créé: {modelfile_path}")
    print(f"   Modèle base: {model_base}")
    print(f"   GPUs: {num_gpu}")

    return modelfile_path


def build_model_from_modelfile(modelfile_path: Path, model_name: str = "llama3.3-70b-multigpu"):
    """
    Construit un modèle Ollama à partir du Modelfile.

    Args:
        modelfile_path: Chemin vers le Modelfile
        model_name: Nom du nouveau modèle
    """
    print(f"\n🔨 Construction du modèle {model_name}...")

    try:
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"✅ Modèle {model_name} créé avec succès")
        if result.stdout:
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de la création du modèle:")
        print(e.stderr)
        return False


def monitor_gpu_during_inference(model_name: str, prompt: str = "Explain what is a moving average in 50 words."):
    """
    Lance une inférence et monitore l'utilisation GPU en temps réel.

    Args:
        model_name: Nom du modèle Ollama
        prompt: Prompt de test
    """
    print("\n🧪 Test d'inférence avec monitoring GPU...")
    print(f"   Modèle: {model_name}")
    print(f"   Prompt: {prompt[:50]}...")

    # Lancer nvidia-smi en monitoring continu dans un subprocess
    print("\n📊 Démarrage monitoring GPU (VRAM)...")

    import threading

    stop_monitoring = threading.Event()

    def monitor_gpu_thread():
        """Thread de monitoring GPU."""
        while not stop_monitoring.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] GPU Status:")
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) == 4:
                        gpu_id, util, mem_used, mem_total = parts
                        print(f"  GPU {gpu_id}: {util}% usage | VRAM: {mem_used}/{mem_total} MB")

            except Exception:
                pass

            time.sleep(2)

    # Démarrer le thread de monitoring
    monitor_thread = threading.Thread(target=monitor_gpu_thread, daemon=True)
    monitor_thread.start()

    # Lancer l'inférence
    print("\n🚀 Lancement inférence...")

    try:
        start_time = time.time()

        cmd = ["ollama", "run", model_name, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)

        elapsed = time.time() - start_time

        print(f"\n✅ Inférence terminée en {elapsed:.1f}s")
        print("\n📝 Réponse:")
        print(result.stdout[:500])

        return True

    except subprocess.TimeoutExpired:
        print("❌ Timeout lors de l'inférence (>60s)")
        return False

    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'inférence:")
        print(e.stderr)
        return False

    finally:
        # Arrêter le monitoring
        stop_monitoring.set()
        time.sleep(1)


def set_environment_variables(num_gpu: int):
    """
    Configure les variables d'environnement pour multi-GPU Ollama.

    Args:
        num_gpu: Nombre de GPUs à utiliser
    """
    print("\n⚙️  Configuration variables d'environnement...")

    # CUDA_VISIBLE_DEVICES : tous les GPUs CUDA (0 = RTX 5080, 1 = RTX 2060)
    # Priorité : GPU 0 (plus puissante) en premier
    gpu_ids = ",".join(str(i) for i in range(num_gpu))

    env_vars = {
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "OLLAMA_NUM_GPU": str(num_gpu),
        "OLLAMA_GPU_OVERHEAD": "0",  # Désactiver overhead pour maximiser VRAM
        "OLLAMA_MAX_LOADED_MODELS": "1",  # Un seul modèle à la fois
    }

    print("\n   Variables à définir dans votre environnement:")
    for key, value in env_vars.items():
        print(f"   export {key}={value}")
        os.environ[key] = value  # Définir pour cette session

    print("\n💡 Pour rendre permanent, ajoutez ces lignes à ~/.bashrc ou au script de lancement")

    return env_vars


def main():
    """Exécute la configuration complète."""
    parser = argparse.ArgumentParser(description="Configurer Ollama multi-GPU")
    parser.add_argument("--test", action="store_true", help="Tester l'inférence après configuration")
    parser.add_argument("--num-gpu", type=int, default=2, help="Nombre de GPUs à utiliser (défaut: 2)")
    parser.add_argument("--model-base", type=str, default="llama3.3:70b-instruct-q4_K_M", help="Modèle de base Ollama")

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 CONFIGURATION OLLAMA MULTI-GPU")
    print("=" * 80)

    # Étape 1: Vérifier nvidia-smi
    if not check_nvidia_smi():
        print("\n❌ Impossible de continuer sans CUDA/nvidia-smi")
        return 1

    # Étape 2: Compter les GPUs
    gpu_count = get_gpu_count()
    print(f"\n📊 {gpu_count} GPU(s) détecté(s)")

    if gpu_count == 0:
        print("❌ Aucun GPU détecté")
        return 1

    num_gpu = min(args.num_gpu, gpu_count)
    print(f"   Utilisation de {num_gpu} GPU(s)")

    # Étape 3: Variables d'environnement
    set_environment_variables(num_gpu)

    # Étape 4: Créer Modelfile
    modelfile_path = create_modelfile_multigpu(
        model_base=args.model_base,
        num_gpu=num_gpu,
    )

    # Étape 5: Builder le modèle
    model_name = f"llama3.3-70b-{num_gpu}gpu"
    success = build_model_from_modelfile(modelfile_path, model_name)

    if not success:
        print("\n❌ Échec de la création du modèle")
        return 1

    # Étape 6: Test optionnel
    if args.test:
        success = monitor_gpu_during_inference(model_name)
        if not success:
            print("\n❌ Test d'inférence échoué")
            return 1

    print("\n" + "=" * 80)
    print("✅ CONFIGURATION TERMINÉE")
    print("=" * 80)
    print(f"\n🎯 Modèle créé: {model_name}")
    print(f"   GPUs utilisés: {num_gpu}")
    print("\n💡 Utilisez ce modèle dans vos backtests en modifiant model_config.py:")
    print(f'   "critic": ["{model_name}"]')

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
