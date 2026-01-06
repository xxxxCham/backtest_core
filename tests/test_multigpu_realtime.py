#!/usr/bin/env python3
"""
Test du multi-GPU Ollama avec monitoring temps réel.

Lance une inférence et monitore l'utilisation des 2 GPUs en parallèle.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def monitor_gpu(duration_s: int = 60, interval_s: float = 1.0):
    """
    Monitore l'utilisation GPU pendant une durée donnée.

    Args:
        duration_s: Durée de monitoring en secondes
        interval_s: Intervalle entre checks
    """
    print("\n" + "=" * 80)
    print("📊 MONITORING GPU EN TEMPS RÉEL")
    print("=" * 80)
    print(f"Durée: {duration_s}s | Intervalle: {interval_s}s\n")

    start_time = time.time()
    max_usage_gpu0 = 0
    max_usage_gpu1 = 0
    max_vram_gpu0 = 0
    max_vram_gpu1 = 0

    while time.time() - start_time < duration_s:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )

            elapsed = time.time() - start_time
            timestamp = time.strftime("%H:%M:%S")

            lines = result.stdout.strip().split('\n')

            if len(lines) >= 2:
                # GPU 0
                parts0 = [x.strip() for x in lines[0].split(',')]
                gpu0_util = int(parts0[1])
                gpu0_vram = int(parts0[2])
                gpu0_total = int(parts0[3])

                # GPU 1
                parts1 = [x.strip() for x in lines[1].split(',')]
                gpu1_util = int(parts1[1])
                gpu1_vram = int(parts1[2])
                gpu1_total = int(parts1[3])

                # Mettre à jour max
                max_usage_gpu0 = max(max_usage_gpu0, gpu0_util)
                max_usage_gpu1 = max(max_usage_gpu1, gpu1_util)
                max_vram_gpu0 = max(max_vram_gpu0, gpu0_vram)
                max_vram_gpu1 = max(max_vram_gpu1, gpu1_vram)

                # Affichage coloré
                gpu0_color = "🟢" if gpu0_util > 10 else "🔴"
                gpu1_color = "🟢" if gpu1_util > 10 else "🔴"

                print(
                    f"[{timestamp}] {elapsed:5.1f}s | "
                    f"GPU0 {gpu0_color} {gpu0_util:3d}% ({gpu0_vram:5d}/{gpu0_total:5d}MB) | "
                    f"GPU1 {gpu1_color} {gpu1_util:3d}% ({gpu1_vram:5d}/{gpu1_total:5d}MB)"
                )

        except Exception as e:
            print(f"❌ Erreur monitoring: {e}")

        time.sleep(interval_s)

    # Résumé
    print("\n" + "=" * 80)
    print("📈 RÉSUMÉ DU MONITORING")
    print("=" * 80)
    print("\nGPU 0 (RTX 5080):")
    print(f"  Max Usage: {max_usage_gpu0}%")
    print(f"  Max VRAM: {max_vram_gpu0} MB")

    print("\nGPU 1 (RTX 2060 SUPER):")
    print(f"  Max Usage: {max_usage_gpu1}%")
    print(f"  Max VRAM: {max_vram_gpu1} MB")

    # Verdict
    print("\n" + "=" * 80)
    if max_usage_gpu1 > 5 and max_vram_gpu1 > 100:
        print("✅ MULTI-GPU FONCTIONNE : Les 2 GPUs sont utilisés !")
    elif max_usage_gpu0 > 5:
        print("⚠️  MONO-GPU : Seul GPU 0 est utilisé")
        print("\n💡 Solution:")
        print("   1. Redémarrer Ollama: D:\\backtest_core\\Start-OllamaMultiGPU.ps1")
        print("   2. Vérifier Modelfile: ollama show llama3.3-70b-2gpu --modelfile")
    else:
        print("❌ AUCUN GPU UTILISÉ : Ollama n'est peut-être pas en cours d'exécution")

    print("=" * 80)

    return max_usage_gpu1 > 5


def run_inference(model: str = "llama3.3-70b-2gpu", prompt: str = "Explain reinforcement learning in 100 words"):
    """
    Lance une inférence Ollama.

    Args:
        model: Nom du modèle Ollama
        prompt: Prompt pour l'inférence
    """
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT INFÉRENCE")
    print("=" * 80)
    print(f"Modèle: {model}")
    print(f"Prompt: {prompt[:60]}...")

    try:
        start_time = time.time()

        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Inférence terminée en {elapsed:.1f}s")
        print("\n📝 Réponse:")
        print(result.stdout[:500])

        return True, elapsed

    except subprocess.TimeoutExpired:
        print("❌ Timeout (>90s)")
        return False, 90

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {e.stderr}")
        return False, 0


def main():
    """Exécute le test complet."""
    print("=" * 80)
    print("🧪 TEST MULTI-GPU OLLAMA - MONITORING TEMPS RÉEL")
    print("=" * 80)

    # Thread de monitoring
    monitoring_duration = 60  # 60 secondes max

    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(monitoring_duration, 1.0),
        daemon=True,
    )

    # Démarrer monitoring
    monitor_thread.start()

    # Attendre 2s pour que le monitoring démarre
    time.sleep(2)

    # Lancer inférence
    success, elapsed = run_inference(
        model="llama3.3-70b-2gpu",
        prompt="Explain what is reinforcement learning in 100 words. Be concise and precise.",
    )

    # Attendre la fin du monitoring
    monitor_thread.join(timeout=5)

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
