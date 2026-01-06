"""Test de création BacktestEngine dans worker process (simule Streamlit)."""
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from agents.integration import create_optimizer_from_engine  # noqa: E402
from agents.llm_client import LLMConfig, LLMProvider  # noqa: E402
from data.loader import load_ohlcv  # noqa: E402


def worker_create_optimizer():
    """Simule l'appel Streamlit dans un worker process."""
    try:
        # Charger données (comme dans l'UI)
        df = load_ohlcv("BTCUSDC", "30m")

        # Créer config LLM
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="deepseek-r1:70b",
        )

        # Créer optimiseur (ligne qui crash dans l'UI)
        strategist, executor = create_optimizer_from_engine(
            llm_config=llm_config,
            strategy_name="ema_cross",
            data=df,
            initial_capital=10000,
            use_walk_forward=True,
            verbose=True,
        )

        return "✅ SUCCESS: Optimizer créé avec succès"

    except Exception as e:
        import traceback
        return f"❌ ERREUR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: create_optimizer_from_engine dans ProcessPoolExecutor")
    print("=" * 70)

    # Test direct
    print("\n1. Test DIRECT (main process):")
    result = worker_create_optimizer()
    print(result)

    # Test dans worker
    print("\n2. Test dans WORKER PROCESS:")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker_create_optimizer)
        result = future.result()
        print(result)

    print("\n" + "=" * 70)
    if "SUCCESS" in result:
        print("✅ Tous les tests passés")
    else:
        print("❌ Test échoué")
