"""Test httpx dans un contexte multiprocessing (simulate Streamlit)."""
import sys
from concurrent.futures import ProcessPoolExecutor


def test_llm_client_in_worker():
    """Teste la création du client LLM dans un worker process."""
    try:
        from agents.llm_client import OllamaClient, LLMConfig, LLMProvider

        config = LLMConfig(provider=LLMProvider.OLLAMA, model='llama3.2')
        OllamaClient(config)

        return "✅ SUCCESS: OllamaClient créé dans worker process"
    except Exception as e:
        return f"❌ ERREUR: {type(e).__name__}: {e}"


def test_gpu_in_worker():
    """Teste CuPy dans un worker process."""
    try:
        from performance.device_backend import ArrayBackend

        backend = ArrayBackend()
        info = f"Device: {backend.device_type}"
        if backend.gpu_available:
            info += f" | GPU: {backend.device_info.name}"

        return f"✅ SUCCESS: {info}"
    except Exception as e:
        return f"❌ ERREUR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: LLM Client dans worker process")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(test_llm_client_in_worker)
        result1 = future1.result()
        print(result1)

    print("\n" + "=" * 60)
    print("TEST 2: GPU Backend dans worker process")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=2) as executor:
        future2 = executor.submit(test_gpu_in_worker)
        result2 = future2.result()
        print(result2)

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    if "SUCCESS" in result1 and "SUCCESS" in result2:
        print("✅ Tous les tests passés")
        sys.exit(0)
    else:
        print("❌ Certains tests ont échoué")
        sys.exit(1)
