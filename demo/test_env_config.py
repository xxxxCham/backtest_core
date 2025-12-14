"""
Exemple d'utilisation des variables d'environnement pour contrôler le comportement
du moteur de backtest.

Ce script illustre comment configurer les variables d'env pour différents scénarios :
1. CPU-only (défaut)
2. GPU avec optimisation mémoire
3. Configuration LLM personnalisée
"""

import os
import sys
from pathlib import Path

# Ajouter le module au path
sys.path.insert(0, str(Path(__file__).parent))


def example_cpu_only():
    """Configuration recommandée pour systèmes CPU-only."""
    print("=== Configuration CPU-only (défaut) ===\n")
    
    # Ne PAS décharger le LLM (évite latence)
    os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'False'
    
    # LLM léger pour CPU
    os.environ['BACKTEST_LLM_PROVIDER'] = 'ollama'
    os.environ['BACKTEST_LLM_MODEL'] = 'deepseek-r1:8b'
    os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
    
    # Logging standard
    os.environ['BACKTEST_LOG_LEVEL'] = 'INFO'
    
    print("✅ Configuration CPU-only activée")
    print(f"   UNLOAD_LLM_DURING_BACKTEST: {os.getenv('UNLOAD_LLM_DURING_BACKTEST')}")
    print(f"   BACKTEST_LLM_MODEL: {os.getenv('BACKTEST_LLM_MODEL')}")
    print()


def example_gpu_optimized():
    """Configuration recommandée pour systèmes GPU avec CuPy."""
    print("=== Configuration GPU Optimisée ===\n")
    
    # ACTIVER le déchargement LLM (libère 100% VRAM)
    os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'True'
    
    # LLM plus lourd possible avec GPU libre
    os.environ['BACKTEST_LLM_PROVIDER'] = 'ollama'
    os.environ['BACKTEST_LLM_MODEL'] = 'deepseek-r1:32b'
    
    # Utiliser GPU pour calculs
    os.environ['USE_GPU'] = 'true'
    
    # Logging détaillé pour profiling
    os.environ['BACKTEST_LOG_LEVEL'] = 'DEBUG'
    
    print("✅ Configuration GPU activée")
    print(f"   UNLOAD_LLM_DURING_BACKTEST: {os.getenv('UNLOAD_LLM_DURING_BACKTEST')}")
    print(f"   BACKTEST_LLM_MODEL: {os.getenv('BACKTEST_LLM_MODEL')}")
    print(f"   USE_GPU: {os.getenv('USE_GPU')}")
    print()


def example_openai():
    """Configuration avec OpenAI au lieu d'Ollama."""
    print("=== Configuration OpenAI ===\n")
    
    # Pas de GPU unload (OpenAI est cloud)
    os.environ['UNLOAD_LLM_DURING_BACKTEST'] = 'False'
    
    # Provider OpenAI
    os.environ['BACKTEST_LLM_PROVIDER'] = 'openai'
    os.environ['BACKTEST_LLM_MODEL'] = 'gpt-4'
    os.environ['OPENAI_API_KEY'] = 'sk-...'  # Remplacer par vraie clé
    
    # Température pour créativité contrôlée
    os.environ['BACKTEST_LLM_TEMPERATURE'] = '0.7'
    os.environ['BACKTEST_LLM_MAX_TOKENS'] = '2000'
    
    print("✅ Configuration OpenAI activée")
    print(f"   BACKTEST_LLM_PROVIDER: {os.getenv('BACKTEST_LLM_PROVIDER')}")
    print(f"   BACKTEST_LLM_MODEL: {os.getenv('BACKTEST_LLM_MODEL')}")
    print(f"   OPENAI_API_KEY: {'*' * 20}...{os.getenv('OPENAI_API_KEY', '')[-4:]}")
    print()


def example_walk_forward():
    """Configuration pour validation walk-forward stricte."""
    print("=== Configuration Walk-Forward Validation ===\n")
    
    # Validation agressive avec 10 fenêtres
    os.environ['WALK_FORWARD_WINDOWS'] = '10'
    os.environ['WALK_FORWARD_MIN_TEST_SAMPLES'] = '100'
    
    # Limite d'overfitting stricte
    os.environ['MAX_OVERFITTING_RATIO'] = '1.3'
    
    print("✅ Configuration Walk-Forward activée")
    print(f"   WALK_FORWARD_WINDOWS: {os.getenv('WALK_FORWARD_WINDOWS')}")
    print(f"   WALK_FORWARD_MIN_TEST_SAMPLES: {os.getenv('WALK_FORWARD_MIN_TEST_SAMPLES')}")
    print(f"   MAX_OVERFITTING_RATIO: {os.getenv('MAX_OVERFITTING_RATIO')}")
    print()


def test_current_config():
    """Affiche la configuration actuelle."""
    print("=== Configuration Actuelle ===\n")
    
    # Variables critiques
    critical_vars = [
        'UNLOAD_LLM_DURING_BACKTEST',
        'BACKTEST_LLM_PROVIDER',
        'BACKTEST_LLM_MODEL',
        'BACKTEST_LOG_LEVEL',
        'USE_GPU',
    ]
    
    for var in critical_vars:
        value = os.getenv(var, '(non définie)')
        print(f"   {var}: {value}")
    
    print()
    
    # Avertissements
    if os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'False').lower() == 'true':
        if os.getenv('USE_GPU', 'false').lower() == 'false':
            print("⚠️  WARNING: UNLOAD_LLM_DURING_BACKTEST=True sans GPU détecté")
            print("   → Cela ajoutera de la latence sans bénéfice")
            print("   → Recommandé: UNLOAD_LLM_DURING_BACKTEST=False")
            print()


def run_quick_backtest():
    """Lance un backtest rapide avec la config actuelle."""
    print("=== Test Rapide Backtest ===\n")
    
    try:
        from agents import create_optimizer_from_engine
        from agents.llm_client import LLMConfig
        from data.loader import load_ohlcv
        import pandas as pd
        
        # Créer des données de test
        print("Génération données de test...")
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000.0
        })
        
        # Charger config LLM
        config = LLMConfig.from_env()
        print(f"LLM Provider: {config.provider}")
        print(f"LLM Model: {config.model}")
        
        # Vérifier variable GPU unload
        unload = os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'False').lower() == 'true'
        print(f"GPU Unload: {unload}")
        
        print("\n✅ Configuration valide, prêt pour backtest")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test des configurations d'environnement")
    parser.add_argument('--scenario', choices=['cpu', 'gpu', 'openai', 'walk-forward', 'current', 'test'],
                        default='current', help="Scénario de configuration à tester")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Backtest Core - Test Configuration Environnement")
    print("=" * 60)
    print()
    
    if args.scenario == 'cpu':
        example_cpu_only()
    elif args.scenario == 'gpu':
        example_gpu_optimized()
    elif args.scenario == 'openai':
        example_openai()
    elif args.scenario == 'walk-forward':
        example_walk_forward()
    elif args.scenario == 'current':
        test_current_config()
    elif args.scenario == 'test':
        test_current_config()
        run_quick_backtest()
    
    print("=" * 60)
    print("Pour appliquer ces configurations :")
    print("  1. Copier .env.example vers .env")
    print("  2. Éditer .env avec vos valeurs")
    print("  3. Ou définir directement : $env:VARIABLE=value (PowerShell)")
    print()
    print("Documentation complète : ENVIRONMENT.md")
    print("=" * 60)
