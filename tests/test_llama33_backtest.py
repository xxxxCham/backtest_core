#!/usr/bin/env python3
"""
Script de test: Llama 3.3 70B dans un backtest réel avec agents.

Objectif:
- Tester l'intégration du modèle Llama 3.3 70B dans un workflow de backtest complet
- Mesurer les temps de réponse réels pour analyses complexes
- Vérifier la distribution GPU pendant l'inférence
- Valider le comportement du modèle comme Critic

Usage:
    python tools/test_llama33_backtest.py

Options:
    --iterations N      Nombre max d'itérations (défaut: 3)
    --force-heavy       Forcer l'utilisation de Llama 3.3 dès iter=1
    --strategy NAME     Stratégie à tester (défaut: ema_cross)
    --monitor-gpu       Monitorer le GPU en continu
"""

import argparse
import os
import sys
import time

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from agents.integration import create_orchestrator_with_backtest
from agents.model_config import RoleModelConfig
from agents.orchestration_logger import OrchestrationLogger, generate_session_id
from agents.orchestrator import Orchestrator


def create_sample_data(n_bars: int = 2000, volatility: float = 0.02) -> pd.DataFrame:
    """
    Crée des données OHLCV synthétiques réalistes.

    Args:
        n_bars: Nombre de barres à générer
        volatility: Volatilité des prix (0.02 = 2% par barre)

    Returns:
        DataFrame OHLCV avec index datetime
    """
    print(f"📊 Génération de {n_bars} barres OHLCV...")

    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')

    # Prix avec tendance et bruit
    trend = np.linspace(100, 120, n_bars)  # Tendance haussière modérée
    noise = np.cumsum(np.random.randn(n_bars) * volatility * 100)
    close = trend + noise

    # OHLC réalistes
    high = close + np.abs(np.random.randn(n_bars)) * volatility * 100
    low = close - np.abs(np.random.randn(n_bars)) * volatility * 100
    open_price = close + (np.random.randn(n_bars) * volatility * 50)
    volume = np.random.randint(10000, 100000, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)

    print(f"   Période: {df.index[0]} → {df.index[-1]}")
    print(f"   Prix: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    print(f"   Range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    return df


def setup_orchestrator(
    strategy_name: str,
    data: pd.DataFrame,
    max_iterations: int,
    force_heavy: bool,
) -> tuple[Orchestrator, OrchestrationLogger, RoleModelConfig]:
    """
    Configure l'orchestrateur avec Llama 3.3 pour le Critic.

    Args:
        strategy_name: Nom de la stratégie à tester
        data: DataFrame OHLCV
        max_iterations: Nombre max d'itérations
        force_heavy: Si True, utilise Llama 3.3 dès iter=1

    Returns:
        Tuple (orchestrator, logger, model_config)
    """
    print("\n" + "="*80)
    print("🔧 CONFIGURATION ORCHESTRATEUR")
    print("="*80)

    # Session ID unique
    session_id = generate_session_id()
    print(f"📋 Session ID: {session_id}")

    # Logger d'orchestration
    logger = OrchestrationLogger(session_id=session_id)

    # Configuration des modèles
    model_config = RoleModelConfig()

    # Vérifier que llama3.3-70b-optimized est disponible
    installed = model_config.get_installed_models()
    if "llama3.3-70b-optimized" not in installed:
        print("❌ llama3.3-70b-optimized non installé !")
        print("   Lancez: python tools/setup_llama33_70b.py")
        sys.exit(1)

    print("✅ llama3.3-70b-optimized détecté")

    # FORCER l'utilisation de Llama 3.3 pour le Critic (test)
    print("🦙 Configuration Critic pour utiliser UNIQUEMENT Llama 3.3 70B...")
    model_config.critic.models = ["llama3.3-70b-optimized"]
    model_config.critic.allow_heavy_after_iteration = 0  # Toujours autoriser

    # Afficher la sélection des modèles pour chaque rôle
    print("\n📌 Modèles sélectionnés par rôle:")
    for iteration in [1, 2, 3]:
        print(f"\n  Itération {iteration}:")
        for role in ["analyst", "strategist", "critic", "validator"]:
            model = model_config.get_model(
                role=role,
                iteration=iteration,
                allow_heavy=force_heavy,
            )
            emoji = "🦙" if "llama3.3" in (model or "") else "🤖"
            print(f"    {emoji} {role.capitalize()}: {model or 'None'}")

    # Paramètres initiaux pour la stratégie
    if strategy_name == "ema_cross":
        initial_params = {
            "fast_period": 12,
            "slow_period": 26,
        }
    elif strategy_name == "bollinger_atr":
        initial_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "atr_threshold": 1.5,
        }
    else:
        # Paramètres par défaut
        initial_params = {}

    print(f"\n📊 Stratégie: {strategy_name}")
    print(f"   Paramètres initiaux: {initial_params}")

    # Créer l'orchestrateur
    print("\n🚀 Création de l'orchestrateur...")
    orchestrator = create_orchestrator_with_backtest(
        strategy_name=strategy_name,
        data=data,
        initial_params=initial_params,
        data_symbol="SYNTHETIC",
        data_timeframe="1H",
        role_model_config=model_config,
        orchestration_logger=logger,
        session_id=session_id,
        max_iterations=max_iterations,
        initial_capital=10000.0,
        use_walk_forward=False,  # Désactiver WF pour test rapide
        n_workers=1,
    )

    print("✅ Orchestrateur créé")

    return orchestrator, logger, model_config


def monitor_gpu_stats():
    """Affiche les stats GPU si disponible."""
    try:
        from performance.gpu import get_gpu_info

        info = get_gpu_info()

        if not info.get("cupy_available", False):
            return None

        vram_total = info.get("cupy_memory_total_gb", 0)
        vram_free = info.get("cupy_memory_free_gb", 0)
        vram_used = vram_total - vram_free

        return {
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "vram_used_gb": vram_used,
            "device": info.get("cupy_device_name", "Unknown"),
        }
    except Exception:
        return None


def run_backtest_with_monitoring(
    orchestrator: Orchestrator,
    logger: OrchestrationLogger,
    monitor_gpu: bool = False,
) -> dict:
    """
    Lance le backtest et monitore les performances.

    Args:
        orchestrator: Orchestrateur configuré
        logger: Logger d'orchestration
        monitor_gpu: Si True, affiche les stats GPU périodiquement

    Returns:
        Résultat du backtest avec métriques de performance
    """
    print("\n" + "="*80)
    print("🚀 LANCEMENT DU BACKTEST AVEC AGENTS")
    print("="*80)

    # Stats GPU initiales
    if monitor_gpu:
        gpu_before = monitor_gpu_stats()
        if gpu_before:
            print("\n📊 GPU avant backtest:")
            print(f"   Device: {gpu_before['device']}")
            print(f"   VRAM: {gpu_before['vram_used_gb']:.2f}/{gpu_before['vram_total_gb']:.2f} GB")

    # Lancer le backtest
    start_time = time.perf_counter()

    try:
        result = orchestrator.run()
        elapsed = time.perf_counter() - start_time

        print(f"\n✅ Backtest terminé en {elapsed:.1f}s")

        # Stats GPU finales
        if monitor_gpu:
            gpu_after = monitor_gpu_stats()
            if gpu_after:
                print("\n📊 GPU après backtest:")
                print(f"   VRAM: {gpu_after['vram_used_gb']:.2f}/{gpu_after['vram_total_gb']:.2f} GB")
                delta_vram = gpu_after['vram_used_gb'] - gpu_before['vram_used_gb']
                print(f"   Δ VRAM: {delta_vram:+.2f} GB")

        # Résumé des résultats
        print("\n" + "="*80)
        print("📋 RÉSUMÉ DES RÉSULTATS")
        print("="*80)

        print(f"\nDécision finale: {result.decision}")
        print(f"État final: {result.final_state.value}")
        print(f"Succès: {'✅' if result.success else '❌'}")

        print("\n📊 Métriques d'exécution:")
        print(f"   Itérations: {result.total_iterations}")
        print(f"   Backtests: {result.total_backtests}")
        print(f"   Temps total: {result.total_time_s:.1f}s")
        print(f"   Appels LLM: {result.total_llm_calls}")
        print(f"   Tokens LLM: {result.total_llm_tokens:,}")

        if result.final_metrics:
            print("\n📈 Métriques finales:")
            print(f"   Sharpe: {result.final_metrics.sharpe_ratio:.2f}")
            print(f"   Return: {result.final_metrics.total_return*100:.1f}%")
            print(f"   Max DD: {result.final_metrics.max_drawdown*100:.1f}%")
            print(f"   Trades: {result.final_metrics.total_trades}")

        print("\n📝 Paramètres finaux:")
        for key, value in result.final_params.items():
            print(f"   {key}: {value}")

        # Logs sauvegardés
        log_path = logger.save_to_file()
        print(f"\n💾 Logs sauvegardés: {log_path}")

        return {
            "success": result.success,
            "elapsed_s": elapsed,
            "iterations": result.total_iterations,
            "llm_calls": result.total_llm_calls,
            "llm_tokens": result.total_llm_tokens,
            "final_params": result.final_params,
            "final_metrics": result.final_metrics,
            "log_path": log_path,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ Erreur après {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "elapsed_s": elapsed,
            "error": str(e),
        }


def main():
    """Exécute le test complet."""
    parser = argparse.ArgumentParser(description="Test Llama 3.3 70B dans un backtest réel")
    parser.add_argument("--iterations", type=int, default=3, help="Nombre max d'itérations")
    parser.add_argument("--force-heavy", action="store_true", help="Forcer Llama 3.3 dès iter=1")
    parser.add_argument("--strategy", type=str, default="ema_cross", help="Stratégie à tester")
    parser.add_argument("--monitor-gpu", action="store_true", help="Monitorer le GPU")
    parser.add_argument("--n-bars", type=int, default=2000, help="Nombre de barres OHLCV")

    args = parser.parse_args()

    print("="*80)
    print("🦙 TEST LLAMA 3.3 70B - BACKTEST RÉEL AVEC AGENTS")
    print("="*80)
    print("\nConfiguration:")
    print(f"   Stratégie: {args.strategy}")
    print(f"   Max itérations: {args.iterations}")
    print(f"   Force heavy: {'Oui' if args.force_heavy else 'Non (auto à iter>=2)'}")
    print(f"   Monitor GPU: {'Oui' if args.monitor_gpu else 'Non'}")
    print(f"   Barres OHLCV: {args.n_bars}")

    # Étape 1: Générer les données
    data = create_sample_data(n_bars=args.n_bars)

    # Étape 2: Configurer l'orchestrateur
    orchestrator, logger, model_config = setup_orchestrator(
        strategy_name=args.strategy,
        data=data,
        max_iterations=args.iterations,
        force_heavy=args.force_heavy,
    )

    # Étape 3: Lancer le backtest
    result = run_backtest_with_monitoring(
        orchestrator=orchestrator,
        logger=logger,
        monitor_gpu=args.monitor_gpu,
    )

    # Étape 4: Résumé final
    print("\n" + "="*80)
    if result["success"]:
        print("✅ TEST RÉUSSI")
        print("="*80)
        print("\n💡 Le modèle Llama 3.3 70B fonctionne correctement dans un backtest réel !")
        print("\n📊 Performance:")
        print(f"   Temps total: {result['elapsed_s']:.1f}s")
        print(f"   Itérations: {result['iterations']}")
        print(f"   Appels LLM: {result['llm_calls']}")

        if result['llm_calls'] > 0:
            avg_time_per_call = result['elapsed_s'] / result['llm_calls']
            print(f"   Temps moyen/appel: {avg_time_per_call:.1f}s")
    else:
        print("❌ TEST ÉCHOUÉ")
        print("="*80)
        if "error" in result:
            print(f"\nErreur: {result['error']}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
