"""
Exemple de sweep optimisé pour système avec 30 GB RAM.

Ce script démontre comment configurer et lancer un sweep de 2,4M combinaisons
avec cache RAM optimisé (100k entries) et GPU queue batching.

Gains attendus:
- Cache hit rate: 95%+ après warmup (~100-200 premiers backtests)
- Throughput: 1000-1500 bt/sec (vs 50 bt/sec baseline)
- Temps total 2.4M: 30-45 minutes (vs ~13 heures sans optimisation)

Requirements:
- 30+ GB RAM
- GPU CUDA compatible (RTX 4090/5080 recommandé)
- CuPy installé: pip install cupy-cuda12x
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1: CONFIGURATION OPTIMISÉE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Exécute un sweep avec configuration RAM optimisée."""

    # Import des configs
    try:
        from config.gpu_config_30gb_ram import (
            get_gpu_config,
            get_indicator_cache_config,
            get_worker_config,
            print_memory_breakdown
        )
    except ImportError as e:
        logger.error(f"Erreur import config: {e}")
        logger.error("Assurez-vous que config/gpu_config_30gb_ram.py existe")
        sys.exit(1)

    # Afficher le breakdown mémoire
    logger.info("=" * 80)
    logger.info("SWEEP OPTIMISÉ 30 GB RAM - Configuration".center(80))
    logger.info("=" * 80)
    print_memory_breakdown()

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2: CHARGER LES DONNÉES
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("Chargement des données...")

    try:
        from data.loader import load_data
        import pandas as pd

        # Charger les données (exemple: BTCUSDC 30m, 2 ans)
        df = load_data(
            symbol="BTCUSDC",
            timeframe="30m",
            start_date="2024-01-01",
            end_date="2026-01-31"
        )

        logger.info(f"Données chargées: {len(df):,} candles ({df.index[0]} → {df.index[-1]})")

    except Exception as e:
        logger.error(f"Erreur chargement données: {e}")
        logger.error("Utilisation de données de démo...")

        # Fallback: données de démo
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=730),
            end=datetime.now(),
            freq="30min"
        )
        df = pd.DataFrame({
            "open": np.random.uniform(40000, 50000, len(dates)),
            "high": np.random.uniform(40000, 50000, len(dates)),
            "low": np.random.uniform(40000, 50000, len(dates)),
            "close": np.random.uniform(40000, 50000, len(dates)),
            "volume": np.random.uniform(100, 1000, len(dates)),
        }, index=dates)

        logger.info(f"Données démo générées: {len(df):,} candles")

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3: DÉFINIR LA GRILLE DE PARAMÈTRES
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("Configuration de la grille de paramètres...")

    # Exemple: bollinger_atr avec grille dense pour 2.4M combinaisons
    param_grid = {
        # Bollinger Bands (200 variantes)
        "bb_period": list(range(15, 35, 1)),    # 20 valeurs
        "bb_std": [round(x * 0.1, 1) for x in range(15, 30)],  # 15 valeurs (1.5-3.0)

        # ATR (15 variantes)
        "atr_period": list(range(10, 25, 1)),  # 15 valeurs

        # Autres paramètres
        "entry_z": [1.5, 2.0, 2.5],            # 3 valeurs
        "k_sl": [1.0, 1.5, 2.0],               # 3 valeurs
        "atr_percentile": [20, 30, 40],        # 3 valeurs

        # Levier fixé
        "leverage": [1],
    }

    # Calcul du nombre total de combinaisons
    from itertools import product
    total_combos = 1
    for param, values in param_grid.items():
        total_combos *= len(values)

    logger.info(f"Total combinaisons: {total_combos:,}")

    # Estimer les indicateurs uniques
    unique_bollinger = len(param_grid["bb_period"]) * len(param_grid["bb_std"])
    unique_atr = len(param_grid["atr_period"])
    total_unique_indicators = unique_bollinger + unique_atr

    logger.info(f"Indicateurs uniques estimés: {total_unique_indicators:,}")
    logger.info(f"  - Bollinger: {unique_bollinger:,}")
    logger.info(f"  - ATR: {unique_atr:,}")

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4: INITIALISER GPU CONTEXT + CACHE RAM
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("Initialisation GPU context + cache RAM...")

    from backtest.gpu_context import init_gpu_context, cleanup_gpu_context

    # Récupérer les configs
    gpu_cfg = get_gpu_config()
    cache_cfg = get_indicator_cache_config()
    worker_cfg = get_worker_config()

    # Initialiser GPU context avec cache RAM 100k
    gpu_initialized = init_gpu_context(
        max_batch_size=gpu_cfg["max_batch_size"],
        max_wait_ms=gpu_cfg["max_wait_ms"],
        use_cache=gpu_cfg["use_cache"],
        indicator_cache_config=cache_cfg
    )

    if not gpu_initialized:
        logger.warning("GPU context non initialisé - utilisation CPU fallback")

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 5: LANCER LE SWEEP
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("Lancement du sweep...")
    logger.info(f"Workers CPU: {worker_cfg['max_workers']}")
    logger.info(f"Thread limit: {worker_cfg['thread_limit']}")
    logger.info(f"Fast metrics: {worker_cfg['fast_metrics']}")

    try:
        from backtest.sweep import SweepEngine

        engine = SweepEngine(
            max_workers=worker_cfg["max_workers"],
            debug_enabled=worker_cfg["debug_enabled"]
        )

        # Lancer le sweep avec fast_metrics
        results = engine.run_sweep(
            df=df,
            strategy="bollinger_atr",
            param_grid=param_grid,
            symbol="BTCUSDC",
            timeframe="30m",
            initial_capital=10000.0,
            fast_metrics=worker_cfg["fast_metrics"]
        )

        # Analyser les résultats
        logger.info("=" * 80)
        logger.info(f"Sweep terminé: {len(results):,} résultats")

        if results:
            # Top 10 configs
            top_10 = sorted(results, key=lambda x: x.get("sharpe", 0.0), reverse=True)[:10]

            logger.info("\nTop 10 configs (par Sharpe):")
            for i, result in enumerate(top_10, 1):
                params = result.get("params_dict", {})
                sharpe = result.get("sharpe", 0.0)
                total_pnl = result.get("total_pnl", 0.0)
                win_rate = result.get("win_rate", 0.0)

                logger.info(
                    f"  #{i}: Sharpe={sharpe:.2f}, PnL={total_pnl:.2f}, "
                    f"WR={win_rate:.1f}%, "
                    f"bb_period={params.get('bb_period')}, "
                    f"bb_std={params.get('bb_std')}, "
                    f"atr_period={params.get('atr_period')}"
                )

        else:
            logger.warning("Aucun résultat valide")

    except KeyboardInterrupt:
        logger.warning("Sweep interrompu par l'utilisateur")

    except Exception as e:
        logger.error(f"Erreur pendant le sweep: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ═══════════════════════════════════════════════════════════════════
        # ÉTAPE 6: CLEANUP + STATS
        # ═══════════════════════════════════════════════════════════════════

        logger.info("Nettoyage GPU context...")

        from backtest.gpu_context import get_gpu_context_stats

        # Récupérer et afficher les stats GPU
        if gpu_initialized:
            try:
                gpu_stats = get_gpu_context_stats()

                logger.info("=" * 80)
                logger.info("GPU STATS FINALES".center(80))
                logger.info("=" * 80)

                if gpu_stats.get("enabled"):
                    total_req = gpu_stats.get("total_requests", 0)
                    cache_hits = gpu_stats.get("cache_hits", 0)
                    gpu_computes = gpu_stats.get("gpu_computes", 0)
                    hit_rate = gpu_stats.get("cache_hit_rate_pct", 0.0)
                    batches = gpu_stats.get("total_batches", 0)
                    avg_batch = gpu_stats.get("avg_batch_size", 0.0)

                    logger.info(f"Total requêtes: {total_req:,}")
                    logger.info(f"Cache hits: {cache_hits:,} ({hit_rate:.1f}%)")
                    logger.info(f"GPU computes: {gpu_computes:,}")
                    logger.info(f"Batches traités: {batches:,}")
                    logger.info(f"Taille batch moyenne: {avg_batch:.1f}")

                else:
                    logger.info("GPU queue désactivé")

            except Exception as e:
                logger.warning(f"Erreur récupération stats: {e}")

        # Cleanup GPU context
        cleanup_gpu_context()
        logger.info("GPU context nettoyé")

        # Cleanup IndicatorBank stats
        try:
            from data.indicator_bank import get_indicator_bank

            bank = get_indicator_bank()
            cache_stats = bank.get_stats()

            logger.info("=" * 80)
            logger.info("INDICATOR CACHE STATS FINALES".center(80))
            logger.info("=" * 80)

            logger.info(f"Cache hits: {cache_stats.hits:,}")
            logger.info(f"Cache misses: {cache_stats.misses:,}")
            logger.info(f"Hit rate: {cache_stats.hit_rate:.1f}%")
            logger.info(f"Entries en mémoire: {cache_stats.entries_count:,}")
            logger.info(f"Taille cache: {cache_stats.total_size_mb:.1f} MB")

        except Exception as e:
            logger.warning(f"Erreur récupération cache stats: {e}")

        logger.info("=" * 80)
        logger.info("SWEEP TERMINÉ")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()