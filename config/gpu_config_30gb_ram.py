"""
Configuration GPU optimisée pour système avec 30 GB RAM.

Cette configuration maximise l'utilisation de la RAM pour cacher les indicateurs
calculés, évitant ainsi les recalculs coûteux pendant un sweep de 2,4M combinaisons.

Architecture:
- Cache RAM: 100 000 entrées (~25 GB utilisé)
- Cache Disque: Désactivé (disk_enabled=False) pour performance max
- GPU Queue: Batch size 50, wait 50ms
- Workers CPU: 24-28 parallèles

Gains attendus:
- Cache hit rate: 95%+ après warmup
- Throughput: 1000-1500 bt/sec
- Temps total sweep 2.4M: ~30-45 min (vs 13h sans cache)
"""

from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# GPU QUEUE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

GPU_QUEUE_CONFIG = {
    # Batching
    "max_batch_size": 50,       # 50 requêtes par batch (optimal pour RTX 5080)
    "max_wait_ms": 50.0,        # 50ms max wait time

    # Cache
    "use_cache": True,          # Utiliser IndicatorBank pour cache
}


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR BANK CONFIGURATION - RAM OPTIMISÉE
# ═══════════════════════════════════════════════════════════════════════════

INDICATOR_CACHE_CONFIG = {
    # Cache Directory (non utilisé si disk_enabled=False)
    "cache_dir": ".indicator_cache",

    # TTL - Court car on garde tout en RAM pendant le sweep
    "ttl": 3600 * 2,            # 2 heures (durée max d'un sweep)

    # ⚠️ DÉSACTIVER CACHE DISQUE - TOUT EN RAM
    "disk_enabled": False,      # Pas de cache disque (RAM only!)

    # 🔥 RAM CACHE - 100K ENTRIES (~25 GB)
    "memory_max_entries": 100000,  # 100 000 indicateurs en RAM

    # Taille max disque (non utilisé si disk_enabled=False)
    "max_size_mb": 0,           # Désactivé

    # Global enable
    "enabled": True,
}


# ═══════════════════════════════════════════════════════════════════════════
# WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

WORKER_CONFIG = {
    # Nombre de workers CPU (pour 32 cœurs avec 30 GB RAM)
    "max_workers": 28,          # 28 workers (laisse 4 cœurs pour GPU worker + OS)

    # Thread limit par worker (évite nested parallelism)
    "thread_limit": 1,          # 1 thread BLAS par worker (28 total)

    # Fast metrics pour sweeps
    "fast_metrics": True,       # Métriques rapides uniquement

    # Debug
    "debug_enabled": False,     # Désactiver debug pour performance
}


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY USAGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

MEMORY_BREAKDOWN = {
    "indicator_cache_ram": {
        "entries": 100000,
        "avg_size_per_entry_kb": 280,  # ATR: 280KB, Bollinger: 840KB
        "total_gb": 25.0,               # ~25 GB pour cache
        "note": "Cache RAM pour indicateurs calculés"
    },

    "worker_overhead": {
        "workers": 28,
        "dataframe_per_worker_mb": 50,  # 35k candles OHLCV
        "total_gb": 1.4,
        "note": "DataFrame + overhead par worker"
    },

    "gpu_memory": {
        "vram_gb": 16,                  # RTX 5080 VRAM
        "system_ram_for_gpu_mb": 500,
        "total_gb": 0.5,
        "note": "Transferts CPU↔GPU et buffers"
    },

    "os_overhead": {
        "total_gb": 2.0,
        "note": "Windows + processus système"
    },

    "total_estimated_gb": 28.9,
    "available_ram_gb": 30.0,
    "margin_gb": 1.1,
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_gpu_config() -> Dict[str, Any]:
    """Retourne la config GPU queue."""
    return GPU_QUEUE_CONFIG.copy()


def get_indicator_cache_config() -> Dict[str, Any]:
    """Retourne la config IndicatorBank optimisée RAM."""
    return INDICATOR_CACHE_CONFIG.copy()


def get_worker_config() -> Dict[str, Any]:
    """Retourne la config workers."""
    return WORKER_CONFIG.copy()


def print_memory_breakdown() -> None:
    """Affiche le breakdown mémoire."""
    print("\n" + "=" * 80)
    print("MEMORY BREAKDOWN - Configuration 30 GB RAM".center(80))
    print("=" * 80)

    for category, details in MEMORY_BREAKDOWN.items():
        if isinstance(details, dict) and "note" in details:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for key, value in details.items():
                if key != "note":
                    print(f"  {key}: {value}")
            print(f"  └─ {details['note']}")

    print("\n" + "-" * 80)
    total = MEMORY_BREAKDOWN["total_estimated_gb"]
    available = MEMORY_BREAKDOWN["available_ram_gb"]
    margin = MEMORY_BREAKDOWN["margin_gb"]

    print(f"Total estimé: {total:.1f} GB / {available:.1f} GB disponible")
    print(f"Marge restante: {margin:.1f} GB ({margin/available*100:.1f}%)")
    print("=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

USAGE_EXAMPLE = """
# Dans votre sweep ou ui/main.py:

from config.gpu_config_30gb_ram import (
    get_gpu_config,
    get_indicator_cache_config,
    get_worker_config,
    print_memory_breakdown
)

# 1. Afficher le breakdown mémoire
print_memory_breakdown()

# 2. Initialiser GPU context avec config optimisée
from backtest.gpu_context import init_gpu_context

gpu_cfg = get_gpu_config()
init_gpu_context(**gpu_cfg)

# 3. Configurer IndicatorBank avec cache RAM 100k
from data.indicator_bank import get_indicator_bank

cache_cfg = get_indicator_cache_config()
bank = get_indicator_bank(**cache_cfg)

# 4. Configurer le sweep avec workers optimisés
from backtest.sweep import SweepEngine

worker_cfg = get_worker_config()
engine = SweepEngine(
    max_workers=worker_cfg["max_workers"],
    debug_enabled=worker_cfg["debug_enabled"]
)

# 5. Lancer le sweep
results = engine.run_sweep(
    df=data,
    strategy="bollinger_atr",
    param_grid=param_grid,
    fast_metrics=worker_cfg["fast_metrics"]
)

# 6. Cleanup
from backtest.gpu_context import cleanup_gpu_context
cleanup_gpu_context()

# Résultat attendu:
# - Cache hit rate: 95%+ après warmup (100-200 premiers backtests)
# - Throughput: 1000-1500 bt/sec (vs 50 bt/sec sans GPU queue)
# - Temps total pour 2.4M: 30-45 minutes (vs ~13 heures)
"""

__all__ = [
    "GPU_QUEUE_CONFIG",
    "INDICATOR_CACHE_CONFIG",
    "WORKER_CONFIG",
    "MEMORY_BREAKDOWN",
    "get_gpu_config",
    "get_indicator_cache_config",
    "get_worker_config",
    "print_memory_breakdown",
]