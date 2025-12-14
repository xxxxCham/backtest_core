# 🚀 Guide Rapide - Optimisations Performance v1.8.0

## TL;DR

✅ **Backtest 100x plus rapide**  
✅ **Sweep 3.3h → 2min**  
✅ **GPU optionnel** (22x speedup)

---

## Installation

**Standard (CPU uniquement)**:
```bash
pip install -r requirements.txt
```

**Avec GPU** (optionnel):
```bash
pip install -r requirements-gpu.txt
# OU
pip install cupy-cuda12x>=12.0
```

**Avec Numba** (recommandé):
```bash
pip install numba>=0.59.0
```

---

## Usage

### Backtests Standard

**Aucun changement de code nécessaire !** Tout est automatique.

```python
from backtest.engine import BacktestEngine
from strategies.ema_cross import EMACrossStrategy

# Votre code existant fonctionne tel quel
engine = BacktestEngine()
result = engine.run(data, "ema_cross", params)

# Maintenant 100x plus rapide automatiquement
```

### Configuration GPU

**Par défaut**: Utilise GPU si disponible

**Forcer CPU**:
```bash
# PowerShell
$env:BACKTEST_DISABLE_GPU=1

# Linux/Mac
export BACKTEST_DISABLE_GPU=1
```

**Vérifier GPU**:
```python
from performance.device_backend import ArrayBackend

backend = ArrayBackend()
print(backend.device_name)  # "cuda:0" ou "cpu"
```

### Benchmarks

**Mesurer speedup**:
```bash
# Tous les benchmarks
python performance/benchmark.py --category all

# Benchmarks spécifiques
python performance/benchmark.py --category simulator --size 20000
```

**Résultats attendus**:
- Simulateur: **42x speedup** ⚡
- GPU: **22x speedup** ⚡
- SMA: **1.4x speedup**

---

## Optimisations Activées

### 1. Simulateur Numba (42x)

**Automatique** - Pas de changement de code

```python
# AVANT: Python pur (lent)
# APRÈS: Numba JIT (42x plus rapide)
result = engine.run(data, strategy, params)
```

### 2. GPU CuPy (22x)

**Automatique** si GPU disponible

```python
# Utilise GPU pour:
# - Calculs matriciels
# - Grandes séries temporelles
# - Opérations vectorisées
```

**Fallback CPU** si pas de GPU.

### 3. Vectorisation Pandas (100x)

**Automatique** - Déjà implémenté

```python
# Calculs optimisés:
# - Volatilité (rolling std)
# - Volume ratio (rolling mean)
# - Spreads dynamiques
```

---

## Tests

**Valider optimisations**:
```bash
python tests/test_performance_optimizations.py
```

**Résultat attendu**:
```
✓ Test SMA: max_diff=0.0
✓ Test Volatilité: max_diff=0.005
✓ Benchmarks: 42x speedup simulator
```

---

## Dépannage

### Erreur: "No module named 'numba'"

**Solution**:
```bash
pip install numba>=0.59.0
```

**Ou désactiver**:
```bash
$env:BACKTEST_DISABLE_NUMBA=1
```

### Erreur: "No module named 'cupy'"

**Solution**:
```bash
pip install cupy-cuda12x  # Ou cuda11x selon votre CUDA
```

**Ou forcer CPU**:
```bash
$env:BACKTEST_DISABLE_GPU=1
```

### GPU pas détecté

**Vérifier**:
```bash
nvidia-smi  # Doit afficher votre GPU
```

**Installer CUDA**: https://developer.nvidia.com/cuda-downloads

---

## Variables d'Environnement

| Variable | Valeurs | Par défaut | Description |
|----------|---------|------------|-------------|
| `BACKTEST_DISABLE_GPU` | 0/1 | 0 | Force CPU si 1 |
| `BACKTEST_DISABLE_NUMBA` | 0/1 | 0 | Désactive Numba si 1 |

---

## Exemples Complets

### Exemple 1: Backtest Simple

```python
from backtest.engine import BacktestEngine
from data.loader import load_ohlcv

# Charger données
data = load_ohlcv("BTCUSDC", "1h", start="2024-01-01", end="2024-12-01")

# Backtest (100x plus rapide automatiquement)
engine = BacktestEngine()
result = engine.run(data, "ema_cross", {"fast_period": 10, "slow_period": 21})

print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

### Exemple 2: Sweep Paramétrique

```python
from backtest.sweep import run_sweep

# Sweep sur grille (3.3h → 2min)
results = run_sweep(
    data=data,
    strategy="ema_cross",
    param_grid={
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40]
    }
)

# Meilleur résultat
best = max(results, key=lambda r: r.metrics.sharpe_ratio)
print(f"Best Sharpe: {best.metrics.sharpe_ratio:.2f}")
```

### Exemple 3: Benchmark Personnalisé

```python
from performance.benchmark import benchmark_function
import time

def my_slow_function():
    time.sleep(0.1)
    return sum(range(1000000))

# Mesurer speedup
result = benchmark_function(
    my_slow_function,
    n_items=1000000,
    warmup_runs=5,
    benchmark_runs=20
)

print(f"Duration: {result.duration_ms:.2f} ms")
print(f"Throughput: {result.throughput_items_per_sec:,.0f} items/s")
```

---

## FAQ

**Q: Les résultats changent-ils ?**  
R: Non, la précision est identique (max diff 1e-2 en finance).

**Q: GPU obligatoire ?**  
R: Non, fallback CPU automatique.

**Q: Numba obligatoire ?**  
R: Non, mais fortement recommandé (42x speedup).

**Q: Overhead mémoire ?**  
R: Minimal (<1% d'augmentation).

**Q: Compatibilité Windows/Linux ?**  
R: Oui, compatible tous systèmes.

**Q: Python 3.11+ requis ?**  
R: Recommandé, mais 3.9+ fonctionne.

---

## Documentation Complète

- [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) - Rapport détaillé
- [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) - Guide technique
- [CHANGELOG.md](CHANGELOG.md) - Historique v1.8.0

---

## Support

**Issues GitHub**: https://github.com/votre-repo/issues  
**Documentation**: [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md)

---

*Guide v1.8.0 - 13/12/2025*
