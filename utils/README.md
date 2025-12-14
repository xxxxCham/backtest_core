# Utils Module - Utilitaires Système

> **Utilitaires transversaux pour backtest_core**  
> Version : 1.8.1 | Date : 13/12/2025

---

## 📊 Vue d'Ensemble

Le module `utils/` regroupe tous les utilitaires système et outils transversaux utilisés par les autres modules. Il garantit la cohérence, la résilience et l'observabilité du système.

**Catégories :**
- ⚙️ **Configuration** : Gestion centralisée des paramètres
- 📝 **Logging** : Logging simple et observabilité avancée
- 🔧 **Paramètres** : Spécifications et contraintes
- 🛡️ **Résilience** : Circuit breaker, error recovery, checkpoints
- 💾 **Ressources** : Monitoring santé, gestion mémoire
- 📊 **Visualisation** : Graphiques interactifs Plotly
- 🎨 **Templates** : Prompts Jinja2 pour LLM

---

## 📁 Structure

```
utils/
├── config.py              → Configuration globale (singleton)
├── log.py                 → Logging simple (legacy)
├── observability.py       → Observabilité intelligente (v1.8.0)
├── parameters.py          → Specs paramètres + contraintes
├── template.py            → Moteur templates Jinja2
├── visualization.py       → Graphiques Plotly (v1.7.0)
├── health.py              → Monitoring santé système
├── memory.py              → Gestion mémoire + cache LRU
├── circuit_breaker.py     → Protection échecs répétés
├── checkpoint.py          → Sauvegarde/reprise état
├── error_recovery.py      → Récupération erreurs
├── gpu_oom.py             → Gestion OOM GPU
└── __init__.py            → Exports publics
```

---

## 🚀 Guide d'Utilisation

### 1. Configuration (`config.py`)

**Objectif :** Configuration globale centralisée (pattern Singleton)

```python
from utils import Config

# Récupérer singleton
config = Config()

# Paramètres disponibles
config.fees_bps              # Frais en basis points (défaut: 10)
config.slippage_bps          # Slippage en BPS (défaut: 5)
config.initial_capital       # Capital initial (défaut: 10000)
config.use_gpu               # Activer GPU (défaut: False)
config.n_jobs                # Workers parallèles (défaut: -1)

# Modifier config
config.fees_bps = 20
config.use_gpu = True

# Reset aux valeurs par défaut
config.reset()
```

**Variables d'environnement supportées :**
```bash
BACKTEST_FEES_BPS=10
BACKTEST_SLIPPAGE_BPS=5
BACKTEST_INITIAL_CAPITAL=10000
BACKTEST_USE_GPU=False
BACKTEST_N_JOBS=-1
```

---

### 2. Logging Simple (`log.py`)

**Objectif :** Logger standard pour usage basique

```python
from utils.log import get_logger

logger = get_logger(__name__)

logger.info("Backtest démarré")
logger.warning("Paramètre fast_period proche de slow_period")
logger.error("Erreur lors du calcul d'indicateur")
```

**Format de sortie :**
```
14:32:15 | INFO     | backtest.engine | Backtest démarré
14:32:17 | WARNING  | strategies.ema  | Paramètre fast_period proche...
```

---

### 3. Observabilité (`observability.py`)

**Objectif :** Système d'observabilité avancé avec tracing et métriques (v1.8.0)

```python
from utils.observability import get_obs_logger, trace_span, generate_run_id

# Logger avec contexte corrélé
run_id = generate_run_id()  # Ex: "a1b2c3d4"
logger = get_obs_logger(__name__, run_id=run_id, strategy="ema_cross")

logger.info("Début backtest")  # [a1b2c3d4][ema_cross] Début backtest

# Span chronométré (zéro coût si DEBUG désactivé)
with trace_span(logger, "calculate_indicators", count=5):
    # ... calculs ...
    pass
# Output: [a1b2c3d4] calculate_indicators (count=5) → 1.234s

# Performance counters
from utils.observability import PerfCounters

counters = PerfCounters()
counters.increment("backtests_run")
counters.add_duration("indicator_calc", 0.5)

print(counters.report())
# backtests_run: 42
# indicator_calc: 21.3s (avg: 0.507s)
```

**Activation Debug :**
```bash
export BACKTEST_LOG_LEVEL=DEBUG
# Ou dans UI : Toggle "Observabilité Debug"
```

---

### 4. Paramètres (`parameters.py`)

**Objectif :** Spécifications de paramètres avec contraintes

```python
from utils.parameters import ParameterSpec, ConstraintValidator, Preset

# Définir spec d'un paramètre
spec = ParameterSpec(
    name="fast_period",
    min_value=5,
    max_value=50,
    default=10,
    step=1,
    description="Période EMA rapide",
)

# Générer valeurs discrètes
values = spec.generate_values(granularity=0.5)  # [5, 10, 15, ..., 50]

# Validation avec contraintes
validator = ConstraintValidator()
validator.add_greater_than('slow_period', 'fast_period')
validator.add_ratio_min('slow_period', 'fast_period', ratio=1.5)

# Filtrer grille invalide
param_grid = [
    {"fast": 10, "slow": 20},
    {"fast": 10, "slow": 12},  # Invalide (ratio < 1.5)
]
valid_grid = validator.filter_grid(param_grid)

# Presets prédéfinis
from utils.parameters import PRESET_AGGRESSIVE

preset = PRESET_AGGRESSIVE
print(preset.fees_bps)  # 15 (frais élevés)
```

**Types de contraintes :**
- `greater_than` : A > B
- `less_than` : A < B
- `ratio_min` : A/B >= ratio
- `ratio_max` : A/B <= ratio
- `difference_min` : A - B >= delta
- `min_value` : A >= valeur
- `max_value` : A <= valeur

---

### 5. Templates Jinja2 (`template.py`)

**Objectif :** Moteur de templates pour prompts LLM (Phase 3)

```python
from utils.template import render_prompt, list_available_templates

# Lister templates disponibles
templates = list_available_templates()
# ['analyst.jinja2', 'strategist.jinja2', 'critic.jinja2', 'validator.jinja2']

# Rendre un template
context = {
    "metrics": {"sharpe_ratio": 1.5, "max_drawdown": 8.0},
    "strategy_name": "ema_cross",
}

prompt = render_prompt("analyst.jinja2", context)
print(prompt)  # Prompt formaté pour l'Analyst Agent
```

**Templates disponibles :**
- `analyst.jinja2` : Analyse quantitative
- `strategist.jinja2` : Propositions optimisation
- `critic.jinja2` : Évaluation risques
- `validator.jinja2` : Décision finale

---

### 6. Visualisation (`visualization.py`)

**Objectif :** Graphiques interactifs Plotly (v1.7.0)

```python
from utils.visualization import (
    plot_trades,
    plot_equity_curve,
    plot_drawdown,
    visualize_backtest,
    load_and_visualize,
)

# Graphique candlestick + trades
fig = plot_trades(ohlcv_df, trades_list)
fig.show()

# Equity curve
fig = plot_equity_curve(equity_curve, title="Performance EMA Cross")
fig.show()

# Drawdown
fig = plot_drawdown(equity_curve)
fig.show()

# Rapport complet (HTML)
visualize_backtest(
    ohlcv=ohlcv_df,
    trades=trades_list,
    metrics=metrics_dict,
    output_file="report.html",
)

# Chargement depuis JSON + visualisation
load_and_visualize(
    results_file="sweep_results.json",
    data_file="BTCUSDT_1h.parquet",
)
```

**CLI :**
```bash
python __main__.py visualize -i results.json -d data.csv --html
```

---

### 7. Monitoring Santé (`health.py`)

**Objectif :** Surveillance santé système (CPU/RAM/GPU/Disk)

```python
from utils.health import HealthMonitor, ResourceStatus

# Créer monitor avec seuils personnalisés
monitor = HealthMonitor(
    cpu_threshold=80.0,      # Alerte si CPU > 80%
    memory_threshold=85.0,   # Alerte si RAM > 85%
    disk_threshold=90.0,     # Alerte si Disk > 90%
    gpu_threshold=95.0,      # Alerte si GPU > 95%
)

# Check santé
status = monitor.check_health()

if status == ResourceStatus.CRITICAL:
    print("⚠️ Ressources critiques!")
    print(monitor.get_alerts())
elif status == ResourceStatus.WARNING:
    print("⚠️ Attention ressources")
else:
    print("✅ Système OK")

# Rapport détaillé
print(monitor.report())
# CPU: 45% | RAM: 12.3/32.0 GB | GPU: 60% (8.1/12.0 GB)
```

---

### 8. Gestion Mémoire (`memory.py`)

**Objectif :** Gestion mémoire + cache LRU (Phase 4)

```python
from utils.memory import MemoryManager, ManagedCache

# Memory Manager global
manager = MemoryManager(
    max_memory_gb=16.0,          # Limite RAM
    cleanup_threshold=0.9,       # Nettoyage si > 90%
    aggressive_mode=False,       # Mode économie mémoire
)

# Vérifier mémoire disponible
if manager.can_allocate(required_gb=2.0):
    data = load_large_dataset()
else:
    print("Mémoire insuffisante")

# Cache LRU avec limite mémoire
cache = ManagedCache(max_size_gb=1.0, max_items=100)

# Ajouter au cache
cache.set("key1", large_dataframe)

# Récupérer depuis cache
df = cache.get("key1")  # None si absent

# Auto-cleanup
manager.cleanup()  # Libère mémoire si nécessaire
```

---

### 9. Circuit Breaker (`circuit_breaker.py`)

**Objectif :** Protection contre échecs répétés (Phase 4)

```python
from utils.circuit_breaker import CircuitBreaker

# Créer circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,       # Ouvrir après 5 échecs
    timeout_seconds=30.0,      # Timeout par appel
    recovery_timeout=60.0,     # Test recovery après 60s
)

# Protéger une fonction
@breaker
def risky_operation():
    result = external_api_call()
    return result

# Appel protégé
try:
    result = risky_operation()
except CircuitBreakerOpen:
    print("Circuit ouvert, service indisponible")

# Vérifier état
print(breaker.state)  # CLOSED | OPEN | HALF_OPEN
print(breaker.failure_count)
```

**États :**
- `CLOSED` : Normal, appels passent
- `OPEN` : Trop d'échecs, appels bloqués
- `HALF_OPEN` : Test de recovery

---

### 10. Checkpoints (`checkpoint.py`)

**Objectif :** Sauvegarde/reprise automatique (Phase 4)

```python
from utils.checkpoint import CheckpointManager

# Créer manager
manager = CheckpointManager(checkpoint_dir="./checkpoints")

# Sauvegarder état
state = {
    "iteration": 42,
    "best_params": {"fast": 10, "slow": 21},
    "results": results_list,
}
manager.save_checkpoint("sweep_session_1", state)

# Reprendre depuis checkpoint
if manager.has_checkpoint("sweep_session_1"):
    state = manager.load_checkpoint("sweep_session_1")
    print(f"Reprise à l'itération {state['iteration']}")

# Lister checkpoints
checkpoints = manager.list_checkpoints()
for name, timestamp in checkpoints:
    print(f"{name} (sauvé le {timestamp})")

# Nettoyer vieux checkpoints
manager.cleanup_old_checkpoints(keep_last=5)
```

---

### 11. Error Recovery (`error_recovery.py`)

**Objectif :** Récupération gracieuse après erreurs (Phase 4)

```python
from utils.error_recovery import RetryHandler, ErrorClassifier

# Handler avec retry exponentiel
handler = RetryHandler(
    max_retries=3,
    base_delay=1.0,        # Délai initial 1s
    max_delay=60.0,        # Délai max 60s
    exponential_base=2.0,  # Backoff x2
)

# Exécuter avec retry
result = handler.execute(
    lambda: unstable_function(),
    on_retry=lambda attempt: print(f"Retry {attempt}..."),
)

# Classifier les erreurs
classifier = ErrorClassifier()

error = ValueError("Invalid parameter")
if classifier.is_transient(error):
    # Erreur transitoire → retry
    result = handler.execute(risky_function)
elif classifier.is_fatal(error):
    # Erreur fatale → stop
    raise error
```

---

### 12. GPU OOM Handler (`gpu_oom.py`)

**Objectif :** Gestion OOM GPU, fallback CPU (Phase 4)

```python
from utils.gpu_oom import GPUOOMHandler

# Handler avec fallback automatique
handler = GPUOOMHandler(
    fallback_to_cpu=True,
    retry_after_cleanup=True,
)

# Exécuter calcul GPU avec protection
try:
    result = handler.execute_safe(
        lambda: gpu_intensive_computation(),
        fallback_fn=lambda: cpu_fallback_computation(),
    )
except OutOfMemoryError:
    print("OOM même après cleanup")
```

---

## 🎯 Cas d'Usage Typiques

### Cas 1 : Backtest Robuste avec Résilience

```python
from utils import Config, get_obs_logger, HealthMonitor
from utils.circuit_breaker import CircuitBreaker
from utils.checkpoint import CheckpointManager

# Config
config = Config()
config.fees_bps = 10

# Observabilité
run_id = generate_run_id()
logger = get_obs_logger(__name__, run_id=run_id)

# Monitoring
health = HealthMonitor()

# Circuit breaker
breaker = CircuitBreaker(failure_threshold=3)

# Checkpoints
checkpoints = CheckpointManager()

# Backtest avec protection
for i, params in enumerate(param_grid):
    # Check santé
    if health.check_health() == ResourceStatus.CRITICAL:
        logger.warning("Ressources critiques, pause...")
        time.sleep(60)
    
    # Protéger exécution
    try:
        with trace_span(logger, "backtest", iteration=i):
            result = breaker(lambda: engine.run(params))()
            
            # Checkpoint tous les 10
            if i % 10 == 0:
                checkpoints.save_checkpoint(f"sweep_{run_id}", {
                    "iteration": i,
                    "results": results,
                })
    except CircuitBreakerOpen:
        logger.error("Circuit breaker ouvert")
        break
```

---

### Cas 2 : Optimisation avec Visualisation

```python
from utils import Config
from utils.visualization import visualize_backtest

# Optimisation
best_result = optimize_strategy(...)

# Visualisation
visualize_backtest(
    ohlcv=ohlcv_df,
    trades=best_result.trades,
    metrics=best_result.metrics.to_dict(),
    output_file="optimization_report.html",
)

# Ouvrir dans navigateur
import webbrowser
webbrowser.open("optimization_report.html")
```

---

## 🔧 Configuration Avancée

### Variables d'Environnement

```bash
# Logging
export BACKTEST_LOG_LEVEL=DEBUG           # DEBUG | INFO | WARNING | ERROR

# Configuration
export BACKTEST_FEES_BPS=10
export BACKTEST_SLIPPAGE_BPS=5
export BACKTEST_INITIAL_CAPITAL=10000

# Performance
export BACKTEST_USE_GPU=False
export BACKTEST_N_JOBS=-1

# Résilience
export BACKTEST_CIRCUIT_BREAKER_THRESHOLD=5
export BACKTEST_CHECKPOINT_ENABLED=True
```

---

## 📊 Dépendances

| Package | Usage | Requis |
|---------|-------|--------|
| `psutil` | Monitoring système | Optionnel |
| `plotly` | Visualisation | Optionnel |
| `jinja2` | Templates LLM | Requis (Phase 3) |

---

## 📚 Références

- [Observability Docs](../docs/OBSERVABILITY.md)
- [Phase 4 Robustesse](../.github/ROADMAP.md#phase-4---robustesse--résilience)
- [Visualization CLI](../.github/CLI_REFERENCE.md#visualize)

---

*Dernière mise à jour : 13/12/2025 | Version : 1.8.1*
