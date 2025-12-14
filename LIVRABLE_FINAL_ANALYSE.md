# Backtest Core - Livrable Final d'Analyse

> **Date:** 13/12/2025  
> **Objectif:** Cartographie complète des 3 modes d'exécution (TEST, GRILLES, AUTONOME)

---

## 1️⃣ Tableau Comparatif

| Rubrique | TEST | GRILLES / SWEEP | AUTONOME / LLM |
|----------|------|-----------------|----------------|
| **Entrypoints CLI** | `python -m backtest_core backtest` | `python -m backtest_core sweep` / `optuna` | ❌ Pas de commande dédiée dans dispatcher |
| **Entrypoints UI** | Backtest Simple (`optimization_mode == "Backtest Simple"`) | Grille de Paramètres (`optimization_mode == "Grille de Paramètres"`) | 🤖 Optimisation LLM (`optimization_mode == "🤖 Optimisation LLM"`) |
| **Fonctions clés** | `cli/commands.py:cmd_backtest()` → `BacktestEngine.run()` | CLI: `cmd_sweep()` → `generate_param_grid()` → boucle `engine.run()` | `create_optimizer_from_engine()` → `AutonomousStrategist.optimize()` → `BacktestExecutor.run()` → `BacktestEngine.run()` |
| **Modules clés** | `backtest/engine.py`, `backtest/simulator.py`, `backtest/performance.py` | + `utils/parameters.py` (granularity), `performance/parallel.py` (cartésien) | + `agents/integration.py`, `agents/backtest_executor.py`, `agents/autonomous_strategist.py`, `agents/llm_client.py` |
| **I/O principal** | Input: `df` + `params` ; Output: `RunResult` | Input: espace discret (grid) ; Output: liste résultats + best (`sweep_results.json`) | Input: `params init` + `param_bounds` ; Output: historique itérations + best |
| **Calcul "granularité / combinaisons"** | N/A (1 config) | CLI: `--granularity` + `generate_param_grid()` + `len(grid)`. UI: `count` × `total_combinations` | ❌ Non calculé: seulement `param_bounds (min/max)` + clamp |
| **Pourquoi stats grilles absentes** | — | Calculables car espace discret explicite | Absent car exploration continue/itérative sans step/granularity |

---

## 2️⃣ Flowcharts Texte

### (1) Mode TEST — Backtest Simple

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI: python -m backtest_core backtest -s ema_cross -d data.parquet    │
│  ou                                                                     │
│  UI: Sélection "Backtest Simple" + clic "🚀 Lancer le Backtest"        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  cli/commands.py:cmd_backtest() ou ui/app.py:safe_run_backtest()         │
│  ↓                                                                        │
│  • Charge le DataFrame OHLCV via data/loader.py                          │
│  • Résolution du chemin ($BACKTEST_DATA_DIR ou local)                    │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  backtest/engine.py:BacktestEngine.run(df, strategy, params)              │
│  ↓                                                                        │
│  1. Résolution stratégie via strategies/base.py:get_strategy()           │
│  2. Calcul indicateurs → indicators/registry.py:calculate_indicator()    │
│  3. Génération signaux → strategy.generate_signals()                     │
│  4. Simulation trades → backtest/simulator.py:simulate_trades()          │
│  5. Equity curve → calculate_equity_curve()                              │
│  6. Métriques → backtest/performance.py:calculate_metrics()              │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: RunResult                                                        │
│  ├── equity: pd.Series (courbe d'équité)                                 │
│  ├── returns: pd.Series (rendements)                                     │
│  ├── trades: pd.DataFrame (liste des trades)                             │
│  ├── metrics: Dict (sharpe, sortino, max_drawdown, win_rate, etc.)       │
│  └── meta: Dict (durée, symbol, timeframe)                               │
└───────────────────────────────────────────────────────────────────────────┘
```

**Fichiers impliqués :**
- [cli/commands.py](cli/commands.py#L359-L475) — `cmd_backtest()`
- [backtest/engine.py](backtest/engine.py#L100-L250) — `BacktestEngine.run()`
- [backtest/simulator.py](backtest/simulator.py) — `simulate_trades()`
- [backtest/performance.py](backtest/performance.py) — `calculate_metrics()`

---

### (2) Mode GRILLES / SWEEP — Optimisation Paramétrique

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI: python -m backtest_core sweep -s ema_cross -d data.parquet       │
│       --granularity 0.3 --metric sharpe                                 │
│  ou                                                                     │
│  UI: Sélection "Grille de Paramètres" + configuration min/max/step     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│  CLI: cmd_sweep()             │   │  UI: app.py "Grille de Paramètres"   │
│  ↓                            │   │  ↓                                    │
│  Construction ParameterSpec   │   │  Collecte min/max/step par param     │
│  depuis strat.param_ranges    │   │  via create_param_range_selector()   │
└───────────────┬───────────────┘   └───────────────┬───────────────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│  utils/parameters.py          │   │  Calcul total_combinations            │
│  generate_param_grid(         │   │  (produit cartésien des counts)       │
│    param_specs,               │   │                                       │
│    granularity=0.3,           │   │  total_combinations *= count          │
│    max_total_combinations     │   │  par paramètre                        │
│  )                            │   │                                       │
│  → Liste[Dict] combinaisons   │   │  → param_grid par produit cartésien   │
└───────────────┬───────────────┘   └───────────────┬───────────────────────┘
                │                                   │
                └───────────────┬───────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  BOUCLE SUR GRILLE                                                        │
│  for params in grid:                                                      │
│      result = BacktestEngine.run(df, strategy, params)                   │
│      scores.append((params, result.metrics))                             │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  AGRÉGATION                                                               │
│  • Tri par métrique cible (sharpe, sortino, etc.)                        │
│  • Sélection du best                                                     │
│  • Export → sweep_results.json / rapport HTML                            │
└───────────────────────────────────────────────────────────────────────────┘
```

**Statistiques calculées :**
| Élément | CLI | UI |
|---------|-----|-----|
| Combinaisons totales | `len(grid)` affiché | `total_combinations` (produit) |
| Par paramètre | implicite via `generate_param_grid()` | `range_data["count"]` |
| Granularité | `--granularity 0.0-1.0` | Slider ou direct min/max/step |

**Fichiers impliqués :**
- [cli/commands.py](cli/commands.py#L505-L650) — `cmd_sweep()`
- [utils/parameters.py](utils/parameters.py#L76-L228) — `parameter_values()`, `generate_param_grid()`
- [ui/app.py](ui/app.py#L776-L816) — calcul `total_combinations`

---

### (3) Mode LLM AUTONOME — Optimisation par Agents

```
┌─────────────────────────────────────────────────────────────────────────┐
│  UI: Sélection "🤖 Optimisation LLM"                                   │
│  • Configuration provider (Ollama/OpenAI)                              │
│  • max_iterations, use_walk_forward                                     │
│  • Pas de CLI dédié (divergence majeure)                               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Construction param_bounds (min/max SANS step)                            │
│  ui/app.py ligne ~1100:                                                   │
│      param_bounds[pname] = (c["min"], c["max"])                          │
│                                                                           │
│  ⚠️ PAS de step → impossible de calculer "combinaisons"                 │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  agents/integration.py:create_optimizer_from_engine()                     │
│  ↓                                                                        │
│  1. Crée LLMClient (Ollama ou OpenAI)                                    │
│  2. Crée backtest_fn = run_backtest_for_agent()                          │
│  3. Crée BacktestExecutor avec backtest_fn                               │
│  4. Crée AutonomousStrategist avec LLMClient                             │
│  → Retourne (strategist, executor)                                       │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  agents/autonomous_strategist.py:AutonomousStrategist.optimize()          │
│  ↓                                                                        │
│  BOUCLE ITÉRATIVE (≤ max_iterations):                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  1. LLM formule hypothèse                                           │ │
│  │  2. LLM propose params (clamp dans param_bounds)                    │ │
│  │  3. BacktestExecutor.run() → BacktestEngine.run()                   │ │
│  │  4. Analyse résultats                                               │ │
│  │  5. LLM décide: continue | accept | stop | change_direction         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  agents/backtest_executor.py:BacktestExecutor.run()                       │
│  ↓                                                                        │
│  → agents/integration.py:run_backtest_for_agent()                        │
│  → backtest/engine.py:BacktestEngine.run()                               │
│  (optionnel: walk-forward validation)                                    │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: OptimizationSession                                              │
│  ├── best_result: BacktestResult (meilleure itération)                   │
│  ├── all_results: List[BacktestResult] (historique)                      │
│  ├── decisions: List[IterationDecision] (décisions LLM)                  │
│  ├── final_status: "success" | "max_iterations" | "timeout"              │
│  └── final_reasoning: str (explication LLM)                              │
│                                                                           │
│  ⚠️ PAS d'espace discret: exploration continue, nombre de combinaisons  │
│     impossible à calculer a priori                                        │
└───────────────────────────────────────────────────────────────────────────┘
```

**Fichiers impliqués :**
- [agents/integration.py](agents/integration.py#L214-L310) — `create_optimizer_from_engine()`
- [agents/autonomous_strategist.py](agents/autonomous_strategist.py#L100-L180) — `AutonomousStrategist.optimize()`
- [agents/backtest_executor.py](agents/backtest_executor.py) — `BacktestExecutor`
- [ui/app.py](ui/app.py#L1100-L1200) — Interface UI mode LLM

---

## 3️⃣ Divergences Majeures avec Preuves

### Divergence 1: Pas de mode LLM en CLI

**Constat:** Le dispatcher CLI n'expose aucune commande pour le mode LLM autonome.

**Preuve:** [cli/__init__.py](cli/__init__.py#L12-L21)
```python
from .commands import (
    cmd_backtest,
    cmd_export,
    cmd_info,
    cmd_list,
    cmd_optuna,
    cmd_sweep,
    cmd_validate,
    cmd_visualize,
)
# ❌ Pas de cmd_llm ou cmd_autonomous
```

**Impact:** L'optimisation LLM n'est accessible que via l'UI Streamlit, pas en ligne de commande.

---

### Divergence 2: Deux générateurs de "grille" différents

| Chemin | Module | Méthode | Granularité |
|--------|--------|---------|-------------|
| CLI `sweep` | `utils/parameters.py` | `generate_param_grid(granularity=...)` | ✅ Oui |
| UI Grille | `ui/app.py` | Produit cartésien direct avec min/max/step | Implicite via step |
| Optuna | `backtest/optuna_optimizer.py` | Échantillonnage bayésien | N/A |

**Preuve 1:** [utils/parameters.py](utils/parameters.py#L163-L228) — `generate_param_grid()`
```python
def generate_param_grid(
    params_specs: Dict[str, ParameterSpec],
    granularity: float = 0.5,  # ← contrôle la réduction
    max_values_per_param: int = 4,
    max_total_combinations: int = 10000
) -> List[Dict[str, Any]]:
```

**Preuve 2:** [ui/app.py](ui/app.py#L776-L800) — calcul UI direct
```python
total_combinations = 1
for param_name, spec in param_specs.items():
    range_data = create_param_range_selector(...)
    total_combinations *= range_data["count"]  # ← produit direct
```

**Impact:** Selon le chemin d'exécution, la notion de "granularité" peut exister (CLI) ou non (UI).

---

### Divergence 3: UI Grille calcule des stats, LLM non

**UI Grille:**
```python
# ui/app.py ligne 812-816
if total_combinations > max_combos:
    st.sidebar.warning(f"⚠️ {total_combinations:,} combinaisons (limite: {max_combos:,})")
else:
    st.sidebar.success(f"✅ {total_combinations:,} combinaisons à tester")
```

**UI LLM:**
```python
# ui/app.py ligne 1100-1105
param_bounds = {}
for pname in params.keys():
    if pname in PARAM_CONSTRAINTS:
        c = PARAM_CONSTRAINTS[pname]
        param_bounds[pname] = (c["min"], c["max"])  # ← Seulement min/max, PAS de step
```

**Impact:** Le mode LLM ne peut pas afficher "nombre de combinaisons" car il n'énumère pas l'espace.

---

### Divergence 4: Pipeline de backtest partagé

**Bonne nouvelle:** Les 3 modes convergent vers le même `BacktestEngine.run()`.

**Preuve:** [agents/integration.py](agents/integration.py#L43-L97) — `run_backtest_for_agent()`
```python
def run_backtest_for_agent(...) -> Dict[str, Any]:
    engine = BacktestEngine(...)
    result = engine.run(df=data, strategy=strategy_name, params=params)
    # ...
```

**Impact:** La cohérence des métriques est garantie entre les modes.

---

### Divergence 5: Orchestrator multi-agents n'exécute pas sans callback

**Constat:** L'Orchestrator (workflow multi-agents) requiert un callback `on_backtest_needed` pour exécuter des backtests.

**Preuve:** [agents/orchestrator.py](agents/orchestrator.py#L380-L400)
```python
def _run_backtest(self, params: Dict[str, Any]) -> Optional[MetricsSnapshot]:
    if self.config.on_backtest_needed:
        result = self.config.on_backtest_needed(params)
        # ...
    return None  # ← Retourne None si pas de callback !
```

**Impact:** Le mode "Orchestration multi-agents" (différent de l'AutonomousStrategist) n'est pas fonctionnel sans configuration explicite du callback.

---

## 4️⃣ Recommandations d'Unification (6 Actions) — ✅ IMPLÉMENTÉES

> **Toutes les actions ci-dessous ont été implémentées le 13/12/2025**

### Action 1: ✅ Créer `compute_search_space_stats()` dans `utils/parameters.py`

**Objectif:** Fonction unifiée pour calculer les stats d'espace de recherche.

```python
# utils/parameters.py (à ajouter)

from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class SearchSpaceStats:
    """Statistiques d'un espace de recherche."""
    total_combinations: int
    per_param_counts: Dict[str, int]
    warnings: List[str]
    has_overflow: bool
    is_continuous: bool  # True si step manquant

def compute_search_space_stats(
    param_space: Dict[str, Union[ParameterSpec, Tuple[float, float], Tuple[float, float, float]]],
    max_combinations: int = 100000,
) -> SearchSpaceStats:
    """
    Calcule les statistiques d'un espace de recherche.
    
    Args:
        param_space: Dictionnaire avec:
            - ParameterSpec: utilise min_val, max_val, step
            - Tuple (min, max): espace continu, retourne is_continuous=True
            - Tuple (min, max, step): espace discret
        max_combinations: Seuil d'avertissement
        
    Returns:
        SearchSpaceStats avec total, counts par param, warnings
    """
    total = 1
    counts = {}
    warnings = []
    is_continuous = False
    
    for name, spec in param_space.items():
        if isinstance(spec, ParameterSpec):
            # Utiliser step si disponible
            if spec.step and spec.step > 0:
                count = int((spec.max_val - spec.min_val) / spec.step) + 1
            else:
                count = -1  # Continu
                is_continuous = True
        elif isinstance(spec, tuple):
            if len(spec) == 3:
                min_v, max_v, step = spec
                count = int((max_v - min_v) / step) + 1
            else:
                count = -1
                is_continuous = True
        else:
            count = 1
        
        counts[name] = count
        if count > 0:
            total *= count
    
    if is_continuous:
        total = -1  # Indique "non calculable"
        warnings.append("Espace continu: nombre de combinaisons non défini")
    
    has_overflow = total > max_combinations
    if has_overflow:
        warnings.append(f"Limite dépassée: {total:,} > {max_combinations:,}")
    
    return SearchSpaceStats(
        total_combinations=total,
        per_param_counts=counts,
        warnings=warnings,
        has_overflow=has_overflow,
        is_continuous=is_continuous,
    )
```

---

### Action 2: ✅ Faire appeler `compute_search_space_stats()` par l'UI mode Grille

**Fichier:** `ui/app.py`

**Avant:**
```python
total_combinations = 1
for param_name, spec in param_specs.items():
    range_data = create_param_range_selector(...)
    total_combinations *= range_data["count"]
```

**Après:**
```python
from utils.parameters import compute_search_space_stats

# Construire l'espace avec step
param_space_with_step = {}
for param_name, range_data in param_ranges.items():
    param_space_with_step[param_name] = (
        range_data["min"], range_data["max"], range_data["step"]
    )

stats = compute_search_space_stats(param_space_with_step, max_combinations=max_combos)

if stats.has_overflow:
    st.sidebar.warning(f"⚠️ {stats.total_combinations:,} combinaisons (limite: {max_combos:,})")
else:
    st.sidebar.success(f"✅ {stats.total_combinations:,} combinaisons à tester")
```

---

### Action 3: ✅ Faire appeler `compute_search_space_stats()` par le CLI `cmd_sweep`

**Fichier:** `cli/commands.py`

```python
# Dans cmd_sweep(), après generate_param_grid()
from utils.parameters import compute_search_space_stats

stats = compute_search_space_stats(param_specs)
if not args.quiet:
    print_info(f"Espace de recherche: {stats.total_combinations:,} combinaisons")
    for name, count in stats.per_param_counts.items():
        print(f"    {name}: {count} valeurs")
```

---

### Action 4: ✅ Étendre `get_strategy_param_bounds()` → `get_strategy_param_space()`

**Fichier:** `agents/integration.py`

```python
def get_strategy_param_space(
    strategy_name: str,
    include_step: bool = True,
) -> Dict[str, Tuple]:
    """
    Récupère l'espace des paramètres avec step si disponible.
    
    Returns:
        Dict {param_name: (min, max)} ou {param_name: (min, max, step)}
    """
    strategy_class = get_strategy(strategy_name)
    strategy = strategy_class()
    
    space = {}
    
    if hasattr(strategy, 'parameter_specs'):
        specs = strategy.parameter_specs
        if isinstance(specs, dict):
            for name, spec in specs.items():
                if hasattr(spec, 'min_val') and hasattr(spec, 'max_val'):
                    if include_step and hasattr(spec, 'step') and spec.step:
                        space[name] = (spec.min_val, spec.max_val, spec.step)
                    else:
                        space[name] = (spec.min_val, spec.max_val)
    
    return space
```

---

### Action 5: ✅ Option UI LLM — afficher estimation si step connu

**Fichier:** `ui/app.py` (section LLM)

```python
# Après construction de param_bounds
from utils.parameters import compute_search_space_stats
from agents.integration import get_strategy_param_space

# Tenter de récupérer le step
full_space = get_strategy_param_space(strategy_key, include_step=True)
stats = compute_search_space_stats(full_space)

if stats.is_continuous:
    st.sidebar.info("ℹ️ Espace continu: exploration adaptative par LLM")
else:
    st.sidebar.caption(f"📊 Espace discret estimé: ~{stats.total_combinations:,} combinaisons")
```

---

### Action 6: ✅ Brancher l'Orchestrator sur `run_backtest_for_agent()`

**Fichier:** Usage dans le code appelant

```python
from agents.orchestrator import Orchestrator, OrchestratorConfig
from agents.integration import run_backtest_for_agent

config = OrchestratorConfig(
    strategy_name="ema_cross",
    initial_params={"fast_period": 12, "slow_period": 26},
    # Fournir le callback manquant
    on_backtest_needed=lambda params: run_backtest_for_agent(
        strategy_name="ema_cross",
        params=params,
        data=df,
    ),
)

orchestrator = Orchestrator(config)
result = orchestrator.run()  # Maintenant fonctionnel !
```

---

## 5️⃣ Résumé Exécutif

| Aspect | État Actuel | Action Requise |
|--------|-------------|----------------|
| CLI Test/Sweep | ✅ Fonctionnel | — |
| CLI LLM | ❌ Absent | Ajouter `cmd_llm()` (optionnel) |
| UI 3 modes | ✅ Fonctionnel | — |
| Stats grille unifiées | ✅ `compute_search_space_stats()` | — |
| Stats LLM | ✅ Estimation affichée | — |
| Orchestrator callback | ✅ `create_orchestrator_with_backtest()` | — |
| BacktestEngine partagé | ✅ Cohérent | — |

---

## 📁 Fichiers Modifiés (13/12/2025)

| Fichier | Modification |
|---------|--------------|
| [utils/parameters.py](utils/parameters.py) | + `SearchSpaceStats`, `compute_search_space_stats()` |
| [ui/app.py](ui/app.py) | Utilisation stats unifiées (Grille + LLM) |
| [cli/commands.py](cli/commands.py) | Affichage stats détaillées dans `cmd_sweep()` |
| [agents/integration.py](agents/integration.py) | + `get_strategy_param_space()`, `create_orchestrator_with_backtest()` |
| [agents/__init__.py](agents/__init__.py) | Exports des nouvelles fonctions |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Index des modifications mis à jour |

---

*Généré automatiquement le 13/12/2025 — Implémentation complète*
