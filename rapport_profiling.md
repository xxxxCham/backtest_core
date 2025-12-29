# Rapport de Profiling du BacktestEngine

## Performance Actuelle

**Mesure** : 100 runs sur 1000 barres OHLCV (stratégie ema_cross)

- **Runs/sec** : 37.4 runs/sec
- **Temps moyen** : 26.8ms par run
- **Objectif** : 500 runs/sec (~2ms par run)
- **Gap** : **13.4x trop lent**

## Breakdown Détaillé (moyennes par run)

| Étape | Temps (ms) | % Total | Priorité |
|-------|------------|---------|----------|
| **OVERHEAD** | ~26ms | **71%** | 🔴 CRITIQUE |
| metrics | 5.61ms | 21% | 🟡 Important |
| simulation | 1.89ms | 7% | 🟢 OK |
| signals | 1.17ms | 4% | 🟢 OK |
| equity | 1.11ms | 4% | 🟢 OK |
| indicators | 0.16ms | 1% | 🟢 OK |
| **Total mesuré** | 10ms | - | - |
| **Total réel** | 36.41ms | - | - |

### Analyse de l'Overhead (~26ms)

L'overhead représente **71% du temps d'exécution**. Sources probables :

1. **Logging excessif** (~15-20ms estimé)
   - `RUN_START` : git_commit, params complets, metadata
   - `DATA_LOADED` : gap detection, statistiques complètes
   - `PARAMS_RESOLVED` : tous les paramètres
   - `RUN_END_SUMMARY` : 13+ métriques loggées
   - **Chacun de ces logs** fait du string formatting avec f-strings

2. **Initialisation BacktestEngine** (~2-3ms)
   - Création d'un nouvel engine à chaque run
   - Génération de run_id (UUID)
   - Setup de logger avec contexte
   - Création de PerfCounters

3. **Validation et détection de gaps** (~3-5ms)
   - `detect_gaps(df)` appelé à chaque run
   - Validation des entrées
   - Statistiques sur les données

4. **Git commit lookup** (~1-2ms)
   - `get_git_commit()` à chaque run

5. **Overhead pandas/numpy** (~2-3ms)
   - Copies de DataFrames
   - Conversions de types

## Optimisations Recommandées (Par Priorité)

### 1. DÉSACTIVER LE LOGGING STRUCTURÉ 🔴 (Gain estimé: +300 runs/sec)

**Impact** : ~20ms → ~5ms

Les logs RUN_START, DATA_LOADED, PARAMS_RESOLVED, RUN_END_SUMMARY sont **CRITIQUES** à désactiver en mode grid search.

**Solution** :
```python
# Dans engine.py, ajouter un paramètre silent_mode
def run(self, df, strategy, params=None, *, silent_mode=False, ...):
    if not silent_mode:
        self.logger.info("RUN_START ...")  # etc.
```

**Fichiers** :
- [backtest/engine.py](backtest/engine.py#L199-L227) : Tous les logs RUN_START, DATA_LOADED, PARAMS_RESOLVED
- [backtest/engine.py](backtest/engine.py#L340-L365) : Tous les logs RUN_END_SUMMARY

### 2. RÉUTILISER LE MÊME ENGINE 🔴 (Gain estimé: +50 runs/sec)

**Impact** : ~3ms → ~0.5ms

Au lieu de créer un nouveau `BacktestEngine` à chaque run, réutiliser la même instance.

**Solution** :
```python
# Au lieu de :
for params in params_grid:
    engine = BacktestEngine(...)  # Nouveau à chaque fois
    result = engine.run(...)

# Faire :
engine = BacktestEngine(...)
for params in params_grid:
    result = engine.run(...)  # Même instance
```

### 3. CACHER LA DÉTECTION DE GAPS 🟡 (Gain estimé: +30 runs/sec)

**Impact** : ~3ms → ~0ms

`detect_gaps(df)` est appelé à chaque run sur les **mêmes données**. Cacher le résultat.

**Solution** :
```python
# Dans engine.py
def run(self, df, strategy, params=None, *, skip_gap_detection=False, ...):
    if not skip_gap_detection:
        gaps_info = detect_gaps(df)
        self.logger.info(f"DATA_LOADED ... gaps_count={gaps_info.get('gaps_count', 0)} ...")
```

**Fichiers** :
- [backtest/engine.py](backtest/engine.py#L218-L227)

### 4. CACHER GIT COMMIT 🟡 (Gain estimé: +20 runs/sec)

**Impact** : ~2ms → ~0ms

`get_git_commit()` est appelé à chaque run. Le commit ne change pas pendant une grid search.

**Solution** :
```python
# Au niveau module
_CACHED_GIT_COMMIT = None

def get_git_commit():
    global _CACHED_GIT_COMMIT
    if _CACHED_GIT_COMMIT is None:
        _CACHED_GIT_COMMIT = _compute_git_commit()
    return _CACHED_GIT_COMMIT
```

**Fichiers** :
- [utils/version.py](utils/version.py) : Fonction get_git_commit()

### 5. OPTIMISER LE CALCUL DES MÉTRIQUES 🟡 (Gain estimé: +40 runs/sec)

**Impact** : 5.61ms → ~2ms

Le calcul des métriques prend 21% du temps. Optimisations possibles :

- Désactiver métriques non-essentielles en grid search
- Vectoriser calculs avec numpy
- Éviter copies de DataFrames

**Solution** :
```python
# Dans performance.py
def calculate_metrics(equity, returns, trades_df, *, minimal=False, ...):
    if minimal:
        # Calculer seulement sharpe, total_return, max_dd
        return {
            'sharpe_ratio': sharpe_ratio(returns, ...),
            'total_return_pct': (equity[-1] / equity[0] - 1) * 100,
            'max_drawdown': max_drawdown(equity)
        }
    else:
        # Calculs complets (20+ métriques)
        ...
```

**Fichiers** :
- [backtest/performance.py](backtest/performance.py) : Fonction calculate_metrics()

### 6. UTILISER SIMULATOR_FAST (NUMBA) 🟢 (Gain estimé: +10 runs/sec)

**Impact** : 1.89ms → ~0.5ms (déjà assez rapide)

Vérifier que `USE_FAST_SIMULATOR = True` et que Numba est installé.

**Fichiers** :
- [backtest/engine.py](backtest/engine.py#L36-L37)
- [backtest/simulator_fast.py](backtest/simulator_fast.py)

## Résumé des Optimisations

| Optimisation | Gain estimé | Complexité | Priorité |
|--------------|-------------|------------|----------|
| Désactiver logging | +300 runs/sec | Facile | 🔴 |
| Réutiliser engine | +50 runs/sec | Facile | 🔴 |
| Cacher gap detection | +30 runs/sec | Facile | 🟡 |
| Optimiser métriques | +40 runs/sec | Moyenne | 🟡 |
| Cacher git commit | +20 runs/sec | Facile | 🟡 |
| Simulator_fast | +10 runs/sec | Facile | 🟢 |

**Total gain estimé** : 37.4 → **~480 runs/sec** (objectif: 500)

## Actions Immédiates

1. ✅ **Désactiver logging** dans sharpe_ratio() (déjà fait)
2. 🔴 **Ajouter silent_mode au BacktestEngine.run()**
3. 🔴 **Documenter pattern de réutilisation d'engine**
4. 🟡 **Ajouter skip_gap_detection parameter**
5. 🟡 **Ajouter minimal=True au calculate_metrics()**

## Code Exemple Optimisé

```python
# Grid search optimisé
engine = BacktestEngine(initial_capital=10000, config=config)

for params in params_grid:
    result = engine.run(
        df=df,
        strategy='ema_cross',
        params=params,
        silent_mode=True,  # Désactive RUN_START, RUN_END_SUMMARY, etc.
        skip_gap_detection=True,  # Pas besoin de recheck les gaps
        minimal_metrics=True  # Calcule seulement sharpe, return, max_dd
    )

    sharpe = result.metrics['sharpe_ratio']
    # ...
```

**Gain estimé avec ce pattern** : 37.4 → **~450 runs/sec**
