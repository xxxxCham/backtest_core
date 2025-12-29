# Optimisations de Performance Appliquées

## Résumé

**Performance avant** : 37.4 runs/sec
**Performance après** : 115 runs/sec
**Gain** : **3x plus rapide** (+207% de performance)

## 1. Désactivation du Logging Structuré (✅ Implémenté)

### Problème Identifié

Le profiling a révélé que **71% du temps d'exécution** était perdu dans l'overhead, principalement :

- Logging structuré (RUN_START, DATA_LOADED, PARAMS_RESOLVED, RUN_END_SUMMARY)
- Détection de gaps sur chaque run
- Lookup git commit à chaque itération
- Formatage de strings coûteux avec f-strings

### Solution Implémentée

Ajout d'un paramètre `silent_mode` au BacktestEngine :

#### Fichiers Modifiés

**[backtest/engine.py](backtest/engine.py)**
- Ligne 165 : Ajout paramètre `silent_mode: bool = False`
- Lignes 199-212 : Désactivation de `pipeline_start` et `RUN_START` en silent_mode
- Lignes 223-233 : Désactivation de `DATA_LOADED` et `detect_gaps()` en silent_mode
- Lignes 251-259 : Désactivation de `PARAMS_RESOLVED` en silent_mode
- Lignes 342-367 : Désactivation de `pipeline_end` et `RUN_END_SUMMARY` en silent_mode

**[ui/app.py](ui/app.py)**
- Ligne 1075 : Ajout paramètre `silent_mode` à `safe_run_backtest()`
- Ligne 1099 : Propagation de `silent_mode` à `engine.run()`
- Lignes 2723, 2813, 2973, 3086, 3384, 3537 : Connexion au bouton DEBUG UI

### Intégration avec l'UI

Le bouton **"Mode DEBUG"** dans la sidebar (🔧 Debug) contrôle maintenant :

1. **Niveau de logs Python** (existant)
   - DEBUG activé → `set_log_level("DEBUG")`
   - DEBUG désactivé → `set_log_level("INFO")`

2. **Logs structurés du BacktestEngine** (NOUVEAU)
   - DEBUG activé → `silent_mode=False` (tous les logs RUN_START, etc.)
   - DEBUG désactivé → `silent_mode=True` (performance maximale)

### Utilisation

```python
# Mode par défaut (interface UI - silent_mode=True)
# → Performance optimale pour grid searches
result = engine.run(df, strategy, params, silent_mode=True)

# Mode debug (logs complets)
# → Utile pour diagnostiquer des problèmes
result = engine.run(df, strategy, params, silent_mode=False)
```

### Impact Mesuré

| Composant | Avant | Après | Gain |
|-----------|-------|-------|------|
| **Total par run** | 26.8ms | 8.7ms | **-67%** |
| Overhead | ~20ms | ~0ms | **-100%** |
| Metrics | 5.6ms | 4.4ms | -21% |
| Simulation | 1.9ms | 2.2ms | +15% |
| **Runs/sec** | **37.4** | **115** | **+207%** |

## 2. Optimisations du Logging dans sharpe_ratio() (✅ Déjà fait)

**[backtest/performance.py](backtest/performance.py)**

Désactivation de ~15 logs par backtest dans la fonction sharpe_ratio() :
- SHARPE_INPUT, SHARPE_SANITY, SHARPE_CALC, SHARPE_OUTPUT
- Warnings de fallback (daily_resample, DatetimeIndex, etc.)
- Warnings de validation (min_samples, low_volatility, etc.)

## 3. Prochaines Optimisations (À Implémenter)

### 3.1 Réutilisation du BacktestEngine (Gain estimé: +30%)

Actuellement, un nouvel engine est créé à chaque run :

```python
# ACTUEL (lent)
for params in params_grid:
    engine = BacktestEngine(...)  # Nouveau à chaque fois
    result = engine.run(...)

# OPTIMISÉ (à implémenter)
engine = BacktestEngine(...)
for params in params_grid:
    result = engine.run(...)  # Réutiliser
```

### 3.2 Cache pour detect_gaps() (Gain estimé: +15%)

Actuellement, `detect_gaps(df)` est recalculé sur les mêmes données. Implémenter un cache.

### 3.3 Optimisation des Métriques (Gain estimé: +20%)

Les métriques prennent **52% du temps**. Options :
- Mode minimal (seulement sharpe, return, max_dd)
- Vectorisation numpy plus agressive
- Éviter copies de DataFrames

### 3.4 Cache Git Commit (Gain estimé: +5%)

`get_git_commit()` est appelé à chaque run. Cacher au niveau module.

## Objectif Final

**Cible** : 500 runs/sec
**Actuel** : 115 runs/sec
**Gap restant** : +335% à gagner

Avec les optimisations 3.1-3.4 implémentées, on devrait atteindre **~250-300 runs/sec**.

Pour atteindre 500 runs/sec, il faudrait :
- Utiliser systématiquement simulator_fast (Numba)
- Optimiser les stratégies (calcul d'indicateurs)
- Parallélisation multi-core

## Tests

✅ **39 tests passent** - Aucune régression fonctionnelle

## Utilisation dans l'Interface

1. Ouvrir l'interface : `streamlit run ui/app.py`
2. Dans la sidebar, section **🔧 Debug**
3. **Décocher "Mode DEBUG"** pour performances maximales (défaut)
4. **Cocher "Mode DEBUG"** pour diagnostiquer des problèmes

Le changement est **immédiat** et s'applique à tous les backtests (simple, grille, LLM).
