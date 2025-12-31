# 🔍 Guide de Profiling - Backtest Core

**Date**: 29 décembre 2025
**Version**: 1.0

---

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Utilisation Basique](#utilisation-basique)
4. [Scénarios de Profiling](#scénarios-de-profiling)
5. [Analyse des Résultats](#analyse-des-résultats)
6. [Optimisations Recommandées](#optimisations-recommandées)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

Le **profiling** consiste à chronométrer chaque fonction de votre code pour identifier les **goulots d'étranglement** (bottlenecks). Cela permet de savoir **où optimiser** pour améliorer les performances.

### Pourquoi profiler ?

- ✅ Identifier les fonctions les plus lentes
- ✅ Détecter les appels inutiles
- ✅ Prioriser les efforts d'optimisation
- ✅ Mesurer l'impact des modifications

### Concepts Clés

| Métrique | Signification | Usage |
|----------|---------------|-------|
| **Temps Cumulé** | Temps total dans la fonction + tout ce qu'elle appelle | Trouver les **points d'entrée** lents |
| **Temps Propre** | Temps UNIQUEMENT dans cette fonction (hors appels) | Trouver les **véritables consommateurs** de CPU |
| **Nombre d'Appels** | Combien de fois la fonction est appelée | Détecter les appels répétitifs |
| **Par Appel** | Temps moyen par appel | Mesurer l'efficacité unitaire |

---

## Installation

Aucune installation nécessaire ! Les outils sont déjà inclus :

```bash
# Vérifier que les scripts existent
dir tools\profiler.py
dir tools\profile_analyzer.py
dir tools\profile.bat
```

---

## Utilisation Basique

### 1. Lister les Stratégies Disponibles

```bash
python tools\profiler.py list
```

**Sortie** :
```
📋 Stratégies disponibles:
   - ema_cross
   - macd_cross
   - rsi_reversal
   - bollinger_atr
   - atr_channel
```

### 2. Profiler un Backtest Simple

```bash
python tools\profiler.py simple --strategy ema_cross
```

**Options** :
- `--symbol BTCUSDT` : Symbole à tester (défaut: BTCUSDT)
- `--timeframe 1h` : Timeframe (défaut: 1h)
- `--start 2024-01-01` : Date de début
- `--end 2024-12-31` : Date de fin

**Exemple avec options** :
```bash
python tools\profiler.py simple --strategy macd_cross --symbol ETHUSDT --timeframe 4h
```

### 3. Profiler une Optimisation Grid Search

```bash
python tools\profiler.py grid --strategy ema_cross --combinations 100
```

**Options** :
- `--combinations 100` : Nombre de combinaisons à tester
- `--symbol`, `--timeframe`, `--start`, `--end` : Idem que simple

**Exemple avec options** :
```bash
python tools\profiler.py grid --strategy rsi_reversal --combinations 50 --start 2024-06-01
```

### 4. Analyser un Rapport Existant

```bash
python tools\profiler.py analyze --report profiling_results\report_20250129_120000.prof
```

---

## Scénarios de Profiling

### Scénario 1 : Backtest Simple Lent

**Problème** : Un backtest simple prend trop de temps.

**Solution** :
```bash
# 1. Profiler le backtest
python tools\profiler.py simple --strategy ema_cross

# 2. Regarder les résultats dans le terminal
#    Chercher les fonctions avec temps cumulé > 1s

# 3. Générer un rapport HTML
python tools\profile_analyzer.py --report profiling_results\backtest_ema_cross_BTCUSDT_20250129_120000.prof --output analysis_simple.html

# 4. Ouvrir analysis_simple.html dans un navigateur
start analysis_simple.html
```

### Scénario 2 : Optimisation Grid Search Très Lente

**Problème** : Une optimisation avec 100 combinaisons prend des heures.

**Solution** :
```bash
# 1. Profiler avec un nombre réduit de combinaisons
python tools\profiler.py grid --strategy macd_cross --combinations 20

# 2. Analyser les résultats
python tools\profile_analyzer.py --report profiling_results\grid_macd_cross_20combos_20250129_120000.prof --output analysis_grid.html

# 3. Identifier les fonctions appelées massivement
#    (regarder la table "Nombre d'Appels" dans le HTML)

# 4. Optimiser ces fonctions, puis re-tester
```

### Scénario 3 : Comparaison Avant/Après Optimisation

**Problème** : Vous avez optimisé du code, vous voulez mesurer l'impact.

**Solution** :
```bash
# AVANT optimisation
python tools\profiler.py grid --strategy ema_cross --combinations 50
# Sauvegarder le rapport: grid_ema_cross_50combos_AVANT.prof

# ... faire vos optimisations ...

# APRÈS optimisation
python tools\profiler.py grid --strategy ema_cross --combinations 50
# Sauvegarder le rapport: grid_ema_cross_50combos_APRES.prof

# Comparer les deux rapports HTML côte à côte
```

---

## Analyse des Résultats

### Interpréter le Terminal

**Exemple de sortie** :
```
================================================================================
TOP 30 FONCTIONS LES PLUS LENTES (tri: cumulative)
================================================================================

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      500    0.050    0.000   10.500    0.021 backtest/engine.py:123(run)
    50000    5.200    0.000    5.200    0.000 strategies/indicators.py:45(ema)
      500    0.300    0.001    3.800    0.008 backtest/engine.py:200(_process_signals)
   250000    2.100    0.000    2.100    0.000 {built-in method pandas.core.series.Series.rolling}
```

**Lecture** :
- **Ligne 1** : `backtest/engine.py:123(run)` - 10.5s cumulé → Point d'entrée lent
- **Ligne 2** : `strategies/indicators.py:45(ema)` - 5.2s propre → **Goulot critique** (optimiser en priorité !)
- **Ligne 3** : `backtest/engine.py:200(_process_signals)` - 3.8s cumulé → Appelle des fonctions lentes
- **Ligne 4** : 250,000 appels à `rolling()` → Peut-on réduire les appels ?

### Interpréter le Rapport HTML

Ouvrez le fichier HTML généré dans un navigateur :

1. **Section "Temps Cumulé"** (🔥 rouge = >10%)
   - Fonctions "points d'entrée" des zones lentes
   - Regarder quelles fonctions elles appellent (section suivante)

2. **Section "Temps Propre"** (⚡ rouge = >10%)
   - **Cibles prioritaires d'optimisation**
   - Ces fonctions consomment réellement du CPU

3. **Section "Nombre d'Appels"** (🔄)
   - Fonctions appelées massivement
   - Peut-on cacher les résultats ?
   - Peut-on vectoriser ?

---

## Optimisations Recommandées

### 1. Vectorisation NumPy/Pandas

**Problème** : Boucles `for` Python lentes.

**Avant** :
```python
def calculate_ema(prices, period):
    ema = []
    for i in range(len(prices)):
        # Calcul ligne par ligne
        ...
    return ema
```

**Après** :
```python
def calculate_ema(prices, period):
    # Vectorisé avec pandas
    return prices.ewm(span=period).mean()
```

**Gain** : 10x à 100x plus rapide

### 2. Cache avec `@lru_cache`

**Problème** : Fonction appelée 10,000 fois avec les mêmes arguments.

**Avant** :
```python
def calculate_indicator(df, period):
    # Calcul lourd
    return result
```

**Après** :
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicator(df_hash, period):
    df = unhash(df_hash)  # Récupérer le DataFrame
    # Calcul lourd
    return result
```

**Gain** : Jusqu'à 100x si appels répétitifs

### 3. Pré-calcul des Indicateurs

**Problème** : Indicateurs recalculés à chaque signal.

**Avant** :
```python
def on_bar(self, bar):
    ema = self.calculate_ema()  # Recalcul à chaque bar
    if bar.close > ema:
        self.buy()
```

**Après** :
```python
def initialize(self):
    self.ema = self.calculate_ema()  # Calcul UNE FOIS

def on_bar(self, bar, idx):
    if bar.close > self.ema[idx]:
        self.buy()
```

**Gain** : N fois plus rapide (N = nombre de barres)

### 4. Utiliser CuPy pour GPU (Optionnel)

Si vous avez un GPU NVIDIA :

```python
import cupy as cp

# Avant (CPU)
import numpy as np
arr = np.array([...])
result = np.sum(arr)

# Après (GPU)
import cupy as cp
arr = cp.array([...])
result = cp.sum(arr)  # 10-100x plus rapide
```

---

## Troubleshooting

### Erreur : `ModuleNotFoundError: No module named 'ui'`

**Solution** : Le profiler configure automatiquement le PYTHONPATH. Si l'erreur persiste :
```bash
# Depuis la racine du projet
set PYTHONPATH=%CD%
python tools\profiler.py simple --strategy ema_cross
```

### Erreur : `FileNotFoundError: data/BTCUSDT_1h.csv`

**Solution** : Téléchargez d'abord les données :
```bash
python data/download_ohlcv.py BTCUSDT 1h
```

### Le rapport `.prof` est introuvable

**Solution** : Vérifiez le répertoire `profiling_results/` :
```bash
dir profiling_results
```

Les rapports sont nommés : `backtest_{strategy}_{symbol}_{timestamp}.prof`

### Le HTML ne s'affiche pas correctement

**Solution** : Ouvrez avec un navigateur moderne (Chrome, Firefox, Edge) :
```bash
start chrome analysis.html
```

---

## Raccourcis Rapides

### Windows Batch Script

```batch
REM Profiler simple
tools\profile simple --strategy ema_cross

REM Profiler grid (50 combinaisons)
tools\profile grid --strategy macd_cross --combinations 50

REM Analyser le dernier rapport
python tools\profile_analyzer.py --report profiling_results\*.prof --output last_analysis.html
```

### PowerShell One-Liners

```powershell
# Profiler + analyser en une commande
python tools\profiler.py simple --strategy ema_cross; `
$latest = Get-ChildItem profiling_results\*.prof | Sort-Object LastWriteTime -Descending | Select-Object -First 1; `
python tools\profile_analyzer.py --report $latest.FullName --output analysis.html; `
Start-Process analysis.html
```

---

## Exemples Complets

### Exemple 1 : Profiler EMA Cross

```bash
# 1. Profiler un backtest simple
python tools\profiler.py simple --strategy ema_cross --start 2024-01-01 --end 2024-12-31

# 2. Générer le rapport HTML
python tools\profile_analyzer.py --report profiling_results\backtest_ema_cross_BTCUSDT_*.prof --output ema_analysis.html

# 3. Ouvrir le rapport
start ema_analysis.html

# 4. Identifier les fonctions > 10% (rouges)
# 5. Optimiser ces fonctions
# 6. Re-profiler pour vérifier l'amélioration
```

### Exemple 2 : Profiler Grid Search MACD

```bash
# 1. Profiler une petite grille (20 combos pour diagnostic rapide)
python tools\profiler.py grid --strategy macd_cross --combinations 20

# 2. Analyser
python tools\profile_analyzer.py --report profiling_results\grid_macd_cross_20combos_*.prof --output macd_grid_analysis.html

# 3. Ouvrir
start macd_grid_analysis.html

# 4. Regarder la section "Nombre d'Appels"
#    Si une fonction est appelée > 100,000 fois → candidat au cache/vectorisation
```

---

## 📊 Métriques de Succès

Après optimisation, re-profiler et comparer :

| Métrique | Objectif | Excellent | Bon | À Améliorer |
|----------|----------|-----------|-----|-------------|
| **Temps total** | -50% | -75% | -50% | -25% |
| **Temps propre max** | < 10% total | < 5% | < 10% | > 15% |
| **Appels/s** | x2 minimum | x5 | x2 | x1.5 |

---

## 🎯 Checklist d'Optimisation

Après chaque profiling :

- [ ] Identifier les 3 fonctions avec le plus haut temps cumulé
- [ ] Identifier les 3 fonctions avec le plus haut temps propre
- [ ] Vérifier si des fonctions sont appelées > 10,000 fois
- [ ] Chercher les boucles `for` Python (candidats à vectorisation)
- [ ] Vérifier les appels répétitifs avec mêmes arguments (candidats au cache)
- [ ] Mesurer le temps avant optimisation
- [ ] Optimiser UNE fonction à la fois
- [ ] Re-profiler après chaque optimisation
- [ ] Documenter les gains (temps avant/après)

---

**Créé par** : Claude Sonnet 4.5
**Date** : 29 décembre 2025
**Projet** : Backtest Core v2.0
