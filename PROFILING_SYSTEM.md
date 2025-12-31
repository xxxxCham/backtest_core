# ✅ Système de Profiling - Backtest Core

**Date**: 29 décembre 2025
**Statut**: OPÉRATIONNEL

---

## 📋 Vue d'Ensemble

Système complet de **profiling de performance** pour identifier et optimiser les goulots d'étranglement (bottlenecks) du code.

### Problème Résolu

> "Exécuter un gros run et chronométrer chaque élément du code pour déterminer qu'est-ce qui prend du temps, qu'est-ce qui n'en prend pas et du coup pour déterminer où mettre davantage d'intentions pour optimiser la rapidité et l'efficacité des calculs"

---

## 🛠️ Outils Créés

| Fichier | Description | Usage |
|---------|-------------|-------|
| [tools/profiler.py](tools/profiler.py) | **Profiler principal** | Chronométre tous les appels de fonctions |
| [tools/profile_demo.py](tools/profile_demo.py) | **Démo avec données synthétiques** | Tester sans vraies données |
| [tools/profile_analyzer.py](tools/profile_analyzer.py) | **Analyseur HTML** | Génère rapports visuels interactifs |
| [tools/profile.bat](tools/profile.bat) | **Script Windows** | Raccourci ligne de commande |
| [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md) | **Documentation complète** | Guide complet (25+ exemples) |

---

## 🚀 Démarrage Rapide

### 1. Tester le Profiler (Données Synthétiques)

```bash
# Générer des données synthétiques et profiler
python tools/profile_demo.py
```

**Ce que ça fait** :
- ✅ Génère 8,761 barres OHLCV synthétiques (1 an de données 1h)
- ✅ Profile un backtest simple (stratégie EMA Cross)
- ✅ Profile une optimisation Grid Search (5 combinaisons)
- ✅ Sauvegarde 2 rapports .prof dans `profiling_results/`
- ✅ Affiche les TOP 20 fonctions les plus lentes

### 2. Générer un Rapport HTML

```bash
# Analyser le dernier rapport
python tools/profile_analyzer.py --report profiling_results/demo_simple_*.prof --output demo_analysis.html

# Ouvrir le rapport
start demo_analysis.html
```

**Ce que vous verrez** :
- 🔥 **TOP 30 Temps Cumulé** : Points d'entrée des zones lentes
- ⚡ **TOP 30 Temps Propre** : Véritables consommateurs de CPU
- 🔄 **TOP 20 Nombre d'Appels** : Fonctions appelées massivement

**Code couleur** :
- 🔴 **ROUGE** (>10%) : **OPTIMISER EN PRIORITÉ**
- 🟠 **ORANGE** (5-10%) : Optimisation recommandée
- 🟢 **VERT** (<5%) : Impact faible

---

## 📊 Utilisation avec Vraies Données

### Si vous avez des données de marché

```bash
# Profiler un backtest simple
python tools/profiler.py simple --strategy ema_cross --start 2024-01-01 --end 2024-12-31

# Profiler une optimisation Grid Search
python tools/profiler.py grid --strategy macd_cross --combinations 100

# Lister les stratégies disponibles
python tools/profiler.py list
```

### Si vous n'avez PAS de données

```bash
# Utiliser le demo avec données synthétiques
python tools/profile_demo.py
```

---

## 🎯 Scénarios d'Utilisation

### Scénario 1 : "Mon backtest est lent"

```bash
# 1. Profiler
python tools/profile_demo.py

# 2. Analyser
python tools/profile_analyzer.py --report profiling_results/demo_simple_*.prof --output analysis.html

# 3. Ouvrir
start analysis.html

# 4. Chercher les fonctions ROUGES (>10%)
# 5. Optimiser ces fonctions (vectorisation, cache, etc.)
```

### Scénario 2 : "Mon optimisation prend des heures"

```bash
# 1. Profiler avec un petit nombre de combinaisons
python tools/profiler.py grid --strategy ema_cross --combinations 20

# 2. Regarder la table "Nombre d'Appels" dans le HTML
# 3. Si une fonction est appelée >100,000 fois → cacher ou vectoriser
```

### Scénario 3 : "Je veux mesurer l'impact de mon optimisation"

```bash
# AVANT
python tools/profiler.py grid --strategy ema_cross --combinations 50
# Noter le temps total

# ... faire vos optimisations ...

# APRÈS
python tools/profiler.py grid --strategy ema_cross --combinations 50
# Comparer le temps total

# Objectif : -50% minimum, -75% excellent
```

---

## 📖 Documentation Complète

Voir [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md) pour :
- ✅ 25+ exemples concrets
- ✅ Guide d'interprétation des résultats
- ✅ Techniques d'optimisation (vectorisation, cache, GPU)
- ✅ Troubleshooting complet
- ✅ Checklist d'optimisation

---

## 🔍 Concepts Clés

### Temps Cumulé vs Temps Propre

| Métrique | Signification | Quand l'utiliser |
|----------|---------------|------------------|
| **Temps Cumulé** | Temps dans la fonction + tout ce qu'elle appelle | Trouver les **points d'entrée** lents |
| **Temps Propre** | Temps UNIQUEMENT dans cette fonction | Trouver les **véritables bottlenecks** |

**Exemple** :
```
backtest/engine.py:run() → 10s cumulé, 0.5s propre
  ↓ appelle
strategies/indicators.py:ema() → 5s cumulé, 5s propre ← OPTIMISER ICI
```

### Nombre d'Appels

Si une fonction est appelée 100,000 fois, même 0.1ms/appel = 10s total !

**Solutions** :
- ✅ Cache (`@lru_cache`)
- ✅ Vectorisation (NumPy/Pandas)
- ✅ Pré-calcul (hors de la boucle)

---

## 📁 Structure des Fichiers

```
backtest_core/
├── tools/
│   ├── profiler.py              # Profiler principal
│   ├── profile_demo.py          # Démo avec données synthétiques
│   ├── profile_analyzer.py      # Analyseur HTML
│   └── profile.bat              # Script Windows
├── docs/
│   └── PROFILING_GUIDE.md       # Guide complet (25+ pages)
├── profiling_results/           # Rapports .prof (créé automatiquement)
│   ├── demo_simple_*.prof
│   ├── demo_grid_*.prof
│   └── backtest_ema_cross_*.prof
└── PROFILING_SYSTEM.md          # Ce fichier
```

---

## ✅ Checklist Rapide

Après chaque profiling :

- [ ] Identifier les 3 fonctions avec le plus haut **temps cumulé**
- [ ] Identifier les 3 fonctions avec le plus haut **temps propre** → **OPTIMISER**
- [ ] Chercher les fonctions appelées >10,000 fois → Cache/vectorisation
- [ ] Chercher les boucles `for` Python → Vectoriser avec Pandas/NumPy
- [ ] Mesurer le temps AVANT optimisation
- [ ] Optimiser UNE fonction à la fois
- [ ] Re-profiler pour mesurer le gain
- [ ] Documenter les gains (avant/après)

---

## 🎓 Techniques d'Optimisation Rapides

### 1. Vectorisation NumPy/Pandas

**AVANT** (lent) :
```python
ema = []
for i in range(len(prices)):
    ema.append(calculate_ema_point(prices[:i]))
```

**APRÈS** (rapide) :
```python
ema = prices.ewm(span=period).mean()
```

**Gain** : 10x à 100x

### 2. Cache avec `@lru_cache`

**AVANT** :
```python
def calculate_indicator(df, period):
    # Calcul lourd répété
    return result
```

**APRÈS** :
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicator(df_hash, period):
    df = unhash(df_hash)
    return result
```

**Gain** : Jusqu'à 100x si appels répétitifs

### 3. Pré-calcul

**AVANT** :
```python
def on_bar(self, bar):
    ema = self.calculate_ema()  # Recalcul à chaque bar
    if bar.close > ema:
        self.buy()
```

**APRÈS** :
```python
def initialize(self):
    self.ema = self.calculate_ema()  # UNE FOIS

def on_bar(self, bar, idx):
    if bar.close > self.ema[idx]:
        self.buy()
```

**Gain** : N fois plus rapide (N = nombre de barres)

---

## 📞 Aide Rapide

### Commandes Rapides

```bash
# Lister les stratégies
python tools/profiler.py list

# Profiler simple (avec données synthétiques)
python tools/profile_demo.py

# Profiler avec vraies données
python tools/profiler.py simple --strategy ema_cross

# Profiler Grid Search
python tools/profiler.py grid --strategy macd_cross --combinations 50

# Analyser un rapport
python tools/profile_analyzer.py --report profiling_results/*.prof --output analysis.html
```

### Erreurs Courantes

| Erreur | Solution |
|--------|----------|
| `ModuleNotFoundError: No module named 'ui'` | Lancer depuis la racine du projet |
| `FileNotFoundError: data/BTCUSDT_1h.csv` | Utiliser `profile_demo.py` (données synthétiques) |
| Rapport .prof introuvable | Vérifier `dir profiling_results` |

---

## 🎯 Objectifs de Performance

Après optimisation :

| Métrique | Objectif | Excellent | Bon | À Améliorer |
|----------|----------|-----------|-----|-------------|
| **Temps total** | -50% | -75% | -50% | -25% |
| **Temps propre max** | < 10% total | < 5% | < 10% | > 15% |
| **Appels/s** | x2 minimum | x5 | x2 | x1.5 |

---

## 📈 Prochaines Étapes

1. **Profiler votre code actuel** :
   ```bash
   python tools/profile_demo.py
   ```

2. **Analyser le rapport HTML** :
   ```bash
   python tools/profile_analyzer.py --report profiling_results/demo_simple_*.prof --output analysis.html
   start analysis.html
   ```

3. **Identifier les fonctions rouges** (>10%)

4. **Optimiser UNE fonction** à la fois

5. **Re-profiler** pour mesurer le gain

6. **Répéter** jusqu'à atteindre les objectifs de performance

---

**Créé par** : Claude Sonnet 4.5
**Date** : 29 décembre 2025
**Projet** : Backtest Core v2.0
