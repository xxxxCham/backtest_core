# 🔍 Packages Optionnels et Optimisations

Guide des packages manquants qui pourraient améliorer les performances et résoudre certains problèmes.

---

## ⚠️ Packages Manquants Détectés

### 🚀 **Performance Critiques** (Recommandés)

#### 1. **Bottleneck** - Accélération Pandas

**Problème résolu**: Ralentissements sur rolling/groupby avec gros datasets

```bash
pip install bottleneck
```

**Impact**:
- ✅ Accélère `rolling()`, `rank()`, `median()` de 5-20x
- ✅ Utilisé automatiquement par Pandas si présent
- ✅ Particulièrement utile pour calculs Bollinger/EMA/ATR

**Utilisation**: Automatique une fois installé

---

#### 2. **Numexpr** - Évaluation Rapide d'Expressions

**Problème résolu**: Calculs NumPy lents sur grandes matrices

```bash
pip install numexpr
```

**Impact**:
- ✅ Accélère expressions complexes de 2-10x
- ✅ Utilise multi-threading automatique
- ✅ Réduit usage mémoire sur gros arrays

**Exemple**:
```python
# Avant (NumPy standard): 100ms
result = (df['close'] - df['low']) / (df['high'] - df['low'])

# Après (avec numexpr): 15ms
# Pandas utilise automatiquement numexpr si installé
```

---

### 📊 **Analyse Statistique Avancée** (Optionnel)

#### 3. **Statsmodels** - Modèles Statistiques

**Problème résolu**: Calculs statistiques avancés (autocorrélation, régression)

```bash
pip install statsmodels
```

**Cas d'usage**:
- Analyse autocorrélation des résidus de stratégie
- Tests de stationnarité (ADF test)
- Régression pour analyse factorielle
- ARIMA/GARCH pour prédictions

**Utilisation**:
```python
from statsmodels.tsa.stattools import adfuller

# Test si une série est stationnaire
result = adfuller(equity_curve)
print(f"P-value: {result[1]}")  # < 0.05 = stationnaire
```

---

#### 4. **Scikit-learn** - Machine Learning

**Problème résolu**: Validation croisée avancée, clustering de stratégies

```bash
pip install scikit-learn
```

**Cas d'usage**:
- Walk-forward validation robuste
- Clustering de patterns de marché
- Feature engineering pour stratégies ML
- Cross-validation time-series

**Utilisation**:
```python
from sklearn.model_selection import TimeSeriesSplit

# Walk-forward validation propre
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    # Backtest sur train, valider sur test
```

---

### ⚡ **Compilation & Performance Extrême** (Expert)

#### 5. **Cython** - Compilation C

**Problème résolu**: Boucles Python très lentes

```bash
pip install cython
```

**Impact**:
- ✅ Compile Python en C pour vitesse native
- ✅ Utile pour indicateurs custom complexes
- ✅ Peut donner 10-100x speedup sur boucles

**Quand l'utiliser**:
- Indicateurs custom avec boucles imbriquées
- Calculs de géométrie de marché complexes
- Simulations Monte Carlo intensives

---

## 🔧 Problèmes Connus et Solutions

### ❌ **Problème 1: Calculs Incorrects sur Petits Datasets**

**Symptôme**: Sharpe Ratio NaN ou infini sur < 30 trades

**Cause**: Division par zéro ou variance nulle

**Solution**:
```python
# Dans metrics_tier_s.py, ajouter validation
if len(returns) < 30:
    logger.warning("Moins de 30 trades, Sharpe Ratio peut être imprécis")
    return 0.0

std = returns.std()
if std < 1e-10:  # Variance quasi-nulle
    return 0.0
```

---

### ❌ **Problème 2: Overflow/Underflow dans Calculs**

**Symptôme**: `RuntimeWarning: overflow encountered in multiply`

**Cause**: Valeurs trop grandes (leverage élevé, prix crypto)

**Solution**:
```python
# Activer gestion des erreurs NumPy
import numpy as np
np.seterr(all='warn')  # Afficher warnings
np.seterr(over='raise')  # Lever exception sur overflow

# Ou utiliser float64 explicitement
df['close'] = df['close'].astype(np.float64)
```

---

### ❌ **Problème 3: Ralentissements sur Grid Search**

**Symptôme**: Grid search > 1h pour 10k combinaisons

**Solutions prioritaires**:

1. **Installer bottleneck + numexpr** (gain immédiat)
   ```bash
   pip install bottleneck numexpr
   ```

2. **Utiliser Numba sur calculs critiques**
   ```python
   from numba import jit

   @jit(nopython=True, cache=True)
   def calculate_signals_fast(close, upper, lower):
       # Code vectorisé pur NumPy
       return signals
   ```

3. **Activer cache Numba**
   ```bash
   # Windows
   set NUMBA_CACHE_DIR=d:\backtest_core\.numba_cache

   # Linux/macOS
   export NUMBA_CACHE_DIR=/path/to/backtest_core/.numba_cache
   ```

---

### ❌ **Problème 4: Précision Décimale sur Prix Crypto**

**Symptôme**: Erreurs d'arrondi sur tokens à faible prix (0.0001 USDT)

**Solution**:
```python
from decimal import Decimal, getcontext

# Précision 28 décimales
getcontext().prec = 28

# Utiliser Decimal pour calculs critiques
entry_price = Decimal(str(price))
quantity = capital / entry_price
```

---

### ❌ **Problème 5: Mémoire Saturée sur Gros Datasets**

**Symptôme**: `MemoryError` sur datasets > 1M lignes

**Solutions**:

1. **Utiliser Pandas chunking**
   ```python
   chunk_size = 100000
   for chunk in pd.read_csv('huge_data.csv', chunksize=chunk_size):
       result = backtest(chunk)
   ```

2. **Downcast dtypes**
   ```python
   df['close'] = df['close'].astype(np.float32)  # 32-bit au lieu de 64
   df['volume'] = pd.to_numeric(df['volume'], downcast='unsigned')
   ```

3. **Utiliser Parquet au lieu de CSV**
   ```python
   # Parquet = 5-10x moins d'espace + plus rapide
   df.to_parquet('data.parquet', compression='snappy')
   df = pd.read_parquet('data.parquet')
   ```

---

## 📦 Installation Complète Recommandée

### Pour Usage Standard

```bash
pip install bottleneck numexpr
```

### Pour Analyse Avancée

```bash
pip install bottleneck numexpr statsmodels scikit-learn
```

### Pour Performance Extrême

```bash
pip install bottleneck numexpr statsmodels scikit-learn cython
```

---

## ✅ Script de Vérification Complet

Créez `check_performance.py`:

```python
#!/usr/bin/env python3
"""Vérifie packages de performance installés."""

packages_performance = {
    'bottleneck': 'Accélération Pandas rolling/groupby',
    'numexpr': 'Évaluation rapide expressions NumPy',
    'statsmodels': 'Modèles statistiques avancés',
    'sklearn': 'Machine learning / validation croisée',
    'cython': 'Compilation C pour boucles critiques',
}

print("=" * 60)
print("PACKAGES DE PERFORMANCE")
print("=" * 60)

missing = []
for pkg, desc in packages_performance.items():
    try:
        __import__(pkg)
        print(f"✅ {pkg:15s} - {desc}")
    except ImportError:
        print(f"❌ {pkg:15s} - {desc}")
        missing.append(pkg)

if missing:
    print(f"\n⚠️  {len(missing)} package(s) manquant(s)")
    print(f"Installation: pip install {' '.join(missing)}")
else:
    print("\n✅ Tous les packages de performance installés!")
```

---

## 🎯 Recommandations par Priorité

### 🔴 **PRIORITÉ 1 - À Installer MAINTENANT**
- ✅ `bottleneck` - Gain immédiat 5-20x sur rolling
- ✅ `numexpr` - Gain 2-10x sur calculs complexes

**Installation**:
```bash
pip install bottleneck numexpr
```

### 🟡 **PRIORITÉ 2 - Si Problèmes d'Analyse**
- `statsmodels` - Tests statistiques robustes
- `scikit-learn` - Walk-forward validation propre

### 🟢 **PRIORITÉ 3 - Optimisation Avancée**
- `cython` - Compilation custom code
- `cupy` - GPU acceleration (si NVIDIA GPU)

---

## 🔍 Diagnostic Rapide

```bash
# Test si packages critiques manquent
python -c "import bottleneck; print('✅ Bottleneck OK')" 2>&1 || echo "❌ Installer: pip install bottleneck"
python -c "import numexpr; print('✅ Numexpr OK')" 2>&1 || echo "❌ Installer: pip install numexpr"
```

---

**Dernière mise à jour**: 2025-01-XX
