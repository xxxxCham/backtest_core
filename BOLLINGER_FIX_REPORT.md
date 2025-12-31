# Résolution du Problème - Bandes de Bollinger

## 🔴 Problème Identifié

L'utilisateur observe dans l'UI Streamlit :
- **bb_period = 42** (période élevée)
- **bb_std = 2.25** (écart-type)

Mais le graphique ne montre PAS des bandes de Bollinger suffisamment éloignées du cours.

## 🔍 Diagnostic

### Cause Racine
Le graphique affiché est un **DIAGRAMME SYMBOLIQUE** (pas les vraies données), généré avec `_create_synthetic_price()` dans `ui/components/charts.py`.

### Problèmes identifiés:

1. **Données trop lisses** :
   ```python
   # Ancien code
   base = 100 + 4 * np.sin(...)  # Oscillation douce
   noise = 0.9 * np.sin(...)      # Bruit faible
   price = base + noise           # Volatilité quasi nulle
   ```
   → Sur des données lisses, l'écart-type est PETIT même avec période=42

2. **Nombre de points insuffisant** :
   - `n = 160` points de données
   - Période = 42 → seulement 118 points après warmup
   - Pas assez de contexte pour voir l'impact d'une période élevée

3. **Conséquence** :
   - Les bandes restent PROCHES du prix
   - L'impact visuel d'une période de 42 n'est PAS visible
   - L'utilisateur pense qu'il y a un bug de calcul

## ✅ Solution Implémentée

### 1. **Données synthétiques plus réalistes**

```python
def _create_synthetic_price(n: int = 160, volatility: float = 2.5) -> tuple:
    np.random.seed(42)
    x = np.arange(n)

    # Tendance de fond
    base = 100 + 4 * np.sin(np.linspace(0, 4 * np.pi, n))

    # Oscillations moyennes fréquences
    mid_freq = 0.9 * np.sin(np.linspace(0, 11 * np.pi, n))

    # 🆕 Marche aléatoire (réalisme)
    random_walk = np.random.randn(n).cumsum() * 0.3

    # 🆕 Chocs de volatilité (pics aléatoires)
    shocks = np.random.randn(n) * volatility

    # Composition finale
    price = base + mid_freq + random_walk + shocks
```

**Impact** : Volatilité augmentée → écart-type plus élevé → bandes plus larges

### 2. **Plus de points de données**

```python
# Ancien
n: int = 160

# Nouveau
n: int = 300  # Doublé pour périodes élevées
```

**Impact** : Plus de contexte pour les calculs rolling, meilleure visualisation

### 3. **Fichiers modifiés**

- `ui/components/charts.py` :
  - `_create_synthetic_price()` : Ajout random_walk + shocks
  - `_render_bollinger_atr_diagram()` : n=300
  - `_render_bollinger_atr_v2_diagram()` : n=300
  - `_render_bollinger_atr_v3_diagram()` : n=300

## 🎯 Résultat Attendu

Après ces modifications, le diagramme symbolique affichera :

1. **Prix plus volatils** avec des variations réalistes
2. **Bandes de Bollinger plus larges** quand bb_period=42
3. **Impact VISIBLE** d'une période élevée sur le graphique
4. **Meilleure correspondance** entre les paramètres et la visualisation

## 🔄 Pour Tester

1. Relancer Streamlit : `streamlit run ui/app.py`
2. Sélectionner stratégie `bollinger_atr`
3. Régler **bb_period = 42** et **bb_std = 2.25**
4. Observer le diagramme → Les bandes DEVRAIENT être plus éloignées

## 📝 Note Importante

**Ce graphique est SYMBOLIQUE** : Il montre la **logique de la stratégie**, pas les vraies données.

Pour voir les bandes sur **vraies données** :
1. Charger un fichier Parquet/CSV
2. Lancer le backtest
3. Consulter le graphique "OHLCV + indicateurs (aperçu)" dans les résultats

---

**Status** : ✅ Corrections appliquées
**Date** : 29/12/2025
**Fichiers modifiés** : 1 (ui/components/charts.py)
**Lignes modifiées** : ~15 lignes
