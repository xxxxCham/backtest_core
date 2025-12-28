# 🎯 Guide des Nouvelles Fonctionnalités

> Comment utiliser les nouveaux packages de visualisation et utilitaires dans Backtest Core

**Date**: 2025-01-XX
**Packages activés**: tqdm, colorama, tabulate, seaborn, plotly-resampler

---

## 🚀 Démarrage Rapide

### 1. Lancer l'interface Streamlit

```bash
cd d:\backtest_core
streamlit run ui/app.py
```

### 2. Exécuter un backtest standard

1. Sélectionnez une stratégie (ex: `bollinger_atr_v3`)
2. Chargez des données (ex: `BTCUSDT_1h.parquet`)
3. Cliquez sur **"Lancer Backtest"**

---

## 📊 Nouvelles Visualisations Disponibles

### 1. **Courbe d'Équité + Drawdown** (NOUVEAU !)

**Où**: Section **"💰 Courbe d'Équité"** (juste après les métriques principales)

**Ce que vous verrez**:
- Graphique à 2 panneaux:
  - **Haut**: Courbe d'équité avec zone remplie
  - **Bas**: Drawdown en temps réel (zones rouges)
- Ligne horizontale du capital initial
- Zoom/pan interactif
- **Downsampling automatique** si >100k points (plotly-resampler)

**Caractéristiques**:
- Style sombre adapté à Streamlit
- Hovertemplate avec détails
- Affichage fluide même avec gros datasets

---

### 2. **Analyse Statistique Avancée** (NOUVEAU !)

**Où**: Cliquez sur **"📊 Analyse Statistique Avancée"** (expander, juste avant l'historique des trades)

**Ce que vous verrez**:

#### Colonne de gauche: Distribution des P&L par Trade
- **Histogramme** + **KDE** (Kernel Density Estimation) avec seaborn
- Lignes verticales:
  - **Orange**: Moyenne des P&L
  - **Bleu**: Médiane des P&L
  - **Blanc pointillé**: Zéro (breakeven)
- Style sombre professionnel

#### Colonne de droite: Distribution des Rendements
- **Histogramme** + **KDE** des rendements périodiques
- Lignes verticales:
  - **Vert**: Moyenne des rendements
  - **Rouge pointillé**: ±1 écart-type (σ)
  - **Blanc pointillé**: Zéro

**Utilité**:
- Identifier l'asymétrie (skewness) des gains/pertes
- Détecter les outliers
- Vérifier la normalité de la distribution

---

## 🎨 Améliorations Visuelles Automatiques

### 1. **Logs Colorés** (colorama)

**Où**: Console / Terminal

**Ce que vous verrez**:
```
09:45:23 | DEBUG    | backtest.engine | Loading data...        [CYAN]
09:45:24 | INFO     | backtest.engine | Backtest complete      [VERT]
09:45:25 | WARNING  | strategies.base | Low trades count       [JAUNE]
09:45:26 | ERROR    | data.loader     | File not found         [ROUGE]
```

**Activation**: Automatique si colorama installé

---

### 2. **Barres de Progression** (tqdm)

**Où**:
- Mode LLM: Test des propositions d'agents
- Grid Search: Itération sur les combinaisons de paramètres

**Ce que vous verrez**:
```
Testing proposals: 100%|████████████| 5/5 [00:12<00:00,  2.5s/proposal]
Simulating trades: 100%|███████████| 8640/8640 [00:02<00:00, 3521bar/s]
```

**Activation**: Automatique si tqdm installé

---

### 3. **Tableaux Formatés** (tabulate)

**Où**: Métriques Tier S (mode console ou rapports)

**Avant** (ASCII box):
```
╔══════════════════════════════╗
║  Sortino Ratio:    2.453     ║
╚══════════════════════════════╝
```

**Après** (tabulate):
```
======================================================================
  RATIOS DE RISQUE AJUSTÉ
──────────────────────────────────────────────────────────────────────
Sortino Ratio        2.453
Calmar Ratio         1.823
SQN (Van Tharp)      3.142
Martin Ratio (UPI)   2.781
======================================================================
```

**Utilisation**:
```python
from backtest.metrics_tier_s import calculate_tier_s_metrics, format_tier_s_report

metrics = calculate_tier_s_metrics(returns, equity, trades_pnl)
print(format_tier_s_report(metrics))  # Tableau élégant automatique
```

---

### 4. **Downsampling Intelligent** (plotly-resampler)

**Où**: Automatique sur tous les graphiques Plotly si >100k points

**Comportement**:
- Dataset **< 100k points**: Affichage normal
- Dataset **≥ 100k points**:
  - Downsample automatique à 2000 points affichés
  - Zoom interactif recalcule le downsampling
  - Fluide même sur datasets massifs

**Message dans les logs**:
```
INFO | charts | Dataset large (250,000 points) - Activation du resampler
```

---

## 🔍 Vérification des Packages

### Tester tous les packages installés

```bash
python -c "import tqdm; import colorama; import tabulate; import seaborn; import plotly_resampler; print('✅ Tous les packages OK')"
```

### Vérifier les versions

```bash
python diagnose.py
```

**Attendu**:
```
[1/6] Vérification Python...
  ✅ Python 3.12.x

[2/6] Vérification Packages...
  ✅ Tous les packages critiques installés
  ✅ Packages performance installés

✅ SYSTÈME OPTIMAL - Aucun problème détecté
```

---

## 📈 Exemples d'Utilisation

### Exemple 1: Voir la distribution des P&L

1. Lancez un backtest avec une stratégie
2. Scrollez jusqu'à **"📊 Analyse Statistique Avancée"**
3. Cliquez pour ouvrir l'expander
4. **Graphique de gauche** = Distribution P&L
   - Si asymétrie vers la droite → Bons gros gains rares
   - Si asymétrie vers la gauche → Gros pertes rares
   - Si symétrique → Distribution équilibrée

### Exemple 2: Analyser le drawdown

1. Lancez un backtest
2. Regardez **"💰 Courbe d'Équité"**
3. **Graphique du bas** = Drawdown
   - Zones rouges = Périodes de perte
   - Plus la zone est profonde = Plus gros drawdown
   - Durée de la zone = Temps de récupération

### Exemple 3: Utiliser les métriques Tier S

```python
from backtest.metrics_tier_s import calculate_tier_s_metrics, format_tier_s_report

# Calculer après un backtest
metrics = calculate_tier_s_metrics(
    returns=result.returns,
    equity=result.equity,
    trades_pnl=result.trades['pnl'],
    initial_capital=10000.0
)

# Afficher le rapport
print(format_tier_s_report(metrics))

# Accéder aux valeurs individuelles
print(f"SQN: {metrics.sqn:.2f}")  # System Quality Number
print(f"Calmar: {metrics.calmar_ratio:.2f}")
print(f"Grade: {metrics.tier_s_grade}")  # A, B, C, D, F
```

---

## 🎯 Cas d'Usage Avancés

### Cas 1: Optimisation LLM avec Progress Bars

**Scénario**: Mode LLM multi-agents avec 10 itérations

**Ce que vous verrez**:
```bash
Testing proposals: 40%|████▍     | 2/5 [00:08<00:12,  4.2s/proposal]
```

**Avantage**: Savoir combien de temps reste avant la fin

---

### Cas 2: Gros Dataset (1M+ lignes)

**Scénario**: Backtest sur données 1m (1 minute) sur 2 ans = ~1M lignes

**Comportement**:
- Chargement: Normal
- Calculs: Accélérés par **bottleneck** + **numexpr** (5-20x plus rapide)
- Affichage graphique: **plotly-resampler** réduit à 2000 points
- Zoom: Recalcule automatiquement pour plus de détails

**Message**:
```
INFO | charts | Dataset large (1,051,200 points) - Activation du resampler
```

---

### Cas 3: Analyse Post-Backtest

**Scénario**: Comparer 3 stratégies différentes

```python
from ui.components.charts import render_comparison_chart

results_list = [
    {"name": "Strategy A", "metrics": {"sharpe_ratio": 2.1}},
    {"name": "Strategy B", "metrics": {"sharpe_ratio": 1.8}},
    {"name": "Strategy C", "metrics": {"sharpe_ratio": 2.5}},
]

render_comparison_chart(
    results_list=results_list,
    metric="sharpe_ratio",
    title="Comparaison Sharpe Ratio",
    key="comparison_sharpe"
)
```

---

## ⚙️ Configuration Optionnelle

### Désactiver les barres de progression

```python
# Dans simulator.py
simulate_trades(df, signals, params, show_progress=False)  # Pas de barre
```

### Désactiver les couleurs (logs)

Si les couleurs ne s'affichent pas correctement:
```bash
# Windows PowerShell
$env:NO_COLOR=1
streamlit run ui/app.py

# Linux/macOS
NO_COLOR=1 streamlit run ui/app.py
```

### Changer le seuil de downsampling

```python
# Dans ui/components/charts.py
RESAMPLER_THRESHOLD = 50000  # Au lieu de 100000
```

---

## 🐛 Dépannage

### Problème: "Seaborn non disponible"

**Solution**:
```bash
pip install seaborn>=0.12.0
```

### Problème: Pas de couleurs dans les logs

**Cause**: colorama non installé ou terminal incompatible

**Solutions**:
```bash
# 1. Installer colorama
pip install colorama

# 2. Ou désactiver (voir Configuration)
```

### Problème: Graphiques lents sur gros datasets

**Vérification**:
```python
import plotly_resampler
print(plotly_resampler.__version__)  # Doit afficher 0.11.0 ou +
```

**Si absent**:
```bash
pip install plotly-resampler>=0.9.0
```

---

## 📝 Résumé des Changements Visibles

| Fonctionnalité | Où la voir | Package utilisé |
|----------------|------------|-----------------|
| **Courbe d'équité + Drawdown** | Section "💰 Courbe d'Équité" | Plotly + plotly-resampler |
| **Distribution P&L** | Expander "📊 Analyse Statistique" (gauche) | Seaborn + Matplotlib |
| **Distribution Rendements** | Expander "📊 Analyse Statistique" (droite) | Seaborn + Matplotlib |
| **Logs colorés** | Console/Terminal | colorama |
| **Barres de progression** | Mode LLM / Grid Search | tqdm |
| **Tableaux formatés** | Métriques Tier S (console) | tabulate |
| **Downsampling automatique** | Tous les graphiques >100k points | plotly-resampler |
| **Accélération Pandas** | Transparent (auto) | bottleneck + numexpr |

---

## ✅ Checklist de Vérification

Après avoir lancé un backtest, vous devriez voir:

- [ ] Courbe d'équité avec zone remplie verte
- [ ] Graphique de drawdown en dessous (zones rouges)
- [ ] Expander "📊 Analyse Statistique Avancée" cliquable
- [ ] 2 graphiques dans l'expander (P&L + Rendements)
- [ ] Logs colorés dans le terminal (si visible)
- [ ] Message de downsampling si >100k points

**Si tous les éléments sont présents** → ✅ Installation réussie !

---

## 🎓 Pour Aller Plus Loin

### Documentation des packages

- **Seaborn**: https://seaborn.pydata.org/
- **plotly-resampler**: https://github.com/predict-idlab/plotly-resampler
- **tqdm**: https://tqdm.github.io/
- **colorama**: https://github.com/tartley/colorama
- **tabulate**: https://github.com/astanin/python-tabulate

### Documentation interne

- [INTEGRATION_PACKAGES.md](INTEGRATION_PACKAGES.md) - Détails techniques de l'intégration
- [PACKAGES_OPTIONNELS.md](../PACKAGES_OPTIONNELS.md) - Liste complète des packages
- [README.md](../README.md) - Vue d'ensemble du projet

---

**Bon backtesting ! 🚀**
