# 📦 Installation Guide - Backtest Core

Guide d'installation complet pour cloner et exécuter le projet sur n'importe quel ordinateur.

---

## 🎯 Prérequis

- **Python 3.10+** (testé avec Python 3.12)
- **Git** installé
- **8 GB RAM** minimum (16 GB recommandé pour optimisations)
- **Connexion Internet** (pour téléchargement des dépendances)

---

## 🚀 Installation Rapide (3 étapes)

### 1️⃣ Cloner le repository

```bash
git clone https://github.com/VOTRE_USERNAME/backtest_core.git
cd backtest_core
```

> ⚠️ Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur GitHub

### 2️⃣ Créer l'environnement virtuel

**Windows (PowerShell/CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ✅ Vérification de l'installation

Testez que tout fonctionne :

```bash
# Test import des modules
python -c "import streamlit, pandas, numpy, plotly; print('✅ Toutes les dépendances sont installées!')"

# Lancer l'interface Streamlit
streamlit run ui/app.py
```

Si l'interface s'ouvre dans votre navigateur sur `http://localhost:8501`, **c'est bon !** 🎉

---

## 📂 Structure du Projet

```
backtest_core/
├── ui/                     # Interface Streamlit
│   ├── app.py             # Point d'entrée principal
│   └── components/        # Composants UI (charts, model selector, etc.)
├── agents/                # Système multi-agents LLM
│   ├── orchestrator.py    # Orchestrateur principal
│   └── *.py              # Agents spécialisés (DataAgent, StrategyAgent, etc.)
├── backtest/              # Moteur de backtest
│   ├── simulator.py       # Simulateur de trading
│   └── metrics_tier_s.py  # Métriques de performance
├── strategies/            # Stratégies de trading
│   ├── bollinger_atr.py   # Mean reversion (V1)
│   ├── bollinger_atr_v2.py # Stop-loss Bollinger (V2)
│   ├── bollinger_atr_v3.py # Entrées/Stop/TP variables (V3)
│   └── *.py              # Autres stratégies
├── indicators/            # Indicateurs techniques
├── data/                  # Données OHLCV
└── requirements.txt       # Dépendances Python
```

---

## 🎮 Utilisation

### Lancer l'interface

```bash
streamlit run ui/app.py
```

L'interface s'ouvre automatiquement dans votre navigateur.

### Workflow typique

1. **Charger des données OHLCV** (CSV ou Parquet)
2. **Sélectionner une stratégie** (Bollinger ATR V3, EMA Cross, etc.)
3. **Configurer les paramètres** via les sliders
4. **Lancer le backtest** (mode simple ou grid search)
5. **Analyser les résultats** (métriques, graphiques, trades)

---

## 🔧 Dépendances Principales

| Package        | Version  | Usage                              |
|----------------|----------|------------------------------------|
| `streamlit`    | ≥1.28    | Interface utilisateur              |
| `pandas`       | ≥2.0     | Manipulation données OHLCV         |
| `numpy`        | ≥1.24    | Calculs vectorisés                 |
| `plotly`       | ≥5.18    | Graphiques interactifs             |
| `numba`        | ≥0.58    | JIT compilation (performance)      |
| `httpx`        | ≥0.27    | Client HTTP (Ollama)               |
| `pyarrow`      | ≥14.0    | Lecture/écriture Parquet           |
| `optuna`       | ≥3.0     | Optimisation bayésienne            |

**Voir [requirements.txt](requirements.txt) pour la liste complète.**

---

## 🐛 Dépannage

### Erreur `ModuleNotFoundError`

```bash
# Vérifier que l'environnement virtuel est activé
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur Streamlit `DuplicateWidgetID`

```bash
# Redémarrer Streamlit avec cache clear
streamlit run ui/app.py --server.runOnSave false
```

### Erreur NumPy/Pandas version

```bash
# Forcer la réinstallation des versions correctes
pip install --force-reinstall -r requirements.txt
```

### Port 8501 déjà utilisé

```bash
# Utiliser un port différent
streamlit run ui/app.py --server.port 8502
```

---

## 🔄 Mise à jour du projet

Pour récupérer les dernières modifications depuis GitHub :

```bash
# Sauvegarder vos modifications locales (optionnel)
git stash

# Récupérer les dernières modifications
git pull origin main

# Mettre à jour les dépendances
pip install --upgrade -r requirements.txt

# Restaurer vos modifications (si stash)
git stash pop
```

---

## 📝 Configuration Optionnelle

### Accélération GPU (optionnel)

Si vous avez une carte NVIDIA, installez les dépendances GPU :

```bash
pip install cupy-cuda12x  # Pour CUDA 12.x
```

### Ollama pour LLM (optionnel)

Pour utiliser les agents LLM :

1. Installer [Ollama](https://ollama.com)
2. Télécharger un modèle :
   ```bash
   ollama pull mistral
   ```
3. L'interface détectera automatiquement Ollama

---

## 💾 Sauvegarder vos modifications sur GitHub

### Première utilisation

```bash
# Configurer Git (première fois)
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"

# Ajouter vos modifications
git add .
git commit -m "Description de vos modifications"
git push origin main
```

### Modifications ultérieures

```bash
# Vérifier les fichiers modifiés
git status

# Ajouter les fichiers modifiés
git add fichier1.py fichier2.py
# OU tout ajouter:
git add .

# Créer un commit avec un message descriptif
git commit -m "Ajout stratégie V4 avec trailing stop"

# Pousser vers GitHub
git push origin main
```

---

## 📞 Support

- **Issues GitHub**: Pour signaler des bugs ou demander des fonctionnalités
- **Documentation**: Voir `docs/` pour plus de détails

---

## ✨ Fonctionnalités Clés

- ✅ **Backtest Ultra-Rapide**: Moteur vectorisé avec Numba JIT
- ✅ **Grid Search Parallèle**: Test de milliers de combinaisons de paramètres
- ✅ **Walk-Forward Analysis**: Validation robuste avec fenêtre glissante
- ✅ **Système Multi-Agents LLM**: Optimisation intelligente des stratégies
- ✅ **9 Stratégies Pré-Configurées**: Bollinger, EMA, MACD, RSI, etc.
- ✅ **30+ Indicateurs Techniques**: ATR, Stochastic, Ichimoku, etc.
- ✅ **Visualisations Interactives**: Equity curves, drawdown, trades

---

**Version**: 2.0.0
**Dernière mise à jour**: 2025-01-XX
**Licence**: MIT
