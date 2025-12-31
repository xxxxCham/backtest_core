# 🚀 Backtest Core

Systeme de backtesting algorithmique avance avec optimisation multi-agents LLM.

Plateforme complète pour développer, tester et optimiser des stratégies de trading quantitatives avec une interface Streamlit moderne et un système d'agents LLM intelligents.

---

## ✨ Fonctionnalités

- ⚡ **Moteur de Backtest Ultra-Rapide**: Vectorisation NumPy + compilation JIT Numba
- 🎯 **9 Stratégies Pré-Configurées**: Bollinger ATR (V1/V2/V3), EMA Cross, MACD, RSI, etc.
- 📊 **30+ Indicateurs Techniques**: ATR, Stochastic, Ichimoku, Fibonacci, etc.
- 🔍 **Grid Search Parallélisé**: Teste des milliers de combinaisons de paramètres
- 🧠 **Système Multi-Agents LLM**: Optimisation intelligente via Ollama (Mistral, Llama)
- 📈 **Walk-Forward Analysis**: Validation robuste avec fenêtres glissantes
- 🎨 **Interface Streamlit Interactive**: Visualisations Plotly, equity curves, drawdown
- 📦 **Support Multi-Formats**: CSV, Parquet, données crypto/actions

---

## 🎯 Installation Rapide

### Option 1: Script Automatique (Recommandé)

#### Windows

```bash
git clone https://github.com/VOTRE_USERNAME/backtest_core.git
cd backtest_core
install.bat
```text

#### Linux/macOS

```bash
git clone https://github.com/VOTRE_USERNAME/backtest_core.git
cd backtest_core
chmod +x install.sh
./install.sh
```

### Option 2: Installation Manuelle

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/backtest_core.git
cd backtest_core

# Créer environnement virtuel
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# OU Activer (Linux/macOS)
source .venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

📖 **Note**: L'installation manuelle ci-dessus suffit pour démarrer.

---

## 🚀 Démarrage Rapide

```bash
# Activer l'environnement virtuel
source .venv/bin/activate  # Linux/macOS
# OU
.venv\Scripts\activate     # Windows

# Lancer l'interface Streamlit
streamlit run ui/app.py
```

L'interface s'ouvre automatiquement sur `http://localhost:8501`

---

## 🤖 Instructions & historique (agents)

Pour les agents (Copilot / Claude Code / Codex / autres), la **source unique de vérité** (règles + journal daté des changements) est :

- [AGENTS.md](AGENTS.md)

## 📖 Documentation

- **[INSTALL.md](INSTALL.md)**: Guide d'installation détaillé
- **[DETAILS_FONCTIONNEMENT.md](DETAILS_FONCTIONNEMENT.md)**: Architecture et points d'entrée
- **`docs/`**: Documentation complète des stratégies et indicateurs

---

## 🎮 Utilisation

### Interface Streamlit (Mode Interactif)

1. Charger des données OHLCV (CSV/Parquet)
2. Sélectionner une stratégie (ex: Bollinger ATR V3)
3. Configurer les paramètres via sliders
4. Lancer le backtest (simple ou grid search)
5. Analyser les résultats (métriques, graphiques, trades)

### API Python (Mode Programmatique)

```python
from backtest.engine import BacktestEngine
from utils.parameters import save_versioned_preset, load_strategy_version

# Charger preset optimisé
preset = load_strategy_version("bollinger_atr_v3", version="0.0.1")
params = preset.get_default_values()

# Lancer backtest
engine = BacktestEngine()
result = engine.run(df=data, strategy="bollinger_atr_v3", params=params)

# Sauvegarder meilleurs paramètres
save_versioned_preset(
    strategy_name="bollinger_atr_v3",
    version="0.0.1",
    preset_name="winner",
    params_values={"bb_std": 2.5, "stop_factor": 0.4, ...}
)
```

---

## 📂 Structure du Projet

```
backtest_core/
├── ui/                     # Interface Streamlit
│   ├── app.py             # 🎯 Point d'entrée principal
│   └── components/        # Composants (charts, model selector, etc.)
├── agents/                # 🧠 Système multi-agents LLM
│   ├── orchestrator.py    # Orchestrateur principal
│   ├── data_agent.py      # Agent de données
│   ├── strategy_agent.py  # Agent de stratégies
│   └── optimization_agent.py  # Agent d'optimisation
├── backtest/              # ⚡ Moteur de backtest
│   ├── simulator.py       # Simulateur de trading vectorisé
│   ├── metrics_tier_s.py  # Métriques Tier-S (Sharpe, Sortino, etc.)
│   └── execution_fast.py  # Exécution optimisée Numba
├── strategies/            # 📈 Stratégies de trading
│   ├── bollinger_atr.py       # Mean reversion V1
│   ├── bollinger_atr_v2.py    # Stop-loss Bollinger V2
│   ├── bollinger_atr_v3.py    # Entrées/Stop/TP variables V3
│   ├── ema_cross.py           # EMA Crossover
│   └── ...                    # Autres stratégies
├── indicators/            # 📊 Indicateurs techniques (30+)
├── data/                  # 💾 Données OHLCV
├── requirements.txt       # 📦 Dépendances Python
├── install.bat            # 🛠️ Installation automatique (Windows)
└── install.sh             # 🛠️ Installation automatique (Linux/macOS)
```

---

## 🔧 Technologies

| Composant          | Technologie                    |
|--------------------|--------------------------------|
| **Interface**      | Streamlit 1.28+                |
| **Calculs**        | NumPy 1.24+, Pandas 2.0+       |
| **Performance**    | Numba JIT, Joblib (parallèle)  |
| **Visualisation**  | Plotly 5.18+, Matplotlib       |
| **LLM**            | Ollama (Mistral, Llama)        |
| **Optimisation**   | Optuna 3.0+ (Bayesian)         |
| **Données**        | PyArrow (Parquet)              |

---

## 🧠 Modèles LLM Avancés (Optionnel)

Le système multi-agents supporte plusieurs modèles LLM via Ollama. Pour les tâches critiques nécessitant un raisonnement profond:

### Llama-3.3-70B-Instruct (Multi-GPU)

Configuration optimisée pour distributions multi-GPU avec offloading RAM:

```bash
# Installation et configuration automatique
python tools/setup_llama33_70b.py

# Vérification de l'intégration
python tools/test_llama33_70b.py
```

#### Prerequis

- 2 GPUs NVIDIA (recommandé: 20GB+ VRAM chacun)
- 32GB+ RAM DDR5 pour offloading
- ~40GB espace disque

#### Caracteristiques

- Distribution automatique sur 2 GPUs
- Quantization Q4 (~40GB VRAM total)
- Utilisé pour rôles Critic (iter>=2) et Validator (iter>=3)
- Temps de réponse: ~5 min pour analyses complexes

---

## 🎯 Stratégies Disponibles

### Mean Reversion
- **Bollinger ATR** (V1): Stop-loss ATR classique
- **Bollinger ATR V2**: Stop-loss Bollinger paramétrable
- **Bollinger ATR V3**: Entrées/Stop/TP variables sur échelle unifiée
- **RSI Reversal**: Retournement sur zones extrêmes
- **Bollinger Dual**: Double condition Bollinger + MA

### Trend Following
- **EMA Cross**: Croisement EMA rapide/lente
- **MA Crossover**: Croisement SMA
- **ATR Channel**: Breakout sur canal ATR

### Momentum
- **MACD Cross**: Croisement MACD/Signal
- **EMA Stochastic Scalp**: Scalping EMA + Stochastic

---

## 📊 Exemples de Résultats

**Grid Search Bollinger ATR V3** (5 tokens, 1h, ~778k combinaisons):
- Sharpe Ratio: 1.85
- Win Rate: 58%
- Max Drawdown: -12%
- Profit Factor: 1.42

---

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-strategie`)
3. Commit vos changements (`git commit -m 'Ajout stratégie XYZ'`)
4. Push la branche (`git push origin feature/nouvelle-strategie`)
5. Ouvrir une Pull Request

---

## 📝 Presets Versionnés

Les presets permettent de sauvegarder et charger des configurations optimisées :

**Naming convention**: `<strategy>@<version>__<preset_slug>`

**Localisation**: `BACKTEST_PRESETS_DIR` ou `data/presets/`

```python
# Sauvegarder un preset après optimisation
save_versioned_preset(
    strategy_name="bollinger_atr_v3",
    version="0.0.1",
    preset_name="winner",
    params_values=best_params
)

# Charger un preset
preset = load_strategy_version("bollinger_atr_v3", version="0.0.1")
params = preset.get_default_values()
```

---

## 🐛 Dépannage

Consultez [INSTALL.md#dépannage](INSTALL.md#-dépannage) pour les problèmes courants.

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Streamlit** pour l'interface moderne
- **Numba** pour l'accélération JIT
- **Ollama** pour les capacités LLM locales
- **Optuna** pour l'optimisation bayésienne

---

**Version**: 2.0.0
**Dernière mise à jour**: 2025-01-XX
**Auteur**: Votre Nom

---

🚀 **Happy Backtesting!**
