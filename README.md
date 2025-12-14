# Backtest Core - Moteur de Backtesting Simplifié

## Description

Moteur de backtesting épuré et modulaire, reconstruit à partir du projet ThreadX.
Focus sur la logique fondamentale de backtesting sans les dépendances complexes (LLM, multi-GPU, etc.).

## Architecture

```
backtest_core/
├── backtest/           # Cœur du moteur de backtest
│   ├── engine.py       # Moteur principal BacktestEngine
│   ├── simulator.py    # Simulation des trades
│   └── performance.py  # Calcul des métriques
├── data/               # Chargement des données
│   ├── loader.py       # Fonctions load_ohlcv
│   └── sample_data/    # Données de test
├── indicators/         # Indicateurs techniques
│   ├── bollinger.py    # Bandes de Bollinger
│   ├── atr.py          # Average True Range
│   ├── rsi.py          # Relative Strength Index
│   ├── ema.py          # Exponential Moving Average
│   └── registry.py     # Registre des indicateurs
├── strategies/         # Stratégies de trading
│   ├── base.py         # Classe de base Strategy
│   ├── bollinger_atr.py# Stratégie Bollinger + ATR
│   └── ema_cross.py    # Stratégie EMA Crossover
├── ui/                 # Interface Streamlit
│   └── app.py          # Application minimale
├── utils/              # Utilitaires
│   ├── log.py          # Logging simplifié
│   └── config.py       # Configuration
├── tests/              # Tests unitaires
│   ├── test_engine.py
│   ├── test_indicators.py
│   └── test_strategy.py
└── demo/               # Scripts de démonstration
    └── quick_test.py   # Test rapide du moteur
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation Rapide

```python
from backtest.engine import BacktestEngine
from data.loader import load_ohlcv
from strategies.bollinger_atr import BollingerATRStrategy

# Charger les données
data = load_ohlcv("BTCUSDT", "1m", start="2024-01-01", end="2024-02-01")

# Configurer la stratégie
strategy = BollingerATRStrategy()
params = {
    "entry_z": 2.0,
    "k_sl": 1.5,
    "leverage": 3
}

# Exécuter le backtest
engine = BacktestEngine()
result = engine.run(data, strategy, params)

# Afficher les résultats
print(f"Profit total: ${result.metrics['total_pnl']:.2f}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

## Lancer l'Interface

```bash
streamlit run ui/app.py
```

## ⚡ Performances (Nouveau en v1.8.0)

**Optimisations mesurées** :
- 🚀 Simulateur Numba JIT: **42x speedup** (16ms → 0.38ms)
- 🎮 GPU CuPy: **22x speedup** (7.8ms → 0.35ms)
- 📊 Calculs vectorisés: **100x speedup** (pandas rolling)

**Impact global** :
- ⏱️ Backtest 10k bars: **100x plus rapide**
- 🔄 Sweep 1000 combos: 3.3h → **2 minutes**

Voir [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) pour détails complets.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) | **🆕 Performances** - Benchmarks détaillés v1.8.0 |
| [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) | **🆕 Guide optimisations** - Vectorisation, Numba, GPU |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | **Mode CLI** - Commandes backtest, sweep, optuna, visualize |
| [ENVIRONMENT.md](ENVIRONMENT.md) | **Variables d'env** - Configuration GPU, LLM, logging |
| [LLM_INTEGRATION_README.md](LLM_INTEGRATION_README.md) | **Agents LLM** - Système d'optimisation autonome |
| [ROADMAP.md](ROADMAP.md) | **Roadmap** - Phases de développement et état |
| [copilot-instructions.md](.github/copilot-instructions.md) | **Architecture** - Référence pour agents IA |

### **🔴 Configuration Critique**

**⚠️ Pour CPU-only systems** (la plupart des utilisateurs) :
```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=False  # Défaut, évite latence
```

**✅ Pour GPU avec CuPy** (optimisation avancée) :
```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=True   # Libère 100% VRAM pour calculs
```

Voir [ENVIRONMENT.md](ENVIRONMENT.md) pour détails complets.

## Principes de Design

1. **Simplicité** - Code minimaliste et lisible
2. **Modularité** - Composants indépendants et interchangeables  
3. **Extensibilité** - Architecture préparée pour réintégrer LLM/optimisation
4. **Performance** - Calculs vectorisés NumPy + GPU accéléré (CuPy optionnel)
5. **Testabilité** - Tests unitaires pour chaque composant

## État du Projet

✅ **Production-ready** (582 tests passants)

- ✅ Phase 1: Walk-Forward, Métriques Tier S, Realistic Execution
- ✅ Phase 2: IndicatorBank, 23 indicateurs, Monte Carlo, Pareto
- ✅ Phase 3: 4 Agents LLM + Mode Autonome
- ✅ Phase 4: Circuit Breaker, Error Recovery, GPU OOM Handler
- ✅ Phase 5: UI/UX Monitoring, Timeline Agents, Themes

Voir [ROADMAP.md](ROADMAP.md) pour progression détaillée.

## Licence

MIT
