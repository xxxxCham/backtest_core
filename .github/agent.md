# Backtest Core - Instructions Agent LLM

> **Fichier de référence pour agents LLM** travaillant sur le projet backtest_core.
> Version: 1.0 | Mise à jour: 17/12/2025

---

## 🎯 Contexte Projet

**backtest_core** est un moteur de backtesting professionnel pour stratégies de trading algorithmique.

| Caractéristique | Valeur |
|-----------------|--------|
| Langage | Python 3.10+ |
| Framework UI | Streamlit |
| Graphiques | Plotly |
| Tests | pytest |
| Calculs | NumPy, Pandas, Numba JIT, CuPy (GPU) |
| LLM | Ollama/OpenAI (agents autonomes) |

---

## 🔴 Règles Critiques

### 1. MODIFIER plutôt que CRÉER
> **Si un fichier existant peut être amélioré, TOUJOURS préférer la modification à la création d'un nouveau fichier.**

### 2. Mettre à jour la documentation
Après chaque modification de code :
1. Mettre à jour [copilot-instructions.md](.github/copilot-instructions.md) - section concernée
2. Ajouter une entrée dans l'Index des Modifications avec la date
3. Si modification CLI → mettre à jour [CLI_REFERENCE.md](.github/CLI_REFERENCE.md)

### 3. Calculs vectorisés obligatoires
```python
# ❌ INTERDIT : boucles Python sur séries de prix
for i in range(len(prices)):
    result[i] = prices[i] * factor

# ✅ OBLIGATOIRE : NumPy vectorisé
result = prices * factor
```

### 4. Conventions de signaux
- `1` = Position LONG
- `-1` = Position SHORT  
- `0` = Neutre / Flat

### 5. Frais en Basis Points (BPS)
```python
fees_bps = 10  # = 0.1%
fees_bps = 100 # = 1%
```

---

## 📁 Architecture - Zones d'Intervention

### backtest/ - Moteur Principal
| Fichier | Rôle | Quand modifier |
|---------|------|----------------|
| `engine.py` | Orchestrateur `BacktestEngine.run()` | Nouvelle fonctionnalité de backtest |
| `simulator.py` | Simulation trades | Logique d'exécution |
| `simulator_fast.py` | Version Numba JIT | Optimisation performance |
| `performance.py` | Métriques standards | Nouvelles métriques |
| `metrics_tier_s.py` | Métriques institutionnelles | SQN, Sortino, Calmar |
| `validation.py` | Walk-Forward | Anti-overfitting |
| `optuna_optimizer.py` | Optimisation bayésienne | Amélioration hyperparams |
| `sweep.py` | Grid search | Optimisation paramétrique |
| `execution.py` | Spread/slippage réaliste | Simulation réaliste |
| `facade.py` | Interface UI↔Backend | Contrats d'API |
| `errors.py` | Hiérarchie d'erreurs | Nouveaux types d'erreurs |

### strategies/ - Stratégies de Trading
**Pattern obligatoire** :
```python
from strategies.base import register_strategy, StrategyBase

@register_strategy("nom_strategie")
class MaStrategy(StrategyBase):
    @property
    def required_indicators(self) -> List[str]:
        return ["indicateur1", "indicateur2"]
    
    def generate_signals(self, df, indicators, params) -> pd.Series:
        # Retourne: 1=long, -1=short, 0=flat
        return signals
```

**Stratégies existantes (10)** :
- `bollinger_atr`, `bollinger_dual`, `ema_cross`, `macd_cross`
- `rsi_reversal`, `rsi_trend_filtered`, `atr_channel`
- `ma_crossover`, `ema_stochastic_scalp`

### indicators/ - Indicateurs Techniques
**Pattern obligatoire** :
```python
from indicators.registry import register_indicator

@register_indicator("nom_indicateur")
def calculate_indicator(high, low, close, period=14, **kwargs):
    """
    Calcule l'indicateur XYZ.
    
    Args:
        high: np.array des prix hauts
        low: np.array des prix bas
        close: np.array des prix de clôture
        period: Période de calcul
    
    Returns:
        np.array ou dict si multiple outputs
    """
    # Calcul vectorisé NumPy
    return result
```

**Indicateurs existants (24)** :
- Momentum: `rsi`, `macd`, `stochastic`, `stoch_rsi`, `momentum`, `roc`, `williams_r`, `cci`
- Tendance: `ema`, `sma`, `adx`, `aroon`, `supertrend`, `ichimoku`, `psar`, `vortex`
- Volatilité: `atr`, `bollinger`, `keltner`, `donchian`
- Volume: `obv`, `mfi`, `vwap`

### agents/ - Intelligence LLM
**Architecture 4 agents + orchestrateur** :
```
Analyst → Strategist → Critic → Validator
              ↑_________|________|
                     (itération)
```

| Agent | Rôle |
|-------|------|
| `analyst.py` | Analyse quantitative des performances |
| `strategist.py` | Génération de propositions de paramètres |
| `critic.py` | Évaluation overfitting et risques |
| `validator.py` | Décision finale APPROVE/REJECT/ITERATE |
| `orchestrator.py` | Coordination du workflow |
| `autonomous_strategist.py` | Mode autonome avec backtests réels |
| `integration.py` | Pont vers BacktestEngine |

### ui/ - Interface Streamlit
⚠️ **Règle stricte** : AUCUNE logique de trading dans ce dossier.
- `app.py` : Point d'entrée unique
- Utilise `facade.py` pour communiquer avec le backend

### utils/ - Utilitaires
| Fichier | Rôle |
|---------|------|
| `config.py` | Configuration singleton |
| `parameters.py` | ParameterSpec, Presets, Contraintes |
| `observability.py` | Logging intelligent, spans, compteurs |
| `visualization.py` | Graphiques Plotly |
| `circuit_breaker.py` | Protection échecs répétés |
| `checkpoint.py` | Sauvegarde/reprise état |
| `error_recovery.py` | Retry avec backoff |
| `gpu_oom.py` | Gestion OOM GPU |

### cli/ - Mode Ligne de Commande
Point d'entrée : `python __main__.py [COMMANDE]`

| Commande | Description |
|----------|-------------|
| `backtest` | Exécuter un backtest simple |
| `sweep` | Optimisation paramétrique (grid) |
| `optuna` | Optimisation bayésienne |
| `visualize` | Graphiques interactifs |
| `list` | Lister ressources |
| `info` | Détails d'une ressource |
| `validate` | Validation système |
| `export` | Export résultats |

---

## 🧪 Tests

### Exécution
```powershell
# Tous les tests
python run_tests.py

# Avec coverage
python run_tests.py --coverage

# Tests spécifiques
pytest tests/test_engine.py -v
pytest tests/ -k "sharpe" -v
```

### Pattern de test
```python
import pytest
from backtest.engine import BacktestEngine

class TestEngine:
    def test_run_returns_result(self, sample_data):
        engine = BacktestEngine()
        result = engine.run(sample_data, strategy, params)
        assert result.metrics['total_pnl'] is not None
```

### Fixtures disponibles
- `sample_data` : DataFrame OHLCV de test
- `engine` : Instance BacktestEngine
- `config` : Configuration de test

---

## ⚡ Performance

### Optimisations actives
1. **Numba JIT** : `simulator_fast.py`, `execution_fast.py`
2. **GPU CuPy** : Backend transparent NumPy/CuPy
3. **IndicatorBank** : Cache disque avec TTL
4. **Vectorisation** : Tout calcul doit être vectorisé

### Benchmarks cibles
| Opération | Temps cible |
|-----------|-------------|
| Backtest 10k bars | < 100ms |
| Sweep 1000 combos | < 2 min |
| Calcul indicateur | < 10ms |

---

## 📝 Conventions de Code

### Imports
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd

# Local
from backtest.engine import BacktestEngine
from indicators.registry import get_indicator
```

### Docstrings (français)
```python
def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Calcule le Sharpe Ratio.
    
    Args:
        returns: Série des rendements
        risk_free: Taux sans risque annualisé
    
    Returns:
        Sharpe Ratio annualisé
    
    Raises:
        ValueError: Si returns est vide
    """
```

### Gestion d'erreurs
```python
from backtest.errors import BacktestError, DataError, StrategyNotFoundError

# Utiliser les erreurs typées
raise DataError("Fichier introuvable", file_path=path)
raise StrategyNotFoundError(f"Stratégie '{name}' non enregistrée")
```

---

## 🔄 Workflow de Modification

### Ajout d'un indicateur
1. Créer `indicators/nom_indicateur.py` avec `@register_indicator`
2. Ajouter tests dans `tests/test_indicators.py`
3. Mettre à jour `copilot-instructions.md` → section indicators/
4. Ajouter dans l'Index des Modifications

### Ajout d'une stratégie
1. Créer `strategies/nom_strategie.py` avec `@register_strategy`
2. Définir `required_indicators` et `generate_signals`
3. Ajouter tests dans `tests/test_strategy.py`
4. Mettre à jour `copilot-instructions.md` → section strategies/
5. Ajouter dans l'Index des Modifications

### Modification du CLI
1. Modifier `cli/commands.py`
2. Mettre à jour `CLI_REFERENCE.md`
3. Ajouter tests dans `tests/test_cli.py`
4. Mettre à jour `copilot-instructions.md` → Index des Modifications

### Ajout d'une métrique
1. Modifier `backtest/performance.py` ou `metrics_tier_s.py`
2. Ajouter tests dans `tests/test_performance.py`
3. Mettre à jour `copilot-instructions.md` → section backtest/

---

## 🚫 À Éviter

| ❌ Ne pas faire | ✅ Faire plutôt |
|-----------------|-----------------|
| Boucles for sur prix | NumPy vectorisé |
| Logique métier dans ui/ | Utiliser facade.py |
| Print statements | Logger avec observability.py |
| Tests sans assertions | Assertions explicites |
| Fichiers dupliqués | Modifier l'existant |
| Magic numbers | Constantes nommées |
| Imports circulaires | Imports dans fonctions si nécessaire |

---

## 📊 Variables d'Environnement Clés

```bash
# Données
BACKTEST_DATA_DIR=D:/Trading/Data

# LLM
BACKTEST_LLM_PROVIDER=ollama
BACKTEST_LLM_MODEL=deepseek-r1:32b
OLLAMA_HOST=http://localhost:11434

# Performance
USE_GPU=true
MAX_WORKERS=8

# Debug
BACKTEST_LOG_LEVEL=DEBUG
```

---

## 🎯 Priorités de Développement

Consulter [ROADMAP.md](.github/ROADMAP.md) pour la roadmap complète.

**Phases complétées** :
- ✅ Phase 1 : Fondations (Walk-Forward, Métriques Tier S)
- ✅ Phase 2 : Performance (IndicatorBank, Numba JIT, GPU)
- ✅ Phase 3 : Intelligence LLM (4 Agents + Autonome)
- ✅ Phase 4 : Robustesse (Circuit Breaker, Recovery)
- ✅ Phase 5 : UI/UX Avancée

---

## 📚 Fichiers de Référence

| Fichier | Contenu |
|---------|---------|
| [copilot-instructions.md](.github/copilot-instructions.md) | Instructions détaillées + Index des Modifications |
| [CLI_REFERENCE.md](.github/CLI_REFERENCE.md) | Documentation CLI complète |
| [ROADMAP.md](.github/ROADMAP.md) | Roadmap stratégique |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Variables d'environnement |
| [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) | Benchmarks détaillés |

---

*Ce fichier doit être consulté en priorité avant toute intervention sur le projet.*
