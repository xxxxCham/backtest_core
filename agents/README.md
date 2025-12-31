# Agents Module - Intelligence LLM

> **Système d'optimisation autonome par agents LLM**  
> Version : 1.8.1 | Phase 3 | Date : 13/12/2025

---

## 🤖 Vue d'Ensemble

Le module `agents/` implémente un système d'optimisation autonome basé sur des agents LLM (Large Language Models). Il permet d'optimiser automatiquement les paramètres de stratégies de trading en utilisant l'intelligence artificielle.

**Deux modes de fonctionnement :**
1. **Mode Autonome (RECOMMANDE)** : L'agent lance des backtests et itere
2. **Mode Orchestre** : Orchestrator multi-agents; backtests uniquement si un callback `on_backtest_needed` est fourni

---

## 📁 Structure

```
agents/
├── base_agent.py              → Classe abstraite pour tous les agents
├── analyst.py                 → Agent Analyst (analyse quantitative)
├── strategist.py              → Agent Strategist (propositions créatives)
├── critic.py                  → Agent Critic (détection overfitting)
├── validator.py               → Agent Validator (décision finale)
├── orchestrator.py            → Orchestrateur workflow multi-agents
├── orchestration_logger.py    → Logging structure (JSONL) pour l'orchestration
├── autonomous_strategist.py   → Agent autonome avec backtests réels
├── backtest_executor.py       → Interface d'exécution backtests
├── integration.py             → Pont vers BacktestEngine
├── state_machine.py           → Machine à états du workflow
├── llm_client.py              → Client LLM unifié (Ollama/OpenAI)
├── model_config.py            → Configuration multi-modèles par rôle
├── ollama_manager.py          → Gestion GPU/VRAM pour LLM
└── __init__.py                → Exports publics
```

---

## 🎯 Architecture - Mode Autonome

### Workflow Itératif

```
┌─────────────┐
│   BASELINE  │ ← Backtest initial avec paramètres par défaut
└──────┬──────┘
       ↓
┌─────────────┐
│   ANALYZE   │ ← Analyst : Analyse quantitative des résultats
└──────┬──────┘
       ↓
┌─────────────┐
│   PROPOSE   │ ← Strategist : Propose nouveaux paramètres
└──────┬──────┘
       ↓
┌─────────────┐
│  BACKTEST   │ ← Exécution backtest avec nouveaux params
└──────┬──────┘
       ↓
┌─────────────┐
│  EVALUATE   │ ← Critic : Évalue overfitting et risques
└──────┬──────┘
       ↓
   ┌────────────┐
   │ ACCEPT ?   │
   └─────┬──────┘
         ├─── OUI → STOP (meilleurs params trouvés)
         └─── NON → Retour à ANALYZE (itération suivante)
```

### Composants Clés

1. **AutonomousStrategist** : Agent principal
   - Lance des backtests réels via `BacktestExecutor`
   - Boucle d'itération jusqu'à convergence
   - Tracking de l'historique des expériences

2. **BacktestExecutor** : Interface d'exécution
   - Abstraction pour lancer des backtests
   - Support walk-forward validation
   - Gestion des erreurs et timeout

3. **Integration Layer** : Pont vers BacktestEngine
   - `run_backtest_for_agent()` : Lance un backtest
   - `create_optimizer_from_engine()` : Crée optimiseur complet
   - `quick_optimize()` : Raccourci rapide

---

## 🚀 Guide d'Utilisation

### 1. Mode Autonome (Recommandé)

**Exemple complet avec intégration BacktestEngine :**

```python
from agents import create_optimizer_from_engine, quick_optimize
from agents.llm_client import LLMConfig, LLMProvider

# Méthode 1: Contrôle complet
config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
strategist, executor = create_optimizer_from_engine(
    llm_config=config,
    strategy_name="ema_cross",  # Stratégie du registre
    data=ohlcv_df,              # DataFrame OHLCV
    use_walk_forward=True,      # Activer validation anti-overfitting
)

session = strategist.optimize(
    executor=executor,
    initial_params={"fast_period": 10, "slow_period": 21},
    param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
    max_iterations=10,
)

# Résultats
print(f"Meilleur Sharpe: {session.best_result.sharpe_ratio:.2f}")
print(f"Params: {session.best_result.request.parameters}")
print(f"Final status: {session.final_status} after {session.current_iteration} iterations")

# Méthode 2: Raccourci rapide
session = quick_optimize(
    config=config,
    strategy_name="ema_cross",
    data=df,
    max_iterations=10,
)
```

**Paramètres Clés :**
- `initial_params` : Point de départ (défaut: stratégie par défaut)
- `param_bounds` : Bornes d'exploration (ex: `{"fast": (5, 50)}`)
- `max_iterations` : Limite itérations (défaut: 10)
- `target_metric` : Métrique à maximiser (défaut: `sharpe_ratio`)
- `use_walk_forward` : Validation robuste (défaut: `True`)

---

### 2. Mode Orchestre (Analyse Multi-Agents)

**Analyse multi-agents (backtests si callback fourni) :**

```python
from agents import create_orchestrator_with_backtest
from agents.llm_client import LLMConfig, LLMProvider

config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
orchestrator = create_orchestrator_with_backtest(
    strategy_name="ema_cross",
    data=ohlcv_df,
    initial_params={"fast_period": 12, "slow_period": 26},
    llm_config=config,
)

result = orchestrator.run()

# Resultat
print(result.decision)      # APPROVE / REJECT / ABORT
print(result.final_params)  # meilleurs parametres retenus
print(result.final_report)  # rapport complet
```

---

### 3. Configuration LLM

**Variables d'environnement :**

```bash
# Ollama (local, gratuit)
export BACKTEST_LLM_PROVIDER=ollama
export BACKTEST_LLM_MODEL=llama3.2
export OLLAMA_HOST=http://localhost:11434

# OpenAI (cloud, payant)
export BACKTEST_LLM_PROVIDER=openai
export BACKTEST_LLM_MODEL=gpt-4
export OPENAI_API_KEY=sk-...
```

**Configuration Python :**

```python
from agents.llm_client import LLMConfig, LLMProvider

# Ollama
config_ollama = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="deepseek-r1:8b",
    host="http://localhost:11434",
    temperature=0.7,
)

# OpenAI
config_openai = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0.5,
)
```

---

### 4. Configuration Multi-Modèles par Rôle

**Attribuer des modèles différents par agent :**

```python
from agents.model_config import RoleModelConfig, ModelCategory, set_global_model_config

# Configuration personnalisée
config = RoleModelConfig(
    analyst="deepseek-r1:32b",      # Modèle fort pour analyse
    strategist="llama3.2",          # Modèle créatif
    critic="mistral",               # Modèle sceptique
    validator="qwen2.5:32b",        # Modèle décisionnel
)

set_global_model_config(config)

# Note: si plusieurs modeles sont listes pour un role, la selection est aleatoire par defaut.
```

**Modèles connus et catégories :**

| Modèle | Catégorie | Taille | Recommandation |
|--------|-----------|--------|----------------|
| `deepseek-r1:32b` | Premier | 32B | **Analyse**, Validation |
| `qwen2.5:32b` | Premier | 32B | Validation, Critique |
| `llama3.2` | Standard | 8B | Strategist, usage général |
| `mistral` | Standard | 7B | Critic, analyse risques |
| `phi3` | Rapide | 3B | Tests, prototypage |

---

### 5. Gestion GPU/VRAM

**Décharger le LLM pendant les backtests (libérer VRAM) :**

```python
from agents import create_autonomous_optimizer, gpu_compute_context

# Méthode 1: Variable d'env (global)
# export UNLOAD_LLM_DURING_BACKTEST=True

# Méthode 2: Paramètre Python
strategist = AutonomousStrategist(
    llm_client=client,
    unload_llm_during_backtest=True,  # Décharge LLM avant backtest
)

# Méthode 3: Context manager pour calculs manuels
with gpu_compute_context("deepseek-r1:32b"):
    # GPU libre pour calculs NumPy/CuPy
    result = heavy_gpu_computation()
# LLM rechargé automatiquement
```

**Impact :**

| Mode | VRAM LLM | VRAM Backtest | Latence |
|------|----------|---------------|---------|
| `unload=False` | Partagé | Partagé | 0s |
| `unload=True` | 0 GB | 100% libre | +2-5s |

---

## 🧠 Les 4 Agents Spécialisés

### 1. Analyst Agent 📊

**Rôle :** Analyse quantitative des performances

**Analyse :**
- Métriques Tier S (Sharpe, Sortino, Calmar)
- Drawdown et volatilité
- Distribution des trades
- Corrélations temporelles

**Output :**
```json
{
  "summary": "Performance correcte mais volatilité élevée...",
  "strengths": ["Sharpe > 1.5", "Win rate 58%"],
  "weaknesses": ["Max drawdown 12%", "Peu de trades"],
  "key_observations": ["Sensibilité à fast_period..."]
}
```

---

### 2. Strategist Agent 🎯

**Rôle :** Propositions créatives d'optimisation

**Propose :**
- Ajustements de paramètres ciblés
- Justifications quantitatives
- Risques anticipés

**Output :**
```json
{
  "proposals": [
    {
      "name": "Réduire fast_period",
      "params": {"fast_period": 8, "slow_period": 21},
      "rationale": "Augmente réactivité aux retournements",
      "expected_impact": "Sharpe +0.2, trades +15%"
    }
  ]
}
```

---

### 3. Critic Agent 🛡️

**Rôle :** Détection overfitting et risques

**Évalue :**
- Ratio train/test
- Stabilité paramètres
- Sur-optimisation
- Robustesse

**Output :**
```json
{
  "concerns": [
    {
      "type": "overfitting",
      "severity": "medium",
      "evidence": "Train Sharpe 2.1 vs Test Sharpe 0.9"
    }
  ],
  "evaluations": [
    {
      "proposal": "Réduire fast_period",
      "overfitting_score": 25,  # 0-100 (plus bas = mieux)
      "robustness_score": 85
    }
  ]
}
```

---

### 4. Validator Agent ✅

**Rôle :** Décision finale APPROVE/REJECT/ITERATE

**Critères :**
- ✅ Sharpe > objectif
- ✅ Drawdown < limite
- ✅ Pas d'overfitting sévère
- ✅ Amélioration vs baseline

**Output :**
```json
{
  "decision": "APPROVE",
  "recommendation": "Adopter Proposal 2",
  "rationale": "Sharpe 1.8, drawdown 5%, robuste sur walk-forward"
}
```

---

## 📊 Suivi de Session

### OptimizationSession

Objet retourné contenant :

```python
session = strategist.optimize(...)

# Résultats finaux
session.best_result           # BacktestResult (métriques, trades)
session.all_results           # Liste des BacktestResult (baseline incluse)
session.decisions             # Liste des IterationDecision du LLM

# Historique
session.current_iteration     # Iteration actuelle
session.start_time            # Date de debut de session

# Statut final
session.final_status          # "success" | "max_iterations" | "timeout" | "no_improvement"
session.final_reasoning       # Raison finale

# Métriques
session.best_result.sharpe_ratio
session.best_result.max_drawdown
session.best_result.total_trades
```

---

## 🎓 Cas d'Usage Avancés

### Cas 1 : Optimisation Multi-Stratégie

```python
strategies = ["ema_cross", "bollinger_atr", "macd_cross"]

for strategy_name in strategies:
    session = quick_optimize(config, strategy_name, df)
    print(f"{strategy_name}: Sharpe {session.best_result.sharpe_ratio:.2f}")
```

---

### Cas 2 : Walk-Forward avec Validation Stricte

```python
strategist, executor = create_optimizer_from_engine(
    llm_config=config,
    strategy_name="ema_cross",
    data=df,
    use_walk_forward=True,      # Active WF (n_windows=6, train_ratio=0.75)
)

session = strategist.optimize(executor, max_iterations=15)
```

---

### Cas 3 : Analyse de Sensibilité

```python
history = executor.history

# Sensibilite d'un parametre
sensitivity = history.analyze_parameter_sensitivity()
fast_stats = sensitivity.get("fast_period")
if fast_stats:
    print(f"fast_period corr: {fast_stats['correlation']:.2f}")

# Meilleurs runs
top_5 = sorted(
    [exp for exp in history.experiments if exp.success],
    key=lambda exp: exp.sharpe_ratio,
    reverse=True,
)[:5]
for exp in top_5:
    print(f"Sharpe {exp.sharpe_ratio:.2f} | Params: {exp.request.parameters}")
```

---

## 🔧 Configuration Avancée

### Paramètres AutonomousStrategist

```python
strategist = AutonomousStrategist(
    llm_client=client,
    unload_llm_during_backtest=True,  # Décharger LLM
    verbose=True,                     # Logs détaillés
    on_progress=lambda i, p: print(f"Iteration {i}: {p}"),
)
```

### Paramètres BacktestExecutor

```python
executor = BacktestExecutor(
    backtest_fn=my_backtest_function,
    strategy_name="ema_cross",
    data=df,
    validation_fn=my_walk_forward_fn,  # optionnel: doit retourner train_sharpe/test_sharpe/overfitting_ratio
)
```

---

## 🐛 Troubleshooting

### Erreur : "Ollama not available"
```bash
# Vérifier Ollama
curl http://localhost:11434/api/version

# Démarrer Ollama
ollama serve
```

### Erreur : "GPU Out of Memory"
```python
# Activer déchargement LLM
export UNLOAD_LLM_DURING_BACKTEST=True
```

### Convergence lente
```python
# Réduire complexité
session = strategist.optimize(
    executor,
    max_iterations=5,              # Moins d'itérations
    param_bounds={"fast": (8, 12)},  # Espace restreint
)
```

### LLM répond mal
```python
# Changer température
config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="llama3.2",
    temperature=0.3,  # Plus déterministe (défaut: 0.7)
)
```

---

## 📚 Références

- [Phase 3 LLM Integration](../.github/ROADMAP.md#phase-3---intelligence-llm)
- [Ollama Documentation](https://ollama.ai/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [GPU Memory Management](../docs/UNLOAD_LLM_FEATURE.md)

---

*Dernière mise à jour : 13/12/2025 | Version : 1.8.1 | Phase 3*
