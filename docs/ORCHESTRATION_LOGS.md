# Système de Logs d'Orchestration LLM - Documentation

> **Date de création** : 18 décembre 2025  
> **Auteur** : GitHub Copilot  
> **Version** : 1.0.0

---

## 📋 Vue d'Ensemble

Le système de logs d'orchestration LLM permet de tracer et visualiser toutes les actions effectuées par les agents LLM lors de l'optimisation autonome de stratégies de trading. Ce système assure la transparence et la traçabilité complète du workflow d'optimisation.

---

## 🎯 Objectifs

1. **Traçabilité** : Enregistrer chaque action des agents LLM
2. **Transparence** : Afficher en temps réel les décisions prises
3. **Debug** : Faciliter le diagnostic en cas de problème
4. **Analyse** : Permettre l'analyse post-mortem des sessions d'optimisation
5. **Audit** : Conserver un historique complet des expérimentations

---

## 🏗️ Architecture

### Composants Clés

```
agents/
├── orchestration_logger.py      # Logger centralisé
│   ├── OrchestrationLogger       # Classe principale
│   ├── OrchestrationLogEntry     # Structure d'un log
│   ├── OrchestrationActionType   # Types d'actions (20+)
│   └── OrchestrationStatus       # Statuts (IN_PROGRESS, COMPLETED, etc.)
│
├── autonomous_strategist.py      # Agent autonome (intégré)
└── integration.py                # Factory create_optimizer_from_engine

ui/
├── orchestration_viewer.py       # Composants de visualisation
│   ├── render_orchestration_logs()         # Timeline des logs
│   ├── render_orchestration_summary_table() # Tableau activités
│   ├── render_orchestration_metrics()       # Métriques clés
│   └── render_full_orchestration_viewer()   # Vue complète
│
└── app.py                        # Interface principale (intégré)
```

---

## 📝 Types d'Actions Enregistrées

Le système enregistre **20+ types d'actions** différentes :

### Analyse
- `ANALYSIS_START` : Début d'analyse
- `ANALYSIS_COMPLETE` : Analyse terminée

### Stratégie
- `STRATEGY_SELECTION` : Sélection d'une stratégie
- `STRATEGY_MODIFICATION` : Modification de stratégie

### Indicateurs
- `INDICATOR_VALUES_CHANGE` : Changement de valeurs d'indicateur
- `INDICATOR_ADD` : Ajout d'un nouvel indicateur
- `INDICATOR_VALIDATION` : Validation d'un indicateur

### Backtests
- `BACKTEST_LAUNCH` : Lancement d'un backtest
- `BACKTEST_COMPLETE` : Backtest terminé avec succès
- `BACKTEST_FAILED` : Backtest échoué

### Décisions
- `DECISION_CONTINUE` : Décision de continuer l'optimisation
- `DECISION_STOP` : Décision d'arrêter
- `DECISION_CHANGE_APPROACH` : Changement d'approche

### Et bien d'autres...

---

## 🔧 Utilisation

### 1. Création du Logger

```python
from agents.orchestration_logger import OrchestrationLogger, generate_session_id

# Générer un ID unique pour la session
session_id = generate_session_id()

# Créer le logger
logger = OrchestrationLogger(session_id=session_id)
```

### 2. Enregistrement des Actions

```python
# Début d'analyse
logger.log_analysis_start(
    agent="AutonomousStrategist",
    details={"strategy": "ema_cross", "initial_params": {...}}
)

# Lancement d'un backtest
logger.log_backtest_launch(
    agent="AutonomousStrategist",
    params={"fast_period": 10, "slow_period": 21},
    combination_id=1,
    total_combinations=10
)

# Résultat du backtest
logger.log_backtest_complete(
    agent="AutonomousStrategist",
    params={"fast_period": 10, "slow_period": 21},
    results={"pnl": 150.0, "sharpe": 1.5},
    combination_id=1
)

# Décision
logger.log_decision(
    agent="AutonomousStrategist",
    decision_type="continue",  # ou "stop", "change_approach"
    reason="Résultats prometteurs",
    details={"next_params": {...}}
)

# Changement d'itération
logger.next_iteration()
```

### 3. Intégration avec AutonomousStrategist

Le logger est automatiquement intégré dans le workflow d'optimisation :

```python
from agents.integration import create_optimizer_from_engine
from agents.llm_client import LLMConfig, LLMProvider
from agents.orchestration_logger import OrchestrationLogger, generate_session_id

# Créer le logger
session_id = generate_session_id()
orchestration_logger = OrchestrationLogger(session_id=session_id)

# Créer l'optimiseur avec le logger
strategist, executor = create_optimizer_from_engine(
    llm_config=llm_config,
    strategy_name="ema_cross",
    data=df,
    orchestration_logger=orchestration_logger,  # ← Passer le logger
)

# Lancer l'optimisation (le logger enregistre automatiquement)sdf
session = strategist.optimize(
    executor=executor,
    initial_params={...},
    param_bounds={...},
)

# Sauvegarder les logs
orchestration_logger.save_to_file()
```

### 4. Visualisation dans Streamlit

Le système est intégré dans l'interface Streamlit (mode "Optimisation LLM") :

```python
from ui.orchestration_viewer import render_full_orchestration_viewer

# Dans l'interface Streamlit
st.markdown("### 📋 Logs d'Orchestration")
render_full_orchestration_viewer(
    orchestration_logger=orchestration_logger,
    max_entries=50
)
```

---

## 🖥️ Interface Utilisateur

### Vue Complète (Onglets)

L'interface propose 3 vues complémentaires :

#### 1. **Timeline des Logs**
- Affichage chronologique des actions
- Filtres par agent (Analyst, Strategist, Critic, Validator)
- Filtres par type d'action
- Filtres par itération
- Couleurs selon le statut :
  - 🟢 Vert : Complété avec succès
  - 🔴 Rouge : Échec
  - 🟡 Jaune : En cours
  - 🔵 Bleu : En attente
  - ⚫ Gris : Annulé

#### 2. **Résumé de Session**
- Métriques clés :
  - ID de session
  - Nombre total d'itérations
  - Nombre d'entrées de log
  - Agents actifs
- Tableau d'activité par agent
- Graphique de distribution des actions

#### 3. **Métriques Détaillées**
- Backtests lancés vs complétés
- Taux de succès
- Changements de stratégie
- Modifications d'indicateurs
- Décisions prises

### Filtrage Avancé

Les logs peuvent être filtrés selon plusieurs critères :

```python
# Tous les logs d'un agent
analyst_logs = logger.get_logs_by_agent("Analyst")

# Tous les logs d'un type
backtest_logs = logger.get_logs_by_type(OrchestrationActionType.BACKTEST_COMPLETE)

# Tous les logs d'une itération
iter_1_logs = logger.get_logs_for_iteration(1)
```

---

## 💾 Persistance

### Sauvegarde Automatique

Les logs sont sauvegardés automatiquement au format JSON :

```python
# Sauvegarder
save_path = logger.save_to_file()
# → orchestration_logs_20251218_060127.json

# Les logs sont sauvegardés dans le répertoire courant
```

### Format JSON

```json
{
  "session_id": "20251218_060127",
  "total_iterations": 5,
  "total_logs": 42,
  "logs": [
    {
      "timestamp": "2025-12-18T06:01:27.123456",
      "action_type": "analysis_start",
      "agent": "AutonomousStrategist",
      "status": "in_progress",
      "details": {
        "strategy": "ema_cross",
        "initial_params": {"fast_period": 10}
      },
      "iteration": 0,
      "session_id": "20251218_060127"
    },
    ...
  ]
}
```

### Rechargement

```python
# TODO: Implémenter la fonctionnalité de rechargement
# logger.load_from_file("orchestration_logs_20251218_060127.json")
```

---

## 🧪 Tests

### Test Complet d'Intégration

Un test complet valide tout le système :

```bash
python test_ui_orchestration_integration.py
```

**Ce test vérifie :**
- ✅ Création du logger
- ✅ Enregistrement des actions
- ✅ Sauvegarde JSON
- ✅ Filtrage des logs
- ✅ Intégration avec AutonomousStrategist
- ✅ Signatures des fonctions
- ✅ Import des composants UI

### Résultats Attendus

```
================================================================================
✅ TOUS LES TESTS RÉUSSIS!
================================================================================

📝 Prochaines étapes:
  1. Lancer l'interface Streamlit: streamlit run ui/app.py
  2. Sélectionner le mode 'Optimisation LLM'
  3. Configurer les paramètres LLM
  4. Lancer l'optimisation
  5. Observer les logs d'orchestration en temps réel
```

---

## 📊 Exemple Complet

```python
from agents.integration import create_optimizer_from_engine
from agents.llm_client import LLMConfig, LLMProvider
from agents.orchestration_logger import OrchestrationLogger, generate_session_id
from data.loader import load_ohlcv

# 1. Charger les données
df = load_ohlcv("BTCUSDC", "1h")

# 2. Créer le logger
session_id = generate_session_id()
logger = OrchestrationLogger(session_id=session_id)

# 3. Configurer le LLM
llm_config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="llama3.2"
)

# 4. Créer l'optimiseur
strategist, executor = create_optimizer_from_engine(
    llm_config=llm_config,
    strategy_name="ema_cross",
    data=df,
    orchestration_logger=logger,
)

# 5. Lancer l'optimisation
session = strategist.optimize(
    executor=executor,
    initial_params={"fast_period": 10, "slow_period": 21},
    param_bounds={"fast_period": (5, 20), "slow_period": (15, 50)},
    max_iterations=10,
)

# 6. Sauvegarder les logs
logger.save_to_file()

# 7. Afficher le résumé
print(logger.generate_summary())

# 8. Analyser les résultats
print(f"Meilleur Sharpe: {session.best_result.sharpe_ratio}")
print(f"Itérations: {session.current_iteration}")
print(f"Actions enregistrées: {len(logger.logs)}")
```

---

## 🔍 Analyse Post-Mortem

### Générer un Résumé

```python
summary = logger.generate_summary()
print(summary)
```

**Résultat :**
```
================================================================================
ORCHESTRATION LOG SUMMARY - Session: 20251218_060127
================================================================================
Total Iterations: 5
Total Log Entries: 42

Actions Count:
  - analysis_start: 1
  - analysis_complete: 1
  - backtest_launch: 10
  - backtest_complete: 10
  - decision_continue: 4
  - decision_stop: 1
  - indicator_values_change: 15

Agent Activity:
  - AutonomousStrategist: 42 actions
```

### Analyser les Décisions

```python
from agents.orchestration_logger import OrchestrationActionType

decisions = logger.get_logs_by_type(OrchestrationActionType.DECISION_CONTINUE)
for decision in decisions:
    print(f"Itération {decision.iteration}: {decision.details['reason']}")
```

---

## 🚀 Workflow Complet en UI

1. **Lancer Streamlit** : `streamlit run ui/app.py`
2. **Sélectionner le mode** : "🤖 Optimisation LLM"
3. **Configurer LLM** :
   - Provider : Ollama ou OpenAI
   - Modèle : llama3.2, deepseek-r1:32b, etc.
4. **Configurer l'optimisation** :
   - Stratégie : ema_cross, bollinger_atr, etc.
   - Paramètres initiaux
   - Max itérations
5. **Lancer** : Le système affiche automatiquement :
   - Progression en temps réel
   - Logs d'orchestration (timeline + résumé)
   - Métriques clés
   - Pensées du LLM
6. **Analyser** : Les logs sont sauvegardés automatiquement

---

## 📖 Référence API

### OrchestrationLogger

```python
class OrchestrationLogger:
    def __init__(self, session_id: Optional[str] = None)
    
    # Logging
    def log_analysis_start(self, agent: str, details: Optional[Dict] = None)
    def log_analysis_complete(self, agent: str, results: Dict[str, Any], status: OrchestrationStatus = COMPLETED)
    def log_backtest_launch(self, agent: str, params: Dict, combination_id: int, total_combinations: int)
    def log_backtest_complete(self, agent: str, params: Dict, results: Dict, combination_id: int)
    def log_decision(self, agent: str, decision_type: str, reason: str, details: Optional[Dict] = None)
    def log_indicator_values_change(self, agent: str, indicator: str, old_values: Dict, new_values: Dict, reason: str)
    
    # Navigation
    def next_iteration(self)
    
    # Filtrage
    def get_logs_for_iteration(self, iteration: int) -> List[OrchestrationLogEntry]
    def get_logs_by_agent(self, agent: str) -> List[OrchestrationLogEntry]
    def get_logs_by_type(self, action_type: OrchestrationActionType) -> List[OrchestrationLogEntry]
    
    # Persistance
    def save_to_file(self, filepath: Optional[str] = None) -> Optional[str]
    def generate_summary(self) -> str
```

---

## 🎓 Bonnes Pratiques

1. **Créer un logger unique par session** : Utilisez `generate_session_id()`
2. **Passer le logger dès la création** : Intégrez-le dans `create_optimizer_from_engine`
3. **Sauvegarder régulièrement** : Appelez `save_to_file()` après l'optimisation
4. **Analyser les logs** : Utilisez les filtres pour identifier les problèmes
5. **Conserver l'historique** : Les logs JSON sont précieux pour l'audit

---

## 🐛 Dépannage

### Problème : Logs non affichés dans l'UI

**Solution** : Vérifiez que le logger est bien passé à `create_optimizer_from_engine` :

```python
strategist, executor = create_optimizer_from_engine(
    ...,
    orchestration_logger=orchestration_logger,  # ← Ne pas oublier
)
```

### Problème : Session_id identique

**Solution** : Appelez `generate_session_id()` pour chaque nouvelle session :

```python
session_id = generate_session_id()  # ← Génère un ID unique basé sur timestamp
logger = OrchestrationLogger(session_id=session_id)
```

### Problème : Logs non sauvegardés

**Solution** : Appelez explicitement `save_to_file()` :

```python
logger.save_to_file()
```

---

## 📚 Ressources

- **Code source** : `agents/orchestration_logger.py`, `ui/orchestration_viewer.py`
- **Tests** : `test_ui_orchestration_integration.py`
- **Intégration** : `agents/autonomous_strategist.py`, `agents/integration.py`
- **UI** : `ui/app.py` (mode "Optimisation LLM")

---

## 🔮 Évolutions Futures

### Version 1.1.0
- [ ] Rechargement des logs depuis JSON
- [ ] Export des logs au format Excel
- [ ] Graphiques interactifs (Plotly)
- [ ] Comparaison de sessions
- [ ] Alertes temps réel (notifications)

### Version 1.2.0
- [ ] Streaming temps réel via WebSocket
- [ ] Dashboard d'analyse multi-sessions
- [ ] Métriques prédictives
- [ ] Intégration avec TensorBoard

---

## 📜 Changelog

### v1.0.0 - 18 décembre 2025
- ✅ Création du système de logs d'orchestration
- ✅ 20+ types d'actions enregistrées
- ✅ Intégration avec AutonomousStrategist
- ✅ Interface Streamlit complète
- ✅ Tests d'intégration
- ✅ Sauvegarde JSON
- ✅ Filtrage avancé
- ✅ Documentation complète

---

*Dernière mise à jour : 18 décembre 2025 - v1.0.0*
