# Demo Scripts - Guide d'Utilisation

Ce dossier contient des scripts de démonstration pour tester rapidement les fonctionnalités du moteur de backtest.

---

## 📁 Scripts Disponibles

### **quick_test.py** - Test Rapide du Moteur
Test minimal du pipeline de backtesting avec données synthétiques.

```bash
python demo/quick_test.py
```

**Usage :**
- Génère des données OHLCV synthétiques
- Exécute un backtest avec stratégie `ema_cross`
- Affiche les métriques de base (Sharpe, Total Return, Drawdown)

**Idéal pour :**
- Vérifier que l'installation fonctionne
- Test rapide après modification de code
- Première découverte du système

---

### **real_data_backtest.py** - Backtest Données Réelles
Backtest avec données historiques réelles (fichiers Parquet/CSV).

```bash
# Configurer le chemin vers les données
$env:BACKTEST_DATA_DIR = "D:\path\to\parquet"

python demo/real_data_backtest.py
```

**Usage :**
- Charge des données historiques depuis `BACKTEST_DATA_DIR`
- Exécute plusieurs stratégies en parallèle
- Compare les performances

**Idéal pour :**
- Valider stratégies sur données réelles
- Comparer plusieurs approches
- Benchmarking performance

---

### **test_env_config.py** - Test Configuration d'Environnement
Valide et teste les variables d'environnement.

```bash
# Afficher config actuelle
python demo/test_env_config.py --scenario current

# Tester config CPU-only
python demo/test_env_config.py --scenario cpu

# Tester config GPU
python demo/test_env_config.py --scenario gpu

# Tester config OpenAI
python demo/test_env_config.py --scenario openai

# Tester walk-forward validation
python demo/test_env_config.py --scenario walk-forward

# Test complet avec backtest
python demo/test_env_config.py --scenario test
```

**Scénarios disponibles :**
- `current` : Affiche la config actuelle
- `cpu` : Configuration CPU-only (défaut recommandé)
- `gpu` : Configuration GPU avec optimisation mémoire
- `openai` : Configuration OpenAI au lieu d'Ollama
- `walk-forward` : Validation walk-forward stricte
- `test` : Exécute un backtest de test

**Idéal pour :**
- Débugger problèmes de configuration
- Comprendre l'impact des variables d'env
- Valider setup avant production

---

### **demo_storage.py** - Test Système de Stockage
Test du système de stockage persistant des résultats.

```bash
python demo/demo_storage.py
```

**Usage :**
- Teste l'enregistrement de résultats dans `backtest_results/`
- Vérifie l'indexation et la recherche
- Valide la sérialisation/désérialisation

**Idéal pour :**
- Vérifier système de stockage fonctionnel
- Débugger problèmes de sauvegarde
- Comprendre la structure des résultats

---

## 🎯 Workflows Recommandés

### **Workflow 1: Premier Contact**
```bash
# 1. Test rapide
python demo/quick_test.py

# 2. Vérifier config
python demo/test_env_config.py --scenario current

# 3. Lancer interface
streamlit run ui/app.py
```

---

### **Workflow 2: Développement Stratégie**
```bash
# 1. Configurer logging debug
$env:BACKTEST_LOG_LEVEL = 'DEBUG'

# 2. Tester avec données réelles
$env:BACKTEST_DATA_DIR = "D:\data"
python demo/real_data_backtest.py

# 3. Afficher résultats détaillés
python __main__.py visualize -i backtest_results/latest.json -d data.parquet
```

---

### **Workflow 3: Optimisation GPU**
```bash
# 1. Configurer GPU optimization
.\set_config.ps1 gpu

# 2. Vérifier config
python demo/test_env_config.py --scenario current

# 3. Lancer optimisation Optuna
python __main__.py optuna -s ema_cross -d BTCUSDC_1h.parquet -n 100 --pruning
```

---

### **Workflow 4: Production**
```bash
# 1. Configurer mode production
.\set_config.ps1 prod

# 2. Valider toute la config
python __main__.py validate --all

# 3. Lancer sweep massif
python __main__.py sweep -s bollinger_atr -d BTCUSDC_4h.parquet --granularity 0.2
```

---

## 🔧 Configuration Variables d'Environnement

### **Fichier .env**
```bash
# Copier le template
cp .env.example .env

# Éditer avec vos valeurs
notepad .env  # Windows
```

### **PowerShell (temporaire)**
```powershell
# Configuration rapide via script
.\set_config.ps1 cpu      # CPU-only
.\set_config.ps1 gpu      # GPU optimisé
.\set_config.ps1 openai   # OpenAI

# Ou manuellement
$env:UNLOAD_LLM_DURING_BACKTEST = 'False'
$env:BACKTEST_LLM_MODEL = 'deepseek-r1:8b'
```

### **Bash (Linux/Mac)**
```bash
export UNLOAD_LLM_DURING_BACKTEST=False
export BACKTEST_LLM_MODEL=deepseek-r1:8b
```

---

## 📊 Variables Critiques

| Variable | Défaut | Critique pour |
|----------|--------|---------------|
| `UNLOAD_LLM_DURING_BACKTEST` | `False` | ⚠️ CPU-only systems |
| `BACKTEST_DATA_DIR` | `data/sample_data` | Données réelles |
| `BACKTEST_LLM_MODEL` | `deepseek-r1:8b` | Performance LLM |
| `USE_GPU` | `true` | Calculs CuPy |
| `BACKTEST_LOG_LEVEL` | `INFO` | Debug |

Voir [ENVIRONMENT.md](../ENVIRONMENT.md) pour documentation complète.

---

## 🐛 Debugging

### **Problème 1: Latence excessive**
```bash
# Vérifier si GPU unload est activé sur CPU
python demo/test_env_config.py --scenario current

# Si UNLOAD_LLM_DURING_BACKTEST=True sur CPU:
$env:UNLOAD_LLM_DURING_BACKTEST = 'False'
```

### **Problème 2: GPU Out of Memory**
```bash
# Activer déchargement LLM
$env:UNLOAD_LLM_DURING_BACKTEST = 'True'

# Tester avec modèle plus léger
$env:BACKTEST_LLM_MODEL = 'deepseek-r1:8b'
```

### **Problème 3: Données introuvables**
```bash
# Vérifier chemin
echo $env:BACKTEST_DATA_DIR

# Définir chemin correct
$env:BACKTEST_DATA_DIR = "D:\Trading\Historical_Data"

# Lister données disponibles
python __main__.py list data
```

### **Problème 4: LLM ne répond pas**
```bash
# Vérifier serveur Ollama
curl http://localhost:11434/api/tags

# Lister modèles disponibles
ollama list

# Télécharger modèle si absent
ollama pull deepseek-r1:8b
```

---

## 📚 Documentation Complète

| Document | Description |
|----------|-------------|
| [ENVIRONMENT.md](../ENVIRONMENT.md) | Variables d'env détaillées |
| [CLI_REFERENCE.md](../.github/CLI_REFERENCE.md) | Commandes CLI |
| [LLM_INTEGRATION_README.md](../LLM_INTEGRATION_README.md) | Système d'agents LLM |
| [README.md](../README.md) | Vue d'ensemble projet |

---

*Dernière mise à jour : 13/12/2025*
