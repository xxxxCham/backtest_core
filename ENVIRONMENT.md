# Configuration - Variables d'Environnement

> **Documentation complète** des variables d'environnement pour contrôler le comportement du système.

---

## 🔧 Configuration Rapide

```bash
# Copier le template
cp .env.example .env

# Éditer avec vos valeurs
notepad .env  # Windows
nano .env     # Linux/Mac
```

---

## 📂 Variables Disponibles

### **Données**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATA_DIR` | `data/sample_data` | Dossier par défaut pour les données d'exemple |
| `BACKTEST_DATA_DIR` | *(vide)* | Dossier personnalisé pour fichiers Parquet/CSV |

**Exemple :**
```bash
BACKTEST_DATA_DIR=D:/Trading/Historical_Data
```

---

### **Trading & Capital**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `INITIAL_CAPITAL` | `10000` | Capital initial en USD |
| `DEFAULT_LEVERAGE` | `1` | Levier par défaut (1 = pas de levier) |

---

### **Performance & Parallélisation**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MAX_WORKERS` | `8` | Nombre de threads pour sweep parallèle |
| `USE_GPU` | `true` | Activer le backend GPU (CuPy) si disponible |

---

### **Logging & Observabilité**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `LOG_LEVEL` | `INFO` | Niveau de log général : `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `BACKTEST_LOG_LEVEL` | *(vide)* | Niveau de log spécifique au backtest (override LOG_LEVEL) |

**Mode Debug Complet :**
```bash
BACKTEST_LOG_LEVEL=DEBUG
```

Affiche :
- Spans chronométrés pour chaque phase
- Détails des indicateurs calculés
- États des agents LLM
- Métriques de performance

---

### **Configuration LLM**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `BACKTEST_LLM_PROVIDER` | `ollama` | Provider LLM : `ollama` ou `openai` |
| `BACKTEST_LLM_MODEL` | `deepseek-r1:8b` | Modèle par défaut |
| `OLLAMA_HOST` | `http://localhost:11434` | URL du serveur Ollama |
| `OPENAI_API_KEY` | *(requis si OpenAI)* | Clé API OpenAI |
| `BACKTEST_LLM_TEMPERATURE` | `0.7` | Température (0.0=déterministe, 1.0=créatif) |
| `BACKTEST_LLM_MAX_TOKENS` | `2000` | Limite de tokens par réponse |

**Exemple Ollama :**
```bash
BACKTEST_LLM_PROVIDER=ollama
BACKTEST_LLM_MODEL=deepseek-r1:32b
OLLAMA_HOST=http://192.168.1.100:11434  # Serveur distant
```

**Exemple OpenAI :**
```bash
BACKTEST_LLM_PROVIDER=openai
BACKTEST_LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

---

### **🔴 GPU Memory Management (CRITIQUE)**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `UNLOAD_LLM_DURING_BACKTEST` | `False` | Décharger LLM du GPU pendant calculs |

**⚠️ IMPORTANT - Comprendre cette variable :**

#### **Problème :**
Les LLMs chargés occupent la VRAM GPU, empêchant les calculs NumPy/CuPy d'utiliser toute la mémoire disponible.

#### **Solution :**
- **CPU-only systems** : `False` (défaut)
  - Pas de GPU → déchargement inutile → latence inutile
  - **Recommandé** pour la plupart des utilisateurs

- **GPU systems avec CuPy** : `True`
  - Libère 100% de la VRAM pour les calculs intensifs
  - Trade-off : +2-5s latence unload/reload entre itérations

#### **Mesure :**
```bash
# Tester sur 1 itération d'optimisation
UNLOAD_LLM_DURING_BACKTEST=False  # Baseline

# Si calculs GPU OOM ou lents :
UNLOAD_LLM_DURING_BACKTEST=True   # Test avec déchargement
```

#### **Exemple d'impact :**

| Configuration | VRAM LLM | VRAM Calculs | Latence Iter | Recommandé |
|---------------|----------|--------------|--------------|------------|
| False (défaut) | 8 GB | 16 GB restants | 0s overhead | ✅ CPU-only |
| True (GPU opt) | 0 GB | 24 GB libres | +3s overhead | ✅ GPU avec CuPy |

---

### **Walk-Forward Validation**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `WALK_FORWARD_WINDOWS` | `5` | Nombre de fenêtres de validation |
| `WALK_FORWARD_MIN_TEST_SAMPLES` | `50` | Taille min du test set |

**Exemple - Validation agressive :**
```bash
WALK_FORWARD_WINDOWS=10
WALK_FORWARD_MIN_TEST_SAMPLES=100
```

---

### **Optuna Optimization**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `OPTUNA_SAMPLER` | `tpe` | Algorithme : `tpe`, `cmaes`, `random` |
| `OPTUNA_ENABLE_PRUNING` | `True` | Activer arrêt précoce des runs peu prometteurs |

---

### **Constraints & Risques**

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MAX_OVERFITTING_RATIO` | `1.5` | Ratio max train/test avant alerte overfitting |

---

## 🎯 Configurations Recommandées

### **1. Développement Local (CPU-only)**
```bash
# .env
DATA_DIR=data/sample_data
INITIAL_CAPITAL=10000
LOG_LEVEL=DEBUG
BACKTEST_LLM_PROVIDER=ollama
BACKTEST_LLM_MODEL=deepseek-r1:8b
UNLOAD_LLM_DURING_BACKTEST=False  # ⚠️ Important
```

### **2. Production GPU (NVIDIA RTX 4090)**
```bash
# .env
BACKTEST_DATA_DIR=/mnt/ssd/trading_data
INITIAL_CAPITAL=100000
USE_GPU=true
LOG_LEVEL=INFO
BACKTEST_LLM_PROVIDER=ollama
BACKTEST_LLM_MODEL=deepseek-r1:32b
UNLOAD_LLM_DURING_BACKTEST=True   # 🚀 GPU optimization ON
MAX_WORKERS=16
```

### **3. Cloud OpenAI**
```bash
# .env
BACKTEST_LLM_PROVIDER=openai
BACKTEST_LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
UNLOAD_LLM_DURING_BACKTEST=False
LOG_LEVEL=WARNING  # Réduire verbosité
```

### **4. Research Station (Multi-GPU)**
```bash
# .env
USE_GPU=true
UNLOAD_LLM_DURING_BACKTEST=True
MAX_WORKERS=32
WALK_FORWARD_WINDOWS=10
OPTUNA_SAMPLER=cmaes
LOG_LEVEL=DEBUG
```

---

## 🧪 Validation Configuration

```bash
# Tester variables d'env chargées
python -c "import os; print(os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'NOT_SET'))"

# Lancer backtest avec override
BACKTEST_LOG_LEVEL=DEBUG python __main__.py backtest -s ema_cross -d data.parquet

# Vérifier GPU
python -c "import cupy; print(cupy.cuda.Device(0).mem_info)"
```

---

## ⚠️ Pièges Courants

### **1. GPU Unload sur CPU**
**Symptôme :** Latence +5s par itération sans gain  
**Cause :** `UNLOAD_LLM_DURING_BACKTEST=True` sur système sans GPU  
**Fix :** `UNLOAD_LLM_DURING_BACKTEST=False`

### **2. VRAM OOM**
**Symptôme :** `CuPy OutOfMemoryError` durant calculs  
**Cause :** LLM occupe toute la VRAM  
**Fix :** `UNLOAD_LLM_DURING_BACKTEST=True`

### **3. Logs trop verbeux**
**Symptôme :** Terminal inondé de spans  
**Cause :** `BACKTEST_LOG_LEVEL=DEBUG`  
**Fix :** `BACKTEST_LOG_LEVEL=INFO`

### **4. Modèle OpenAI introuvable**
**Symptôme :** `AuthenticationError` ou `ModelNotFound`  
**Cause :** `OPENAI_API_KEY` manquante ou invalide  
**Fix :** Vérifier clé API valide

---

## 📊 Monitoring Variables

**En Python :**
```python
import os
from agents import LLMConfig

config = LLMConfig.from_env()
print(f"Provider: {config.provider}")
print(f"Model: {config.model}")
print(f"GPU Unload: {os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'False')}")
```

**En CLI :**
```bash
python __main__.py validate --all  # Vérifie toute la config
```

---

## 🔗 Références

- [Configuration LLM](LLM_INTEGRATION_README.md)
- [CLI Reference](CLI_REFERENCE.md)
- [Copilot Instructions](copilot-instructions.md)

---

*Dernière mise à jour : 13/12/2025*
