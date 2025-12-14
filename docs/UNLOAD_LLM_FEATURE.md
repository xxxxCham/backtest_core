# Variable d'Environnement UNLOAD_LLM_DURING_BACKTEST

## Vue d'Ensemble

La variable d'environnement `UNLOAD_LLM_DURING_BACKTEST` permet de contrôler le déchargement du modèle LLM de la mémoire GPU pendant les backtests autonomes.

**Valeur par défaut** : `False` (compatible CPU-only)

---

## 🎯 Objectif

### Problème Initial
Lors des backtests autonomes avec agents LLM, le modèle LLM reste en VRAM GPU, limitant l'espace disponible pour les calculs NumPy/CuPy intensifs.

### Solution
Décharger temporairement le LLM du GPU pendant les calculs de backtest :
1. **Décharge** : LLM quitté du GPU → VRAM libre
2. **Calculs** : Backtest avec 100% VRAM disponible
3. **Recharge** : LLM ramené en GPU pour prochaine itération

---

## 📊 Impact Performance

### Avec UNLOAD_LLM=True (GPU Optimization)
- ✅ **VRAM libre** : 100% disponible pour calculs
- ✅ **Pas d'OOM** : Évite les erreurs Out-of-Memory
- ⚠️ **Latence** : +2-5s par itération (rechargement modèle)

**Recommandé pour** :
- GPU avec VRAM limitée (< 12 GB)
- Modèles LLM volumineux (> 4 GB)
- Backtests sur grandes séries (> 100k bars)

### Avec UNLOAD_LLM=False (Default)
- ✅ **Pas de latence** : LLM reste en mémoire
- ✅ **Itérations rapides** : 0s overhead
- ⚠️ **VRAM partagée** : Risque d'OOM sur petites GPU

**Recommandé pour** :
- Systèmes CPU-only (majorité des utilisateurs)
- GPU avec VRAM abondante (> 16 GB)
- Modèles LLM petits (< 2 GB)
- Backtests sur petites séries (< 50k bars)

---

## 🔧 Configuration

### Méthode 1 : Fichier .env

```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=False  # Default, CPU-compatible
```

ou

```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=True   # GPU optimization
```

### Méthode 2 : Variable d'Environnement

**PowerShell** :
```powershell
$env:UNLOAD_LLM_DURING_BACKTEST = "True"
```

**Linux/Mac** :
```bash
export UNLOAD_LLM_DURING_BACKTEST=True
```

### Méthode 3 : Paramètre Python

```python
from agents import create_optimizer_from_engine
from agents.llm_client import LLMConfig, LLMProvider

config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
strategist, executor = create_optimizer_from_engine(
    llm_config=config,
    strategy_name="ema_cross",
    data=df,
    unload_llm_during_backtest=True,  # Override env var
)
```

---

## 💡 Valeurs Acceptées

### True (Déchargement activé)
- `True` (case-insensitive)
- `1`
- `yes` (case-insensitive)

### False (Déchargement désactivé)
- `False` (case-insensitive)
- `0`
- `no` (case-insensitive)
- *(non définie)* → Default = False

---

## 🧪 Tests

10 tests unitaires valident le comportement :

```bash
python -m pytest tests/test_unload_llm_env.py -v
```

**Tests couverts** :
1. ✅ Valeur par défaut `False` si variable non définie
2. ✅ Parsing `True`, `1`, `yes` → True
3. ✅ Parsing `False`, `0`, `no` → False
4. ✅ Case-insensitivity (`TRUE`, `true`, `True`)
5. ✅ Override par paramètre explicite
6. ✅ Appel `GPUMemoryManager` si True
7. ✅ Pas d'appel `GPUMemoryManager` si False

**Résultat** :
```
10 passed in 2.84s ✅
```

---

## 📚 Implémentation

### Fichiers Modifiés

1. **`agents/autonomous_strategist.py`**
   - Lecture de `UNLOAD_LLM_DURING_BACKTEST`
   - Logique conditionnelle dans `_run_backtest_with_gpu_optimization()`
   - Correction bug : `self.llm_client` → `self.llm`

2. **`.env.example`**
   - Documentation de la variable
   - Valeur par défaut : `False`

3. **`ENVIRONMENT.md`**
   - Section complète sur GPU Memory Management
   - Exemples CPU-only vs GPU systems
   - Troubleshooting OOM

### Fichiers Créés

1. **`tests/test_unload_llm_env.py`** (250 lignes)
   - 10 tests unitaires
   - Validation comportement complet

---

## 🐛 Troubleshooting

### Problème : Latence importante entre itérations

**Cause** : `UNLOAD_LLM_DURING_BACKTEST=True` sur système CPU-only  
**Solution** : `UNLOAD_LLM_DURING_BACKTEST=False`

**Vérification** :
```python
import os
print(os.getenv('UNLOAD_LLM_DURING_BACKTEST', 'NOT_SET'))
```

### Problème : OOM (Out of Memory) GPU

**Cause** : `UNLOAD_LLM_DURING_BACKTEST=False` avec GPU trop petite  
**Solution** : `UNLOAD_LLM_DURING_BACKTEST=True`

**Vérification VRAM** :
```bash
nvidia-smi
```

### Problème : Variable ignorée

**Cause** : Paramètre explicite override la variable d'env  
**Solution** : Passer `unload_llm_during_backtest=None` pour auto-détection

---

## 📖 Documentation Complète

| Fichier | Description |
|---------|-------------|
| [ENVIRONMENT.md](ENVIRONMENT.md) | Documentation toutes variables d'env |
| [.env.example](.env.example) | Template configuration |
| [LLM_INTEGRATION_README.md](LLM_INTEGRATION_README.md) | Guide complet agents LLM |

---

## 🔄 Workflow Typique

### Scénario 1 : Développement Local (CPU-only)

```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=False  # Pas de latence
```

**Résultat** :
- Itérations rapides
- Pas d'overhead
- Compatible tous systèmes

### Scénario 2 : Production GPU (VRAM limitée)

```bash
# .env
UNLOAD_LLM_DURING_BACKTEST=True   # Libère VRAM
```

**Résultat** :
- 100% VRAM pour backtests
- Pas d'OOM
- +2-5s par itération acceptable

### Scénario 3 : Benchmarking

```bash
# Tester les deux modes
UNLOAD_LLM_DURING_BACKTEST=False python benchmark.py
UNLOAD_LLM_DURING_BACKTEST=True python benchmark.py

# Comparer temps total et pics mémoire
```

---

## 🎓 Références

### Code Source
- `agents/autonomous_strategist.py` lignes 180-250
- `agents/ollama_manager.py` (GPUMemoryManager)

### Tests
- `tests/test_unload_llm_env.py`

### Documentation
- [ENVIRONMENT.md](ENVIRONMENT.md) section "GPU Memory Management"
- [README.md](README.md) section "Configuration Critique"

---

*Feature implémentée : 13/12/2025*  
*Tests : 10/10 passants ✅*
