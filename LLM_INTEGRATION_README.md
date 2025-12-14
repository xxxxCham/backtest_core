# 🤖 Intégration LLM - Système Fonctionnel

## Vue d'ensemble

Le système d'intégration LLM a été complètement refait en s'inspirant du système fonctionnel de **ThreadX_big**. Le nouveau système offre :

✅ **Détection automatique** des modèles Ollama installés
✅ **Auto-démarrage** d'Ollama si nécessaire
✅ **Sélection intelligente** des modèles avec informations (taille, description)
✅ **Fallback robuste** si Ollama n'est pas disponible
✅ **Interface utilisateur** intuitive avec feedback en temps réel

---

## 📁 Fichiers créés/modifiés

### 1. **Nouveau module** - [agents/ollama_manager.py](agents/ollama_manager.py)

Gestionnaire Ollama complet avec :
- `ensure_ollama_running()` - Démarre Ollama automatiquement
- `list_ollama_models()` - Liste les modèles installés
- `is_ollama_available()` - Vérifie la connexion
- `unload_model()` - Décharge un modèle de la mémoire
- `cleanup_all_models()` - Nettoie tous les modèles
- `prepare_for_llm_run()` - Prépare l'environnement complet

### 2. **Nouveau composant UI** - [ui/components/model_selector.py](ui/components/model_selector.py)

Sélecteur de modèles LLM avec :
- Liste dynamique des modèles disponibles
- Tri par ordre de recommandation
- Informations sur chaque modèle (taille, description)
- Fallback intelligent si Ollama n'est pas accessible
- Catégories de modèles recommandés :
  * `RECOMMENDED_FOR_ANALYSIS` - Pour l'analyse de données
  * `RECOMMENDED_FOR_STRATEGY` - Pour la génération de stratégies
  * `RECOMMENDED_FOR_CRITICISM` - Pour la critique/validation
  * `RECOMMENDED_FOR_FAST` - Pour des tests rapides

### 3. **UI mise à jour** - [ui/app.py](ui/app.py)

Intégration complète dans l'UI Streamlit :
- Détection automatique de la connexion Ollama
- Bouton pour démarrer Ollama si nécessaire
- Sélecteur de modèles avec liste dynamique
- Affichage des informations du modèle sélectionné

---

## 🚀 Utilisation

### Dans l'interface Streamlit

1. **Lancer l'application**
   ```bash
   streamlit run ui/app.py
   ```

2. **Sélectionner le mode LLM**
   - Dans la sidebar, choisir "🤖 Optimisation LLM"

3. **Configuration automatique**
   - L'interface vérifie automatiquement si Ollama est connecté
   - Si Ollama n'est pas démarré, un bouton "🚀 Démarrer Ollama" apparaît
   - La liste des modèles se remplit automatiquement

4. **Sélection du modèle**
   - Choisir un modèle dans la liste déroulante
   - Les modèles sont triés par recommandation (les meilleurs en premier)
   - Les informations (taille, description) s'affichent automatiquement

### En Python direct

```python
from agents.ollama_manager import (
    ensure_ollama_running,
    list_ollama_models,
    is_ollama_available,
)

# Vérifier si Ollama est disponible
if is_ollama_available():
    print("✅ Ollama connecté")
else:
    # Démarrer Ollama automatiquement
    success, message = ensure_ollama_running()
    print(message)

# Lister les modèles disponibles
models = list_ollama_models()
print(f"Modèles installés : {models}")
```

### Avec le composant UI

```python
import streamlit as st
from ui.components.model_selector import (
    render_model_selector,
    RECOMMENDED_FOR_STRATEGY,
)

# Dans votre page Streamlit
model = render_model_selector(
    label="Modèle Strategist",
    key="strategist_model",
    preferred_order=RECOMMENDED_FOR_STRATEGY,
    help_text="Sélectionnez un modèle pour générer des stratégies"
)

st.write(f"Modèle sélectionné : {model}")
```

---

## 📊 Modèles recommandés

Le système inclut une liste de modèles recommandés avec leurs caractéristiques :

| Modèle | Taille | Usage recommandé | Performance |
|--------|--------|------------------|-------------|
| **deepseek-r1:70b** | 34 GB | Stratégies complexes | ⭐⭐⭐⭐⭐ |
| **deepseek-r1:32b** | 19 GB | Optimal - Meilleur rapport | ⭐⭐⭐⭐⭐ |
| **qwq:32b** | 23 GB | Raisonnement & Analyse | ⭐⭐⭐⭐ |
| **qwen2.5:32b** | 19 GB | Alternative polyvalente | ⭐⭐⭐⭐ |
| **mistral:22b** | 13 GB | Équilibré - Bon pour critique | ⭐⭐⭐⭐ |
| **gemma3:27b** | 17 GB | Analyse rapide | ⭐⭐⭐ |
| **deepseek-r1:8b** | 5 GB | Tests rapides | ⭐⭐⭐ |
| **mistral:7b-instruct** | 4 GB | Ultra rapide | ⭐⭐ |
| **llama3.2** | 2 GB | Léger pour tests | ⭐⭐ |

---

## 🔧 Configuration

### Variables d'environnement (optionnel)

Le système utilise les variables d'environnement par défaut, mais vous pouvez les personnaliser :

```bash
# Provider LLM (ollama ou openai)
BACKTEST_LLM_PROVIDER=ollama

# Modèle par défaut
BACKTEST_LLM_MODEL=deepseek-r1:32b

# URL Ollama
OLLAMA_HOST=http://localhost:11434

# OpenAI (si utilisé)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 🎯 Fonctionnalités clés

### 1. Détection automatique

L'interface détecte automatiquement si Ollama est disponible :
- ✅ **Ollama connecté** → Affiche la liste des modèles installés
- ⚠️ **Ollama non détecté** → Propose de le démarrer automatiquement
- ❌ **Ollama non installé** → Utilise une liste de modèles en fallback

### 2. Auto-démarrage

Si Ollama n'est pas démarré, un simple clic sur "🚀 Démarrer Ollama" :
- Lance le service Ollama en arrière-plan
- Attend qu'il soit prêt (max 10 secondes)
- Rafraîchit automatiquement l'interface

### 3. Informations sur les modèles

Pour chaque modèle, l'interface affiche :
- **Nom** : deepseek-r1:32b
- **Taille** : ~19 GB
- **Description** : Optimal - Meilleur rapport qualité/prix

### 4. Tri intelligent

Les modèles sont triés par ordre de recommandation :
1. Les modèles recommandés pour la tâche en premier
2. Ensuite, les autres modèles par ordre alphabétique

---

## 🔄 Différences avec l'ancien système

### Avant (dysfonctionnel)

❌ Saisie manuelle du nom du modèle (risque d'erreur)
❌ Pas de vérification de la connexion Ollama
❌ Pas d'information sur les modèles disponibles
❌ Pas de fallback en cas de problème

### Après (système fonctionnel)

✅ Sélection depuis une liste dynamique
✅ Détection automatique de la connexion
✅ Auto-démarrage d'Ollama si nécessaire
✅ Informations complètes sur chaque modèle
✅ Fallback robuste avec liste de modèles recommandés
✅ Tri par ordre de recommandation

---

## 📝 API Reference

### ollama_manager

```python
from agents.ollama_manager import *

# Vérifier disponibilité
is_available = is_ollama_available() -> bool

# Démarrer Ollama
success, message = ensure_ollama_running() -> Tuple[bool, str]

# Lister modèles
models = list_ollama_models() -> List[str]

# Décharger un modèle
success = unload_model("deepseek-r1:32b") -> bool

# Nettoyer tous les modèles
count = cleanup_all_models() -> int

# Préparer pour un run LLM
success, message = prepare_for_llm_run() -> Tuple[bool, str]
```

### model_selector

```python
from ui.components.model_selector import *

# Obtenir liste des modèles
models = get_available_models_for_ui(
    preferred_order=RECOMMENDED_FOR_STRATEGY,
    fallback=None
) -> List[str]

# Obtenir infos sur un modèle
info = get_model_info("deepseek-r1:32b") -> dict
# Retourne: {
#     "name": "deepseek-r1:32b",
#     "size_gb": 19,
#     "description": "Optimal - Meilleur rapport qualité/prix"
# }

# Rendu d'un sélecteur Streamlit
model = render_model_selector(
    label="Modèle LLM",
    key="llm_model",
    preferred_order=RECOMMENDED_FOR_STRATEGY,
    help_text="Sélectionnez un modèle"
) -> str
```

---

## 🧪 Tests

Pour tester le système :

```bash
# 1. Tester ollama_manager
python -c "from agents.ollama_manager import *; print(list_ollama_models())"

# 2. Lancer l'UI
streamlit run ui/app.py
```

---

## 🎨 Captures d'écran

### Mode Ollama connecté

```
✅ Ollama connecté

┌─────────────────────────────┐
│ Modèle Ollama               │
├─────────────────────────────┤
│ deepseek-r1:32b            │ ◀ Sélectionné
│ qwq:32b                     │
│ mistral:22b                 │
│ ...                         │
└─────────────────────────────┘

📦 ~19 GB | Optimal - Meilleur rapport qualité/prix
```

### Mode Ollama non détecté

```
⚠️ Ollama non détecté

┌─────────────────────────────┐
│ 🚀 Démarrer Ollama          │
└─────────────────────────────┘
```

---

## 🔗 Intégration avec ThreadX_big

Ce système est directement inspiré de ThreadX_big :
- **Architecture** : Même structure modulaire
- **Composants** : ollama_manager + model_selector
- **UI Pattern** : Détection + Auto-start + Liste dynamique
- **Robustesse** : Fallback + Gestion d'erreurs

---

## 📚 Ressources

- **Code source** :
  - [agents/ollama_manager.py](agents/ollama_manager.py)
  - [ui/components/model_selector.py](ui/components/model_selector.py)
  - [ui/app.py](ui/app.py) (lignes 657-706)

- **Référence ThreadX_big** :
  - `D:\ThreadX_big\src\threadx\llm\ollama_manager.py`
  - `D:\ThreadX_big\src\threadx\ui\components\model_selector.py`

---

**Version** : 1.0.0
**Date** : Décembre 2025
**Statut** : ✅ Système fonctionnel et testé
