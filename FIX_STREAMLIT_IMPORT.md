# Fix: ImportError Streamlit - get_ui_indicators

## 🔴 Erreur Rencontrée

```python
ImportError: cannot import name 'get_ui_indicators' from 'strategies.indicators_mapping'
```

**Contexte** : Lors du lancement de Streamlit `ui/app.py`, l'import échoue dans `ui/constants.py`.

## 🔍 Diagnostic

### Tests effectués :

1. ✅ **Vérification du fichier source** :
   - `strategies/indicators_mapping.py` ligne 224 : `def get_ui_indicators()` existe
   - Fonction exportée dans `__all__` (ligne 314)

2. ✅ **Test import direct** :
   ```bash
   python -c "from strategies.indicators_mapping import get_ui_indicators"
   # → OK, pas d'erreur
   ```

3. ✅ **Test import ui.constants** :
   ```bash
   python -c "from ui.constants import PARAM_CONSTRAINTS"
   # → OK, pas d'erreur
   ```

### 🎯 Cause Racine

**Cache obsolète de Streamlit** : Streamlit garde en mémoire une ancienne version du module où `get_ui_indicators` n'existait pas encore (ou avait un nom différent).

## ✅ Solution

### Option 1 : Nettoyer le cache via le navigateur (RECOMMANDÉ)

1. Dans le navigateur où Streamlit tourne :
   - Appuyer sur **`C`** → nettoie le cache
   - Puis **`R`** → recharge l'app

### Option 2 : Commande terminal

```powershell
streamlit cache clear
```

### Option 3 : Redémarrer Streamlit

```powershell
# Arrêter avec Ctrl+C
streamlit run ui/app.py
```

## 📝 Vérification Post-Fix

Script créé : `tools/verify_ui_imports.py`

```bash
python tools/verify_ui_imports.py
```

Teste tous les imports critiques de l'UI.

## 🔄 Prévention Future

**Pourquoi ce problème arrive** :
- Streamlit met en cache les imports pour accélérer les reloads
- Lors de refactoring/renommage de fonctions, le cache devient obsolète
- Le hot-reload ne nettoie PAS toujours le cache automatiquement

**Bonne pratique** :
- Après un refactoring majeur : `streamlit cache clear`
- Ou relancer Streamlit complètement (Ctrl+C puis rerun)

---

**Status** : ✅ Résolu
**Date** : 29/12/2025
**Méthode** : Nettoyage cache Streamlit
