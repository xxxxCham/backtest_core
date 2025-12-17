# Pytest-Watch - Guide de Configuration

## ⚠️ Problème avec pyproject.toml

**Erreur rencontrée** :
```
configparser.ParsingError: Source contains parsing errors: 'D:\\backtest_core\\pyproject.toml'
```

**Cause** : `pytest-watch` est un outil ancien qui essaie de parser `pyproject.toml` comme un fichier INI classique, mais le format TOML moderne utilise une syntaxe différente (notamment pour les arrays `[]`).

---

## ✅ Solutions

### Solution 1 : Script PowerShell (RECOMMANDÉ)

Utiliser le script fourni qui contourne le problème :

```powershell
.\watch-tests.ps1
.\watch-tests.ps1 tests/test_execution.py
.\watch-tests.ps1 -Verbose
```

### Solution 2 : Commande directe (sans config)

Lancer pytest-watch avec options CLI uniquement :

```powershell
.venv\Scripts\pytest-watch.exe --clear --nobeep --runner "pytest" tests/ -- -v --tb=short
```

### Solution 3 : Alternatives modernes

#### VS Code Test Explorer ⭐ (RECOMMANDÉ)
1. Ouvrir la barre latérale Tests (Ctrl+Shift+T)
2. Activer "Auto Run" en haut à droite
3. Les tests se relancent automatiquement à chaque sauvegarde

**Avantages** :
- ✅ Intégré à VS Code (pas de dépendance)
- ✅ Interface graphique intuitive
- ✅ Aucune configuration requise
- ✅ Fonctionne immédiatement

#### pytest-testmon (⚠️ Problèmes de compatibilité)
```powershell
pip install pytest-testmon
pytest --testmon  # ⚠️ Ne fonctionne PAS avec pytest 9.0+
```
- ❌ Plugin non chargé automatiquement avec pytest 9.0.2
- ❌ Nécessite configuration manuelle complexe
- ⚠️ Non recommandé actuellement

#### pytest-xdist (Obsolète pour watch mode)
```powershell
pip install pytest-xdist
pytest -f tests/  # ⚠️ Option -f supprimée dans versions récentes
```
- ❌ looponfail retiré des versions modernes
- ℹ️ Utile uniquement pour parallélisation (`-n auto`)

---

## 📁 Fichiers de configuration

### `.pytest-watch.cfg` (INI)
Configuration compatible avec pytest-watch (non utilisée par défaut car le tool parse quand même pyproject.toml).

### `pytest-watch.ini` (INI)
Alternative au format TOML (non prise en charge automatiquement).

### `pyproject.toml` (TOML)
Configuration principale du projet - **NE PAS MODIFIER** pour pytest-watch.

---

## 🔧 Détails techniques

### Pourquoi pytest-watch échoue ?

1. pytest-watch utilise `configparser` (parser INI de Python)
2. `configparser` ne comprend pas la syntaxe TOML moderne
3. Les arrays TOML (`key = ["value1", "value2"]`) causent des erreurs de parsing INI

### Exemple d'erreur :
```toml
# TOML valide mais INI invalide
classifiers = [
    "Development Status :: 4 - Beta",
]  # <- Le ] seul sur une ligne cause une erreur INI
```

### Solution permanente (future) :

Passer à `pytest-testmon` ou attendre que pytest-watch supporte TOML nativement (peu probable car le projet est peu maintenu).

---

## 📊 Comparaison des outils

| Outil | Support TOML | Vitesse | Maintenance | Recommandation |
|-------|-------------|---------|-------------|----------------|
| pytest-watch | ❌ | Moyen | Faible | ⚠️ Legacy |
| pytest-testmon | ✅ | Rapide | Active | ⭐ Recommandé |
| pytest-xdist -f | ✅ | Moyen | Active | ✅ Simple |
| VS Code Test | N/A | Rapide | Active | ⭐ IDE intégré |

---

## 🎯 Recommandation finale

**Pour ce projet** : Utiliser `watch-tests.ps1` pour compatibilité immédiate.

**Pour nouveaux projets** : Migrer vers `pytest-testmon` :
```powershell
pip install pytest-testmon
pytest --testmon
```

---

*Dernière mise à jour : 16/12/2025*
