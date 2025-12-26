# 📘 Guide GitHub - Travailler depuis N'importe Quel Ordinateur

Guide complet pour utiliser GitHub et travailler sur votre projet depuis n'importe quel ordinateur.

---

## 🎯 Vue d'ensemble

**GitHub** vous permet de :
- ✅ Sauvegarder votre code en ligne (cloud)
- ✅ Cloner votre projet sur n'importe quel ordinateur
- ✅ Synchroniser vos modifications entre plusieurs machines
- ✅ Revenir en arrière en cas d'erreur
- ✅ Collaborer avec d'autres développeurs

---

## 📋 Checklist de Préparation

### ✅ Sur Votre Ordinateur Principal (Avant de Partir)

1. **Vérifier que Git est configuré**
   ```bash
   git config --global user.name "Votre Nom"
   git config --global user.email "votre.email@example.com"
   ```

2. **Créer un repository GitHub** (si pas déjà fait)
   - Aller sur https://github.com
   - Cliquer "New repository"
   - Nom: `backtest_core`
   - Visibilité: Privé (recommandé) ou Public
   - NE PAS initialiser avec README (déjà existant)

3. **Lier votre projet local à GitHub**
   ```bash
   cd d:\backtest_core

   # Initialiser Git (si pas déjà fait)
   git init

   # Lier au repository distant
   git remote add origin https://github.com/VOTRE_USERNAME/backtest_core.git

   # Vérifier la liaison
   git remote -v
   ```

4. **Commiter et pousser TOUT votre code**
   ```bash
   # Vérifier les fichiers modifiés
   git status

   # Ajouter tous les fichiers
   git add .

   # Créer un commit
   git commit -m "Initial commit - Backtest Core complet avec V2/V3"

   # Pousser vers GitHub (première fois)
   git branch -M main
   git push -u origin main
   ```

5. **Vérifier sur GitHub**
   - Aller sur https://github.com/VOTRE_USERNAME/backtest_core
   - Tous vos fichiers doivent être visibles
   - ✅ README.md doit s'afficher en page d'accueil

---

## 💻 Sur l'Autre Ordinateur (Installation)

### Étape 1: Installer Git

**Windows:**
- Télécharger depuis https://git-scm.com/download/win
- Installer avec les options par défaut

**Linux:**
```bash
sudo apt update
sudo apt install git
```

**macOS:**
```bash
brew install git
```

### Étape 2: Cloner le Projet

```bash
# Naviguer dans le dossier où vous voulez le projet
cd ~/Documents  # ou C:\Users\VotreNom\Documents sur Windows

# Cloner depuis GitHub
git clone https://github.com/VOTRE_USERNAME/backtest_core.git

# Entrer dans le dossier
cd backtest_core
```

### Étape 3: Installation Automatique

**Windows:**
```bash
install.bat
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

### Étape 4: Vérification

```bash
# Activer l'environnement
source .venv/bin/activate  # Linux/macOS
# OU
.venv\Scripts\activate     # Windows

# Lancer l'interface
streamlit run ui/app.py
```

Si l'interface s'ouvre, **c'est parfait !** 🎉

---

## 🔄 Workflow de Développement

### Avant de Commencer à Travailler

Toujours récupérer les dernières modifications :

```bash
cd backtest_core
git pull origin main
```

### Pendant le Travail

Commitez régulièrement (toutes les 30 min ou après chaque fonctionnalité) :

```bash
# Vérifier ce qui a changé
git status

# Voir les différences
git diff

# Ajouter les fichiers modifiés
git add .

# Créer un commit avec message descriptif
git commit -m "Ajout paramètre bb_std_v4 pour stratégie V4"

# Pousser vers GitHub
git push origin main
```

### À la Fin de la Session

**TOUJOURS** pousser vos modifications :

```bash
git add .
git commit -m "Fin de session - Optimisation V3 terminée"
git push origin main
```

---

## 🛡️ Bonnes Pratiques Git

### Messages de Commit Clairs

❌ **Mauvais exemples:**
```bash
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

✅ **Bons exemples:**
```bash
git commit -m "Ajout stratégie Bollinger ATR V4 avec trailing stop"
git commit -m "Fix bug calcul Sharpe Ratio pour périodes courtes"
git commit -m "Optimisation performance grid search (4x plus rapide)"
git commit -m "UI: Ajout graphique comparaison stratégies"
```

### Fréquence des Commits

- **Trop peu**: Risque de perdre du travail
- **Trop souvent**: Historique illisible

**Recommandé**:
- Après chaque fonctionnalité terminée
- Après correction d'un bug
- Avant de tester une modification risquée
- Fin de session de travail

### Ne JAMAIS Commiter

❌ Fichiers à ne PAS commiter (déjà dans .gitignore) :
- `.venv/` (environnement virtuel)
- `__pycache__/` (cache Python)
- `*.pyc` (bytecode compilé)
- `.env` (secrets/clés API)
- `data/sample_data/*.csv` (données volumineuses)
- `orchestration_logs_*.json` (logs temporaires)

---

## 🔧 Commandes Git Essentielles

### Vérifier l'état

```bash
# Voir les fichiers modifiés
git status

# Voir les différences
git diff

# Voir l'historique
git log --oneline --graph --all
```

### Annuler des Modifications

```bash
# Annuler modifications NON commitées (fichier spécifique)
git checkout -- fichier.py

# Annuler TOUTES les modifications NON commitées
git reset --hard HEAD

# Revenir au commit précédent (⚠️ DANGER: perte des commits récents)
git reset --hard HEAD~1

# Annuler le DERNIER commit (garder les modifications)
git reset --soft HEAD~1
```

### Branches (Avancé)

```bash
# Créer une branche pour tester
git checkout -b test-nouvelle-feature

# Revenir à main
git checkout main

# Fusionner la branche test dans main
git merge test-nouvelle-feature

# Supprimer la branche
git branch -d test-nouvelle-feature
```

---

## 🚨 Résolution de Problèmes

### Conflit lors de `git pull`

```bash
# Erreur: "Your local changes would be overwritten"
# Solution: Stasher vos modifications temporairement
git stash
git pull origin main
git stash pop

# Résoudre les conflits manuellement si nécessaire
# Puis:
git add .
git commit -m "Résolution conflits après pull"
```

### Mot de Passe GitHub Demandé à Chaque Fois

**Solution 1: HTTPS avec Token**
```bash
# Créer un Personal Access Token sur GitHub
# Settings → Developer settings → Personal access tokens → Generate new token
# Utiliser le token comme mot de passe

# Sauvegarder le token (Windows)
git config --global credential.helper wincred

# Sauvegarder le token (Linux/macOS)
git config --global credential.helper store
```

**Solution 2: SSH** (recommandé)
```bash
# Générer clé SSH
ssh-keygen -t ed25519 -C "votre.email@example.com"

# Copier la clé publique
cat ~/.ssh/id_ed25519.pub  # Linux/macOS
type %USERPROFILE%\.ssh\id_ed25519.pub  # Windows

# Ajouter sur GitHub: Settings → SSH and GPG keys → New SSH key

# Changer l'URL du remote
git remote set-url origin git@github.com:VOTRE_USERNAME/backtest_core.git
```

### Repository Trop Volumineux

```bash
# Vérifier la taille
git count-objects -vH

# Supprimer les gros fichiers de l'historique (⚠️ AVANCÉ)
# Utiliser BFG Repo-Cleaner ou git-filter-branch
# Voir: https://docs.github.com/en/repositories/working-with-files/managing-large-files
```

---

## 📊 Cas d'Usage Concrets

### Scénario 1: Travail sur 2 Ordinateurs

**Ordinateur Personnel (Jour 1 - Soir):**
```bash
cd backtest_core
# ... travail sur stratégie V4 ...
git add .
git commit -m "WIP: Stratégie V4 - structure de base"
git push origin main
```

**Ordinateur Ami (Jour 2 - Matin):**
```bash
cd backtest_core
git pull origin main  # Récupère le travail d'hier
# ... continuer le travail ...
git add .
git commit -m "Stratégie V4 - tests unitaires ajoutés"
git push origin main
```

**Ordinateur Personnel (Jour 2 - Soir):**
```bash
cd backtest_core
git pull origin main  # Récupère le travail de ce matin
# ... finaliser ...
git add .
git commit -m "Stratégie V4 - finalisée et testée"
git push origin main
```

### Scénario 2: Tester une Idée Risquée

```bash
# Créer une branche de test
git checkout -b test-nouvelle-logique

# Faire vos modifications
# ... code ...

# Tester
streamlit run ui/app.py

# Si ça marche:
git checkout main
git merge test-nouvelle-logique

# Si ça ne marche pas:
git checkout main
git branch -D test-nouvelle-logique  # Supprimer la branche
```

---

## ✅ Checklist Avant de Quitter l'Ordinateur

- [ ] `git status` - Vérifier qu'il n'y a pas de modifications oubliées
- [ ] `git add .` - Ajouter toutes les modifications
- [ ] `git commit -m "Message clair"` - Commiter avec un bon message
- [ ] `git push origin main` - Pousser vers GitHub
- [ ] Vérifier sur https://github.com/VOTRE_USERNAME/backtest_core que tout est bien là

---

## 🎓 Ressources Complémentaires

- **GitHub Docs**: https://docs.github.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Interactive Git Tutorial**: https://learngitbranching.js.org/

---

## 💡 Résumé en 5 Points

1. **AVANT de partir**: `git add . && git commit -m "Message" && git push`
2. **SUR l'autre PC**: `git clone https://github.com/...`
3. **AVANT de travailler**: `git pull`
4. **PENDANT le travail**: Commits réguliers
5. **FIN de session**: `git push`

---

**🚀 Vous êtes prêt à coder partout !**
