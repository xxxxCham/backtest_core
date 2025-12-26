#!/bin/bash
# ============================================================
# Backtest Core - Script d'Installation Automatique (Linux/macOS)
# ============================================================

set -e  # Arrêter en cas d'erreur

echo ""
echo "========================================"
echo " Backtest Core - Installation"
echo "========================================"
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "[ERREUR] Python 3 n'est pas installé"
    echo "Installez Python 3.10+ depuis: https://www.python.org/downloads/"
    exit 1
fi

echo "[OK] Python détecté:"
python3 --version

# Créer l'environnement virtuel
echo ""
echo "[ETAPE 1/3] Création de l'environnement virtuel..."
if [ -d ".venv" ]; then
    echo "[INFO] Environnement virtuel déjà existant, suppression..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "[OK] Environnement virtuel créé"

# Activer l'environnement virtuel
echo ""
echo "[ETAPE 2/3] Activation de l'environnement virtuel..."
source .venv/bin/activate
echo "[OK] Environnement virtuel activé"

# Installer les dépendances
echo ""
echo "[ETAPE 3/3] Installation des dépendances..."
echo "[INFO] Mise à jour de pip..."
python -m pip install --upgrade pip --quiet

echo "[INFO] Installation de requirements.txt..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo " Installation RÉUSSIE!"
echo "========================================"
echo ""
echo "Pour lancer l'interface:"
echo "  1. Activer l'environnement: source .venv/bin/activate"
echo "  2. Lancer Streamlit:        streamlit run ui/app.py"
echo ""
echo "Documentation complète: INSTALL.md"
echo ""

# Test rapide
echo "[TEST] Vérification des imports..."
python -c "import streamlit, pandas, numpy, plotly; print('[OK] Toutes les dépendances sont installées!')"

echo ""
