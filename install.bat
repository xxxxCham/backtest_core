@echo off
REM ============================================================
REM Backtest Core - Script d'Installation Automatique (Windows)
REM ============================================================

echo.
echo ========================================
echo  Backtest Core - Installation
echo ========================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo Telechargez Python depuis: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detecte:
python --version

REM Créer l'environnement virtuel
echo.
echo [ETAPE 1/3] Creation de l'environnement virtuel...
if exist .venv (
    echo [INFO] Environnement virtuel deja existant, suppression...
    rmdir /s /q .venv
)
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERREUR] Impossible de creer l'environnement virtuel
    pause
    exit /b 1
)
echo [OK] Environnement virtuel cree

REM Activer l'environnement virtuel
echo.
echo [ETAPE 2/3] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)
echo [OK] Environnement virtuel active

REM Installer les dépendances
echo.
echo [ETAPE 3/3] Installation des dependances...
echo [INFO] Mise a jour de pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Installation de requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERREUR] Installation des dependances echouee
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Installation REUSSIE!
echo ========================================
echo.
echo Pour lancer l'interface:
echo   1. Activer l'environnement: .venv\Scripts\activate
echo   2. Lancer Streamlit:        streamlit run ui\app.py
echo.
echo Documentation complete: INSTALL.md
echo.

REM Test rapide
echo [TEST] Verification des imports...
python -c "import streamlit, pandas, numpy, plotly; print('[OK] Toutes les dependances sont installees!')"
if %errorlevel% neq 0 (
    echo [ATTENTION] Certains imports ont echoue, verifiez requirements.txt
)

echo.
pause
