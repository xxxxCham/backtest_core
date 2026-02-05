@echo off
REM Backtest Core - Lancement Streamlit avec variables d'environnement
REM Cree le 04/02/2026

echo ========================================
echo Demarrage Backtest Core Streamlit
echo ========================================
echo.

REM Chemin du projet
cd /d "%~dp0"
echo Repertoire de travail: %CD%
echo.

REM Definition des variables d'environnement
set BACKTEST_DATA_DIR=D:\my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet
echo ✅ Repertoire de donnees: %BACKTEST_DATA_DIR%
echo.

REM Verification de l'environnement virtuel
if exist ".venv\Scripts\python.exe" (
    echo ✅ Environnement virtuel trouve: .venv
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    echo ❌ ERREUR: Environnement virtuel non trouve
    echo    Executez: python -m venv .venv
    pause
    exit /b 1
)

REM Verification de Streamlit
%PYTHON_EXE% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ ERREUR: Streamlit non installe
    echo    Executez: pip install streamlit
    pause
    exit /b 1
)

echo ✅ Streamlit installe et pret
echo.
echo 🚀 Lancement de Streamlit...
echo 📍 URL: http://localhost:8501
echo.
echo 🛑 Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.

REM Lancement de Streamlit avec variables d'environnement
%PYTHON_EXE% -m streamlit run ui\app.py

if errorlevel 1 (
    echo.
    echo ❌ ERREUR: Streamlit a rencontre une erreur
    pause
)
