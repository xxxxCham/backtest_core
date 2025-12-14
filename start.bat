@echo off
REM ============================================================
REM Backtest Core - Lanceur Application
REM ============================================================
REM Double-cliquez sur ce fichier pour lancer l'application
REM ============================================================

title Backtest Core - Streamlit

echo.
echo ============================================
echo         Backtest Core - Launcher
echo ============================================
echo.

cd /d "%~dp0"

REM Configuration des donnees
set BACKTEST_DATA_DIR=D:\ThreadX_big\data\crypto\processed\parquet

REM Verifier l'environnement virtuel
if not exist ".venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel non trouve!
    echo Executez d'abord: python -m venv .venv
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
echo [INFO] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Lancer Streamlit
echo [INFO] Lancement de Streamlit...
echo.
echo  URL: http://localhost:8501
echo.
echo  Appuyez sur Ctrl+C pour arreter
echo.

streamlit run ui/app.py --server.port 8501

pause
