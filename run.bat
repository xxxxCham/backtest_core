@echo off
:: ============================================================
:: Backtest Core - Lanceur Windows (Double-clic)
:: ============================================================
:: Ce script active l'environnement et lance l'application
:: ============================================================

title Backtest Core - Loading...
cd /d "%~dp0"

echo.
echo ============================================
echo        Backtest Core - Demarrage
echo ============================================
echo.

:: Vérifier si l'environnement existe
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERREUR] Impossible de creer l'environnement
        pause
        exit /b 1
    )
    
    echo [INFO] Activation de l'environnement...
    call .venv\Scripts\activate.bat
    
    echo [INFO] Installation des dependances...
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    echo.
    echo [OK] Environnement configure!
) else (
    echo [OK] Environnement trouve
    call .venv\Scripts\activate.bat
)

echo.
echo [INFO] Demarrage de l'application Streamlit...
echo [INFO] URL: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arreter
echo --------------------------------------------

streamlit run ui/app.py --server.port 8501

pause
