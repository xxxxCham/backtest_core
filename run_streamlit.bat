@echo off
REM Launch Streamlit UI and open the browser.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Demarrage Backtest Core Streamlit
echo ========================================
echo.
echo Repertoire de travail: %CD%
echo.

REM Activate venv if present.
if exist ".venv\Scripts\activate.bat" (
    echo Activation de l'environnement virtuel...
    call ".venv\Scripts\activate.bat"
    if errorlevel 1 (
        echo ERREUR: Impossible d'activer l'environnement virtuel
        echo Verifiez que .venv existe et est correctement installe
        pause
        exit /b 1
    )
    echo Environnement virtuel active: %VIRTUAL_ENV%
) else (
    echo AVERTISSEMENT: Environnement virtuel .venv non trouve
    echo Tentative de lancement avec Python systeme...
)

echo.
echo Verification de Streamlit...
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Streamlit n'est pas installe
    echo Executez: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Lancement de Streamlit (ui\app.py)...
echo URL: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arreter l'application
echo ========================================
echo.

REM Start Streamlit (pas de nouvelle fenetre pour voir les erreurs)
python -m streamlit run ui\app.py

REM Si on arrive ici, Streamlit s'est arrete
echo.
echo Streamlit arrete.
pause
