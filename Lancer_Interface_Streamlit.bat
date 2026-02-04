@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ========================================
echo Demarrage interface Streamlit
echo ========================================
echo.
echo Repertoire: %CD%
echo.

REM Variables d'environnement
set "BACKTEST_DATA_DIR=D:\my_soft\gestionnaire_telechargement_multi-timeframe\processed\parquet"
set "BACKTEST_WORKER_THREADS=1"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
)

if exist ".venv\Scripts\activate.bat" (
    echo Activation de l'environnement virtuel...
    call ".venv\Scripts\activate.bat"
    if errorlevel 1 (
        echo AVERTISSEMENT: Impossible d'activer l'environnement virtuel.
        echo Utilisation directe de l'executable Python de .venv...
    ) else (
        echo Environnement virtuel active: %VIRTUAL_ENV%
    )
) else (
    echo AVERTISSEMENT: Environnement virtuel .venv non trouve.
)

if "%PYTHON_EXE%"=="" (
    set "PYTHON_EXE=python"
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERREUR: Python introuvable.
        echo Installez Python ou creez un venv: python -m venv .venv
        pause
        exit /b 1
    )
)

echo.
echo Verification de Streamlit...
"%PYTHON_EXE%" -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Streamlit n'est pas installe.
    echo Executez: "%PYTHON_EXE%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

set "PORT=8501"
set "MAX_PORT=8510"

echo.
echo Recherche d'un port disponible...

:PORT_LOOP
if %PORT% GTR %MAX_PORT% (
    echo ERREUR: Aucun port disponible entre 8501 et %MAX_PORT%.
    pause
    exit /b 1
)

netstat -ano | findstr ":%PORT% " | findstr /I "LISTENING ECOUTE" >nul 2>&1
if errorlevel 1 (
    goto :PORT_READY
) else (
    echo   Port %PORT% occupe, essai du suivant...
    set /a PORT=%PORT%+1
    goto :PORT_LOOP
)

:PORT_READY
echo.
echo Port disponible: %PORT%
echo Lancement de Streamlit (ui\app.py)...
echo URL: http://localhost:%PORT%
echo.
echo Appuyez sur Ctrl+C pour arreter l'application
echo ========================================
echo.

REM Ouvrir le navigateur apres 1 seconde
timeout /t 1 /nobreak >nul 2>&1 & start http://localhost:%PORT%

echo Demarrage de Streamlit...
echo.
"%PYTHON_EXE%" -m streamlit run ui\app.py --server.port=%PORT% --browser.gatherUsageStats=false
set STREAMLIT_EXIT_CODE=%ERRORLEVEL%

echo.
if %STREAMLIT_EXIT_CODE% NEQ 0 (
    echo ERREUR: Streamlit s'est arrete avec le code %STREAMLIT_EXIT_CODE%
    echo.
    echo Appuyez sur une touche pour fermer cette fenetre...
    pause >nul
) else (
    echo Streamlit arrete normalement.
)
pause
