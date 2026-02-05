@echo off
REM Script de diagnostic pour tester l'environnement

echo ========================================
echo Diagnostic de l'environnement
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo 1. Repertoire de travail:
echo    %CD%
echo.

echo 2. Test Python systeme:
python --version 2>nul
if errorlevel 1 (
    echo    ERREUR: Python non trouve dans PATH
) else (
    echo    OK
)
echo.

echo 3. Test environnement virtuel:
if exist ".venv\Scripts\python.exe" (
    echo    .venv existe: OUI
    .venv\Scripts\python.exe --version
) else (
    echo    .venv existe: NON
    echo    ERREUR: Environnement virtuel non trouve
)
echo.

echo 4. Test activation venv:
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo    Environnement active: %VIRTUAL_ENV%
    echo.

    echo 5. Test modules Python installes:
    python -c "import streamlit; print('   Streamlit:', streamlit.__version__)" 2>nul
    if errorlevel 1 (
        echo    ERREUR: Streamlit non installe
    )

    python -c "import pandas; print('   Pandas:', pandas.__version__)" 2>nul
    if errorlevel 1 (
        echo    ERREUR: Pandas non installe
    )

    python -c "import numpy; print('   NumPy:', numpy.__version__)" 2>nul
    if errorlevel 1 (
        echo    ERREUR: NumPy non installe
    )

    echo.
    echo 6. Test import application:
    python -c "from ui.app import main; print('   Import ui.app: OK')" 2>nul
    if errorlevel 1 (
        echo    ERREUR: Impossible d'importer ui.app
        echo    Tentative d'afficher l'erreur:
        python -c "from ui.app import main" 2>&1
    )
) else (
    echo    ERREUR: activate.bat non trouve
)

echo.
echo ========================================
echo Diagnostic termine
echo ========================================
pause
