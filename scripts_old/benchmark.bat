@echo off
REM ============================================
REM Benchmark System - Backtest Core
REM Teste les performances CPU/RAM/Numba/Joblib
REM ============================================

echo.
echo ================================================
echo     BENCHMARK SYSTEME BACKTEST CORE
echo ================================================
echo.

REM Activer l'environnement virtuel
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERREUR: Environnement virtuel non trouve!
    echo Executez d'abord: python -m venv .venv
    pause
    exit /b 1
)

REM Charger les variables d'environnement
if exist ".env" (
    echo [OK] Chargement .env...
    for /f "tokens=*" %%a in (.env) do (
        set "%%a" 2>nul
    )
)

REM Lancer le benchmark
echo.
echo Lancement du benchmark...
echo.

python -m tools.benchmark_system %*

echo.
echo ================================================
echo Benchmark termine!
echo ================================================
pause
