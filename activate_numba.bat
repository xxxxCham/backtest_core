@echo off
REM ============================================================================
REM Configuration Numba/NumPy Threading (CMD/Batch)
REM ============================================================================
REM Usage: activate_numba.bat avant de lancer Python

echo === Configuration Numba Threading ===

REM Détecter nombre de cœurs (approximatif)
set NUMBA_NUM_THREADS=32
set NUMBA_THREADING_LAYER=tbb
set NUMBA_CACHE_DIR=%CD%\.numba_cache
set OMP_NUM_THREADS=32
set MKL_NUM_THREADS=32
set OPENBLAS_NUM_THREADS=32

echo NUMBA_NUM_THREADS=%NUMBA_NUM_THREADS%
echo NUMBA_THREADING_LAYER=%NUMBA_THREADING_LAYER%

REM Valider
python -c "import numba; print('Threading layer:', numba.config.THREADING_LAYER); print('Threads:', numba.config.NUMBA_NUM_THREADS)"

echo.
echo Variables activees pour cette session CMD
echo Pour permanence : ajouter au PATH systeme ou lancer ce .bat avant Python
