@echo off
REM =========================================================
REM SCRIPT DE REDEMARRAGE STREAMLIT AVEC OPTIMISATIONS
REM =========================================================

echo.
echo ========================================
echo REDEMARRAGE STREAMLIT OPTIMISE
echo ========================================
echo.

REM 1. Arrêter tous les processus Python/Streamlit
echo [1/4] Arret des processus Python/Streamlit...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

REM 2. Nettoyer le cache Python
echo [2/4] Nettoyage cache Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

REM 3. Nettoyer le cache Streamlit
echo [3/4] Nettoyage cache Streamlit...
if exist "%USERPROFILE%\.streamlit\cache" rd /s /q "%USERPROFILE%\.streamlit\cache"

REM 4. Relancer Streamlit
echo [4/4] Demarrage Streamlit avec optimisations...
echo.
echo ========================================
echo STREAMLIT DEMARRE - Ouvrez:
echo http://localhost:8501
echo ========================================
echo.
python -m streamlit run ui\app.py --server.maxUploadSize 500
