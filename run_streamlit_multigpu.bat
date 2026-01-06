@echo off
REM Script de lancement unifié Ollama Multi-GPU + Streamlit
REM Lance Ollama avec configuration multi-GPU puis démarre l'UI Streamlit

echo ========================================
echo Demarrage Ollama Multi-GPU + Streamlit
echo ========================================
echo.

REM 1. Demarrer Ollama multi-GPU
echo [1/2] Configuration Ollama multi-GPU (RTX 5080 + RTX 2060)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-OllamaMultiGPU.ps1"

if errorlevel 1 (
    echo.
    echo ERREUR: Ollama n'a pas demarre correctement
    pause
    exit /b 1
)

echo.
echo [2/2] Lancement Streamlit...
timeout /t 3 /nobreak >nul

REM 2. Activer venv et lancer Streamlit
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM Définir variables d'environnement
set PYTHONPATH=%~dp0
set MODELS_JSON_PATH=D:\models\models.json
set CUDA_VISIBLE_DEVICES=0,1

REM Lancer Streamlit dans une nouvelle fenêtre
start "Backtest Core Streamlit" cmd /c "streamlit run ui\app.py"

REM 3. Ouvrir navigateur automatiquement après démarrage Streamlit
echo.
echo Attente demarrage Streamlit (http://localhost:8501)...
powershell -NoProfile -Command "$url='http://localhost:8501'; for($i=0; $i -lt 60; $i++){ try{ $tcp = New-Object Net.Sockets.TcpClient; $tcp.Connect('127.0.0.1',8501); $tcp.Close(); Start-Process $url; break } catch{ Start-Sleep -Seconds 1 } }" >nul 2>&1

echo.
echo ========================================
echo Application demarree!
echo ========================================
echo.
echo Ollama API:    http://127.0.0.1:11434
echo Streamlit UI:  http://localhost:8501
echo.
echo GPU Config:    RTX 5080 (GPU 1) + RTX 2060 (GPU 0)
echo.
echo Fermez cette fenetre pour arreter uniquement ce script
echo (Ollama et Streamlit continueront en arriere-plan)
echo ========================================
pause
