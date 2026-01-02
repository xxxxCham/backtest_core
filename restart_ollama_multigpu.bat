@echo off
REM Script pour redémarrer Ollama avec support multi-GPU

echo ========================================
echo REDÉMARRAGE OLLAMA MULTI-GPU
echo ========================================

REM Arrêter Ollama actuel
echo.
echo 1. Arrêt d'Ollama...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul

REM Définir les variables d'environnement pour multi-GPU
echo.
echo 2. Configuration variables d'environnement...
set CUDA_VISIBLE_DEVICES=0,1
set OLLAMA_NUM_GPU=2
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_FLASH_ATTENTION=1

echo    CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo    OLLAMA_NUM_GPU=%OLLAMA_NUM_GPU%
echo    OLLAMA_GPU_OVERHEAD=%OLLAMA_GPU_OVERHEAD%
echo    OLLAMA_MAX_LOADED_MODELS=%OLLAMA_MAX_LOADED_MODELS%

REM Démarrer Ollama avec ces variables
echo.
echo 3. Démarrage Ollama multi-GPU...
start "Ollama Multi-GPU" ollama serve

echo.
echo ========================================
echo ✅ Ollama redémarré en mode multi-GPU
echo ========================================
echo.
echo 💡 Vérifiez l'utilisation GPU avec:
echo    nvidia-smi
echo.
echo 💡 Testez avec:
echo    ollama run llama3.3-70b-2gpu "Hello, test multi-GPU"
echo.
pause
