# Demarrer Ollama avec configuration multi-GPU
$env:CUDA_VISIBLE_DEVICES = "1,0"
$env:OLLAMA_NUM_GPU = "2"
$env:OLLAMA_GPU_OVERHEAD = "0"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_FLASH_ATTENTION = "1"

Write-Host "Demarrage Ollama multi-GPU (RTX 5080 + RTX 2060)..." -ForegroundColor Green
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Normal
Start-Sleep -Seconds 3
Write-Host "Ollama demarre!" -ForegroundColor Green
