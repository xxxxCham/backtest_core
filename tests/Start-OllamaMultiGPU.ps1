# Start-OllamaMultiGPU.ps1
# Script PowerShell pour démarrer Ollama en mode multi-GPU
# EXÉCUTER EN TANT QU'ADMINISTRATEUR

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DÉMARRAGE OLLAMA MULTI-GPU" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Étape 1: Arrêter Ollama existant
Write-Host "1. Arrêt d'Ollama..." -ForegroundColor Yellow
$ollamaProcesses = Get-Process ollama -ErrorAction SilentlyContinue

if ($ollamaProcesses) {
    Write-Host "   Trouvé $($ollamaProcesses.Count) processus Ollama" -ForegroundColor Yellow
    foreach ($proc in $ollamaProcesses) {
        try {
            Stop-Process -Id $proc.Id -Force
            Write-Host "   ✓ Processus $($proc.Id) arrêté" -ForegroundColor Green
        }
        catch {
            Write-Host "   ✗ Impossible d'arrêter processus $($proc.Id): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    Start-Sleep -Seconds 3
} else {
    Write-Host "   Aucun processus Ollama trouvé" -ForegroundColor Gray
}

# Étape 2: Définir les variables d'environnement
Write-Host ""
Write-Host "2. Configuration variables d'environnement..." -ForegroundColor Yellow

# PRIORITÉ GPU:
#   GPU 0 (RTX 5080) = Primaire (plus performante, ~16 GB VRAM)
#   GPU 1 (RTX 2060 SUPER) = Secondaire (8 GB VRAM)
#   GPU 2 (AMD iGPU) = Ignorée (pas CUDA, non visible par nvidia-smi)
$env:CUDA_VISIBLE_DEVICES = "0,1"  # RTX 5080 en premier, RTX 2060 en second
$env:OLLAMA_NUM_GPU = "2"
$env:OLLAMA_GPU_OVERHEAD = "0"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_FLASH_ATTENTION = "1"

Write-Host "   CUDA_VISIBLE_DEVICES  = $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Cyan
Write-Host "   OLLAMA_NUM_GPU        = $env:OLLAMA_NUM_GPU" -ForegroundColor Cyan
Write-Host "   OLLAMA_GPU_OVERHEAD   = $env:OLLAMA_GPU_OVERHEAD" -ForegroundColor Cyan
Write-Host "   OLLAMA_MAX_LOADED_MODELS = $env:OLLAMA_MAX_LOADED_MODELS" -ForegroundColor Cyan
Write-Host "   OLLAMA_FLASH_ATTENTION = $env:OLLAMA_FLASH_ATTENTION" -ForegroundColor Cyan

# Étape 3: Démarrer Ollama
Write-Host ""
Write-Host "3. Démarrage Ollama multi-GPU..." -ForegroundColor Yellow

try {
    # Démarrer dans une nouvelle fenêtre pour voir les logs
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Normal

    Write-Host "   ✓ Ollama démarré" -ForegroundColor Green
    Start-Sleep -Seconds 2

} catch {
    Write-Host "   ✗ Erreur démarrage: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Étape 4: Vérifier le démarrage
Write-Host ""
Write-Host "4. Vérification..." -ForegroundColor Yellow

$attempt = 0
$maxAttempts = 10

while ($attempt -lt $maxAttempts) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction Stop
        Write-Host "   ✓ Ollama répond sur http://127.0.0.1:11434" -ForegroundColor Green
        break
    }
    catch {
        if ($attempt -eq $maxAttempts) {
            Write-Host "   ✗ Ollama ne répond pas après $maxAttempts tentatives" -ForegroundColor Red
            exit 1
        }
        Write-Host "   Tentative $attempt/$maxAttempts..." -ForegroundColor Gray
        Start-Sleep -Seconds 1
    }
}

# Étape 5: Afficher les GPUs
Write-Host ""
Write-Host "5. État GPU:" -ForegroundColor Yellow

try {
    $gpuInfo = & nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    foreach ($line in $gpuInfo) {
        Write-Host "   GPU: $line" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ✗ nvidia-smi non disponible" -ForegroundColor Red
}

# Résumé
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ CONFIGURATION TERMINÉE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Pour tester le multi-GPU:" -ForegroundColor Yellow
Write-Host "   Terminal 1: nvidia-smi -l 1" -ForegroundColor White
Write-Host "   Terminal 2: ollama run llama3.3-70b-2gpu 'Test multi-GPU'" -ForegroundColor White
Write-Host ""
Write-Host "📊 Vérifiez que GPU 0 ET GPU 1 sont utilisés pendant l'inférence" -ForegroundColor Yellow
Write-Host ""

# Pause pour voir les résultats
Read-Host "Appuyez sur Entrée pour fermer..."
