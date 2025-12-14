# ============================================================
# Backtest Core - Setup Environment Script
# ============================================================
# Usage: .\setup_env.ps1 [-GPU] [-Force]
# Options:
#   -GPU   : Installer les dépendances GPU (CuPy, etc.)
#   -Force : Recréer l'environnement même s'il existe
# ============================================================

param(
    [switch]$GPU,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$VenvPath = ".venv"
$PythonVersion = "3.12"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Backtest Core - Configuration Env      " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Python est installé
try {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
    $version = python --version 2>&1
    Write-Host "[OK] Python trouvé: $version" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Python non trouvé. Installez Python $PythonVersion+" -ForegroundColor Red
    exit 1
}

# Supprimer l'ancien environnement si -Force
if ($Force -and (Test-Path $VenvPath)) {
    Write-Host "[INFO] Suppression de l'ancien environnement..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvPath
}

# Créer l'environnement virtuel
if (-not (Test-Path $VenvPath)) {
    Write-Host "[INFO] Création de l'environnement virtuel..." -ForegroundColor Yellow
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] Échec de création de l'environnement" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Environnement créé: $VenvPath" -ForegroundColor Green
} else {
    Write-Host "[OK] Environnement existant: $VenvPath" -ForegroundColor Green
}

# Activer l'environnement
Write-Host "[INFO] Activation de l'environnement..." -ForegroundColor Yellow
$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
. $activateScript

# Mettre à jour pip
Write-Host "[INFO] Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Installer les dépendances
if ($GPU) {
    Write-Host "[INFO] Installation des dépendances GPU..." -ForegroundColor Yellow
    pip install -r requirements-gpu.txt --quiet
    Write-Host "[OK] Dépendances GPU installées" -ForegroundColor Green
} else {
    Write-Host "[INFO] Installation des dépendances CPU..." -ForegroundColor Yellow
    pip install -r requirements.txt --quiet
    Write-Host "[OK] Dépendances CPU installées" -ForegroundColor Green
}

# Vérification
Write-Host ""
Write-Host "[INFO] Vérification de l'installation..." -ForegroundColor Yellow
python -c "import numpy; import pandas; import streamlit; print('  numpy:', numpy.__version__); print('  pandas:', pandas.__version__); print('  streamlit:', streamlit.__version__)"

if ($GPU) {
    python -c "import cupy; print('  cupy:', cupy.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] CuPy GPU disponible" -ForegroundColor Green
    } else {
        Write-Host "[WARN] CuPy non disponible (GPU)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   Installation terminée avec succès!     " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour activer l'environnement:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Pour lancer l'application:" -ForegroundColor Cyan
Write-Host "  .\run.ps1" -ForegroundColor White
Write-Host "  # ou" -ForegroundColor Gray
Write-Host "  streamlit run ui/app.py" -ForegroundColor White
Write-Host ""
