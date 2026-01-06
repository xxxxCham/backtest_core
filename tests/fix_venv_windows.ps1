#!/usr/bin/env pwsh
# Script de réparation automatique de l'environnement virtuel Windows
# Résout le problème de .venv créé sous WSL avec chemins Unix

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "REPARATION ENVIRONNEMENT VIRTUEL WINDOWS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Étape 1 : Vérifier Python Windows disponible
Write-Host "[1/7] Verification Python Windows..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR: Python non trouve dans PATH Windows" -ForegroundColor Red
    Write-Host "Installez Python depuis https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "      Python detecte: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Étape 2 : Désactiver environnement virtuel actuel si activé
Write-Host "[2/7] Desactivation environnement virtuel actuel..." -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "      Environnement actif detecte: $env:VIRTUAL_ENV" -ForegroundColor Gray
    deactivate 2>$null
    $env:VIRTUAL_ENV = $null
    $env:PATH = $env:PATH -replace "[^;]*\.venv[^;]*;?", ""
    Write-Host "      Desactive avec succes" -ForegroundColor Green
} else {
    Write-Host "      Aucun environnement actif" -ForegroundColor Gray
}
Write-Host ""

# Étape 3 : Supprimer ancien .venv corrompu
Write-Host "[3/7] Suppression ancien .venv corrompu..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "      Ancien .venv detecte - Suppression en cours..." -ForegroundColor Gray
    Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    # Vérification suppression
    if (Test-Path ".venv") {
        Write-Host "      AVERTISSEMENT: Suppression partielle - Tentative forcee..." -ForegroundColor Yellow
        Get-ChildItem -Path ".venv" -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
        Remove-Item -Path ".venv" -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path ".venv")) {
        Write-Host "      Ancien .venv supprime avec succes" -ForegroundColor Green
    } else {
        Write-Host "      ERREUR: Impossible de supprimer .venv completement" -ForegroundColor Red
        Write-Host "      Fermez tous les programmes utilisant .venv et reessayez" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "      Aucun ancien .venv detecte" -ForegroundColor Gray
}
Write-Host ""

# Étape 4 : Créer nouveau .venv Windows natif
Write-Host "[4/7] Creation nouvel environnement virtuel Windows..." -ForegroundColor Yellow
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERREUR: Impossible de creer .venv" -ForegroundColor Red
    exit 1
}

# Vérifier création
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "      Nouvel environnement cree avec succes" -ForegroundColor Green
    $venvPython = & ".venv\Scripts\python.exe" --version
    Write-Host "      Version Python dans .venv: $venvPython" -ForegroundColor Green
} else {
    Write-Host "      ERREUR: .venv cree mais python.exe manquant" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Étape 5 : Activer nouvel environnement
Write-Host "[5/7] Activation nouvel environnement..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERREUR: Impossible d'activer .venv" -ForegroundColor Red
    Write-Host "      Essayez: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "      Environnement active: $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host ""

# Étape 6 : Mise à jour pip et installation dépendances
Write-Host "[6/7] Installation des dependances..." -ForegroundColor Yellow
Write-Host "      Mise a jour pip..." -ForegroundColor Gray
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "      AVERTISSEMENT: Mise a jour pip echouee (non critique)" -ForegroundColor Yellow
}

Write-Host "      Installation requirements.txt (base - peut prendre plusieurs minutes)..." -ForegroundColor Gray
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERREUR: Installation des dependances de base echouee" -ForegroundColor Red
    Write-Host "      Verifiez requirements.txt et votre connexion internet" -ForegroundColor Red
    exit 1
}
Write-Host "      Dependances de base installees avec succes" -ForegroundColor Green
Write-Host ""

Write-Host "      Installation requirements-performance.txt (optimisations)..." -ForegroundColor Gray
pip install -r requirements-performance.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "      AVERTISSEMENT: Installation packages performance echouee (non critique)" -ForegroundColor Yellow
} else {
    Write-Host "      Packages performance installes avec succes" -ForegroundColor Green
}
Write-Host ""

Write-Host "      Installation requirements-gpu.txt (acceleration GPU)..." -ForegroundColor Gray
pip install -r requirements-gpu.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "      AVERTISSEMENT: Installation packages GPU echouee (non critique)" -ForegroundColor Yellow
    Write-Host "      Note: Necessaire uniquement si CUDA/GPU NVIDIA disponible" -ForegroundColor Gray
} else {
    Write-Host "      Packages GPU installes avec succes" -ForegroundColor Green
}
Write-Host ""

# Étape 7 : Vérification installation
Write-Host "[7/7] Verification installation..." -ForegroundColor Yellow

$modules = @(
    @{Name="streamlit"; Import="import streamlit; print(streamlit.__version__)"},
    @{Name="pandas"; Import="import pandas; print(pandas.__version__)"},
    @{Name="numpy"; Import="import numpy; print(numpy.__version__)"},
    @{Name="cython"; Import="import cython; print(cython.__version__)"},
    @{Name="cupy"; Import="import cupy as cp; print(cp.__version__ + ' - GPUs: ' + str(cp.cuda.runtime.getDeviceCount()))"},
    @{Name="ui.app"; Import="from ui.app import main; print('OK')"}
)

$allOk = $true
foreach ($module in $modules) {
    $result = python -c $module.Import 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      [$($module.Name)] OK - Version: $result" -ForegroundColor Green
    } else {
        Write-Host "      [$($module.Name)] ERREUR: $result" -ForegroundColor Red
        $allOk = $false
    }
}
Write-Host ""

# Résumé final
Write-Host "========================================" -ForegroundColor Cyan
if ($allOk) {
    Write-Host "REPARATION TERMINEE AVEC SUCCES !" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Prochaines etapes:" -ForegroundColor Yellow
    Write-Host "  1. L'environnement virtuel est deja active" -ForegroundColor White
    Write-Host "  2. Lancez l'application avec:" -ForegroundColor White
    Write-Host "     .\run_streamlit.bat" -ForegroundColor Cyan
    Write-Host "     OU" -ForegroundColor Gray
    Write-Host "     streamlit run ui\app.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Pour les prochaines sessions:" -ForegroundColor Yellow
    Write-Host "  - Activez .venv avec: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "REPARATION TERMINEE AVEC ERREURS" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Certains modules ne se sont pas installes correctement." -ForegroundColor Yellow
    Write-Host "Verifiez les messages d'erreur ci-dessus." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
