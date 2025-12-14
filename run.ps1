# ============================================================
# Backtest Core - Run Application
# ============================================================
# Usage: .\run.ps1 [Options]
# Options:
#   -Tests    : Lancer les tests au lieu de l'application
#   -Sweep    : Lancer un sweep de paramètres
#   -Demo     : Lancer la démo de validation
#   -Port     : Port pour Streamlit (défaut: 8501)
# ============================================================

param(
    [switch]$Tests,
    [switch]$Sweep,
    [switch]$Demo,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$VenvPath = ".venv"

# Configuration des données
$env:BACKTEST_DATA_DIR = "D:\ThreadX_big\data\crypto\processed\parquet"

# Couleurs
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERREUR] $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "        Backtest Core - Launcher          " -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si l'environnement existe
if (-not (Test-Path $VenvPath)) {
    Write-Warn "Environnement non trouvé. Création..."
    Write-Host ""
    
    # Créer l'environnement
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Impossible de créer l'environnement"
        exit 1
    }
    
    # Activer et installer
    . "$VenvPath\Scripts\Activate.ps1"
    Write-Info "Installation des dépendances..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    Write-Ok "Environnement configuré"
    Write-Host ""
} else {
    # Activer l'environnement existant
    . "$VenvPath\Scripts\Activate.ps1"
    Write-Ok "Environnement activé: $VenvPath"
}

# Afficher la version Python
$pyVersion = python --version 2>&1
Write-Info "Python: $pyVersion"

# Exécuter selon l'option
if ($Tests) {
    Write-Host ""
    Write-Info "Lancement des tests..."
    Write-Host "--------------------------------------------" -ForegroundColor Gray
    python -m pytest tests/ -v --tb=short
    $exitCode = $LASTEXITCODE
    Write-Host "--------------------------------------------" -ForegroundColor Gray
    if ($exitCode -eq 0) {
        Write-Ok "Tous les tests passent!"
    } else {
        Write-Err "Certains tests ont échoué"
    }
}
elseif ($Sweep) {
    Write-Host ""
    Write-Info "Lancement du sweep de paramètres..."
    Write-Host "--------------------------------------------" -ForegroundColor Gray
    python -c @"
from backtest.sweep import SweepEngine, quick_sweep
from strategies.bollinger_atr import BollingerATRStrategy
from data.loader import load_ohlcv
import sys

try:
    df = load_ohlcv('BTCUSDT', '1h')
    print(f'Données chargées: {len(df)} barres')
    
    results = quick_sweep(
        df=df,
        strategy='bollinger_atr',
        param_grid={
            'bb_period': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'entry_z': [1.5, 2.0],
        },
        max_workers=8
    )
    print(results.summary())
except Exception as e:
    print(f'Erreur: {e}', file=sys.stderr)
    sys.exit(1)
"@
    Write-Host "--------------------------------------------" -ForegroundColor Gray
}
elseif ($Demo) {
    Write-Host ""
    Write-Info "Lancement de la démo de validation..."
    Write-Host "--------------------------------------------" -ForegroundColor Gray
    python validate_backtest.py
    Write-Host "--------------------------------------------" -ForegroundColor Gray
}
else {
    # Lancement de l'application Streamlit
    Write-Host ""
    Write-Info "Démarrage de l'application Streamlit..."
    Write-Info "URL: http://localhost:$Port"
    Write-Host ""
    Write-Host "Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Gray
    Write-Host "--------------------------------------------" -ForegroundColor Gray
    
    streamlit run ui/app.py --server.port $Port --server.headless true
}

Write-Host ""
