#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script de profiling complet pour Backtest Core

.DESCRIPTION
    Exécute un profiling avec données synthétiques, génère un rapport HTML
    et ouvre automatiquement le résultat dans le navigateur.

.PARAMETER Mode
    Mode de profiling : 'demo', 'simple', ou 'grid'

.PARAMETER Open
    Ouvrir automatiquement le rapport HTML

.EXAMPLE
    .\tools\profile.ps1
    Exécute le profiling demo complet

.EXAMPLE
    .\tools\profile.ps1 -Open
    Exécute et ouvre le rapport HTML automatiquement
#>

[CmdletBinding()]
param(
    [ValidateSet('demo', 'simple', 'grid')]
    [string]$Mode = 'demo',

    [switch]$Open
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Couleurs
function Write-Header { param([string]$Text) Write-Host "`n$('=' * 60)" -ForegroundColor Cyan ; Write-Host $Text -ForegroundColor Yellow ; Write-Host "$('=' * 60)`n" -ForegroundColor Cyan }
function Write-Step { param([string]$Text) Write-Host "→ $Text" -ForegroundColor Green }
function Write-Error { param([string]$Text) Write-Host "✗ ERREUR: $Text" -ForegroundColor Red }
function Write-Success { param([string]$Text) Write-Host "✓ $Text" -ForegroundColor Green }

Write-Header "🔍 PROFILING - Backtest Core"

# Étape 1 : Profiling
Write-Step "Exécution du profiling..."
try {
    python tools/profile_demo.py
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Le profiling a échoué (code $LASTEXITCODE)"
        exit 1
    }
    Write-Success "Profiling terminé avec succès"
} catch {
    Write-Error "Erreur lors du profiling : $_"
    exit 1
}

# Étape 2 : Génération rapport HTML
Write-Step "Génération des rapports HTML..."
$reports = Get-ChildItem profiling_results\demo_*_$(Get-Date -Format 'yyyyMMdd')_*.prof -ErrorAction SilentlyContinue

if ($reports.Count -eq 0) {
    # Fallback: prendre les plus récents
    $reports = Get-ChildItem profiling_results\demo_*.prof | Sort-Object LastWriteTime -Descending | Select-Object -First 2
}

foreach ($report in $reports) {
    $outputName = $report.BaseName -replace '_\d{8}_\d{6}$', ''
    $outputHtml = "${outputName}_analysis.html"

    Write-Step "Analyse de $($report.Name)..."
    try {
        python tools/profile_analyzer.py --report $report.FullName --output $outputHtml
        Write-Success "Rapport généré : $outputHtml"

        if ($Open) {
            Start-Process $outputHtml
        }
    } catch {
        Write-Error "Erreur lors de la génération du rapport : $_"
    }
}

# Résumé
Write-Header "📊 RÉSUMÉ"
Write-Host "Fichiers de profiling générés :"
Get-ChildItem profiling_results\demo_*.prof | Sort-Object LastWriteTime -Descending | Select-Object -First 2 | ForEach-Object {
    Write-Host "  • $($_.Name) ($([math]::Round($_.Length/1KB, 2)) KB)"
}

Write-Host "`nRapports HTML :"
Get-ChildItem demo_*_analysis.html -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  • $($_.Name)"
}

Write-Host "`n🎯 Prochaines étapes :" -ForegroundColor Magenta
Write-Host "  1. Ouvrir les rapports HTML pour voir les bottlenecks"
Write-Host "  2. Identifier les fonctions en ROUGE (>10% du temps)"
Write-Host "  3. Optimiser : vectorisation, cache, pré-calcul"
Write-Host "  4. Re-profiler pour mesurer les gains"

Write-Host "`n📖 Documentation :" -ForegroundColor Cyan
Write-Host "  • Guide complet : docs/PROFILING_GUIDE.md"
Write-Host "  • Vue d'ensemble : PROFILING_SYSTEM.md"
Write-Host ""
