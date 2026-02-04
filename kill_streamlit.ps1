# Script pour tuer tous les processus Streamlit
Write-Host "Nettoyage des ports 8501-8510..." -ForegroundColor Yellow

# Tuer les processus par PID
$processIds = @(34496, 29960, 35124, 6940, 14856, 10140, 19820, 24592, 25424)

foreach ($processId in $processIds) {
    try {
        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Arret du processus $processId..." -ForegroundColor Yellow
            Stop-Process -Id $processId -Force
            Write-Host "OK - Processus $processId arrete" -ForegroundColor Green
        }
    } catch {
        Write-Host "Processus $processId deja ferme" -ForegroundColor Gray
    }
}

Write-Host "Nettoyage termine!" -ForegroundColor Green
