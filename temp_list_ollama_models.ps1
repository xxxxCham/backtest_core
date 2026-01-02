# Script temporaire pour lister tous les modèles Ollama installés
$manifestPath = "D:\models\ollama\manifests\registry.ollama.ai\library"
$models = Get-ChildItem -Path $manifestPath -Directory

Write-Host "=== MODELES OLLAMA INSTALLES ===" -ForegroundColor Cyan
Write-Host ""

$count = 0
foreach ($model in $models) {
    $tags = Get-ChildItem -Path $model.FullName -File
    foreach ($tag in $tags) {
        $count++
        Write-Host "$($model.Name):$($tag.Name)"
    }
}

Write-Host ""
Write-Host "Total: $count modèles" -ForegroundColor Green
