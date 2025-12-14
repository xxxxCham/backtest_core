# Backtest Core - Configuration Rapide
# =====================================
# Script PowerShell pour basculer facilement entre configurations

param(
    [Parameter(Position=0)]
    [ValidateSet('cpu', 'gpu', 'openai', 'debug', 'prod', 'reset')]
    [string]$Config = 'cpu'
)

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Backtest Core - Configuration Rapide" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

switch ($Config) {
    'cpu' {
        Write-Host "Configuration: CPU-only (défaut)" -ForegroundColor Green
        Write-Host ""
        
        $env:UNLOAD_LLM_DURING_BACKTEST = 'False'
        $env:BACKTEST_LLM_PROVIDER = 'ollama'
        $env:BACKTEST_LLM_MODEL = 'deepseek-r1:8b'
        $env:OLLAMA_HOST = 'http://localhost:11434'
        $env:BACKTEST_LOG_LEVEL = 'INFO'
        $env:USE_GPU = 'false'
        
        Write-Host "✅ UNLOAD_LLM_DURING_BACKTEST = False (pas de latence)" -ForegroundColor Green
        Write-Host "✅ BACKTEST_LLM_MODEL = deepseek-r1:8b (léger)" -ForegroundColor Green
        Write-Host "✅ USE_GPU = false" -ForegroundColor Green
    }
    
    'gpu' {
        Write-Host "Configuration: GPU Optimisé" -ForegroundColor Yellow
        Write-Host ""
        
        $env:UNLOAD_LLM_DURING_BACKTEST = 'True'
        $env:BACKTEST_LLM_PROVIDER = 'ollama'
        $env:BACKTEST_LLM_MODEL = 'deepseek-r1:32b'
        $env:OLLAMA_HOST = 'http://localhost:11434'
        $env:BACKTEST_LOG_LEVEL = 'INFO'
        $env:USE_GPU = 'true'
        
        Write-Host "✅ UNLOAD_LLM_DURING_BACKTEST = True (libère VRAM)" -ForegroundColor Yellow
        Write-Host "✅ BACKTEST_LLM_MODEL = deepseek-r1:32b (lourd)" -ForegroundColor Yellow
        Write-Host "✅ USE_GPU = true" -ForegroundColor Yellow
    }
    
    'openai' {
        Write-Host "Configuration: OpenAI Cloud" -ForegroundColor Magenta
        Write-Host ""
        
        $env:UNLOAD_LLM_DURING_BACKTEST = 'False'
        $env:BACKTEST_LLM_PROVIDER = 'openai'
        $env:BACKTEST_LLM_MODEL = 'gpt-4'
        $env:BACKTEST_LOG_LEVEL = 'WARNING'
        $env:USE_GPU = 'false'
        
        # Vérifier clé API
        if (-not $env:OPENAI_API_KEY) {
            Write-Host "⚠️  OPENAI_API_KEY non définie!" -ForegroundColor Red
            Write-Host "   Définir avec: `$env:OPENAI_API_KEY = 'sk-...'" -ForegroundColor Red
        }
        
        Write-Host "✅ BACKTEST_LLM_PROVIDER = openai" -ForegroundColor Magenta
        Write-Host "✅ BACKTEST_LLM_MODEL = gpt-4" -ForegroundColor Magenta
    }
    
    'debug' {
        Write-Host "Configuration: Debug Complet" -ForegroundColor Blue
        Write-Host ""
        
        $env:BACKTEST_LOG_LEVEL = 'DEBUG'
        $env:WALK_FORWARD_WINDOWS = '10'
        $env:WALK_FORWARD_MIN_TEST_SAMPLES = '100'
        $env:MAX_OVERFITTING_RATIO = '1.3'
        
        Write-Host "✅ BACKTEST_LOG_LEVEL = DEBUG (verbeux)" -ForegroundColor Blue
        Write-Host "✅ WALK_FORWARD_WINDOWS = 10 (strict)" -ForegroundColor Blue
        Write-Host "✅ MAX_OVERFITTING_RATIO = 1.3 (strict)" -ForegroundColor Blue
    }
    
    'prod' {
        Write-Host "Configuration: Production" -ForegroundColor DarkGreen
        Write-Host ""
        
        $env:BACKTEST_LOG_LEVEL = 'WARNING'
        $env:UNLOAD_LLM_DURING_BACKTEST = 'False'
        $env:USE_GPU = 'true'
        $env:MAX_WORKERS = '16'
        
        Write-Host "✅ BACKTEST_LOG_LEVEL = WARNING (minimal)" -ForegroundColor DarkGreen
        Write-Host "✅ MAX_WORKERS = 16 (parallélisme max)" -ForegroundColor DarkGreen
    }
    
    'reset' {
        Write-Host "Configuration: Reset (défaut système)" -ForegroundColor Gray
        Write-Host ""
        
        Remove-Item Env:UNLOAD_LLM_DURING_BACKTEST -ErrorAction SilentlyContinue
        Remove-Item Env:BACKTEST_LLM_PROVIDER -ErrorAction SilentlyContinue
        Remove-Item Env:BACKTEST_LLM_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:BACKTEST_LOG_LEVEL -ErrorAction SilentlyContinue
        Remove-Item Env:USE_GPU -ErrorAction SilentlyContinue
        
        Write-Host "✅ Toutes les variables effacées" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan

# Afficher variables critiques
Write-Host ""
Write-Host "Variables d'environnement actuelles:" -ForegroundColor White
Write-Host "  UNLOAD_LLM_DURING_BACKTEST = $env:UNLOAD_LLM_DURING_BACKTEST"
Write-Host "  BACKTEST_LLM_PROVIDER = $env:BACKTEST_LLM_PROVIDER"
Write-Host "  BACKTEST_LLM_MODEL = $env:BACKTEST_LLM_MODEL"
Write-Host "  BACKTEST_LOG_LEVEL = $env:BACKTEST_LOG_LEVEL"
Write-Host "  USE_GPU = $env:USE_GPU"

Write-Host ""
Write-Host "💡 Usage:" -ForegroundColor White
Write-Host "   .\set_config.ps1 cpu      # Configuration CPU-only" -ForegroundColor Gray
Write-Host "   .\set_config.ps1 gpu      # Configuration GPU optimisé" -ForegroundColor Gray
Write-Host "   .\set_config.ps1 openai   # Configuration OpenAI" -ForegroundColor Gray
Write-Host "   .\set_config.ps1 debug    # Mode debug verbeux" -ForegroundColor Gray
Write-Host "   .\set_config.ps1 prod     # Mode production" -ForegroundColor Gray
Write-Host "   .\set_config.ps1 reset    # Reset toutes les variables" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 Documentation: ENVIRONMENT.md" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Cyan
