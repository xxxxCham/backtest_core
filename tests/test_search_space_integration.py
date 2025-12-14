"""
Tests d'intégration pour compute_search_space_stats.

Vérifie que la fonction est correctement utilisée dans :
- CLI (cli/commands.py)
- UI (ui/app.py)
- SweepEngine (backtest/sweep.py)
- AutonomousStrategist (agents/autonomous_strategist.py)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parameters import compute_search_space_stats


class TestCLIIntegration:
    """Tests d'intégration avec le CLI."""
    
    def test_cli_uses_stats_in_sweep(self):
        """Vérifier que cmd_sweep utilise compute_search_space_stats."""
        from cli.commands import cmd_sweep
        import inspect
        
        # Vérifier que compute_search_space_stats est utilisé dans le code
        source = inspect.getsource(cmd_sweep)
        assert 'compute_search_space_stats' in source


class TestUIIntegration:
    """Tests d'intégration avec l'UI Streamlit."""
    
    def test_ui_imports_function(self):
        """Vérifier que l'UI importe compute_search_space_stats."""
        # Note: On ne peut pas importer ui/app.py directement car Streamlit
        # n'est pas disponible dans les tests, mais on peut vérifier le code source
        ui_app_path = Path(__file__).parent.parent / "ui" / "app.py"
        
        if ui_app_path.exists():
            content = ui_app_path.read_text(encoding="utf-8")
            assert 'compute_search_space_stats' in content
            assert 'SearchSpaceStats' in content
    
    def test_ui_uses_stats_in_grid_mode(self):
        """Vérifier que l'UI utilise les stats en mode grille."""
        ui_app_path = Path(__file__).parent.parent / "ui" / "app.py"
        
        if ui_app_path.exists():
            content = ui_app_path.read_text(encoding="utf-8")
            # Chercher l'utilisation dans le mode grille
            assert 'stats = compute_search_space_stats' in content or \
                   'stats=compute_search_space_stats' in content
    
    def test_ui_uses_stats_in_llm_mode(self):
        """Vérifier que l'UI utilise les stats en mode LLM."""
        ui_app_path = Path(__file__).parent.parent / "ui" / "app.py"
        
        if ui_app_path.exists():
            content = ui_app_path.read_text(encoding="utf-8")
            # Chercher l'utilisation dans le mode LLM
            assert 'llm_space_stats' in content or 'stats.summary()' in content


class TestSweepEngineIntegration:
    """Tests d'intégration avec SweepEngine."""
    
    def test_sweep_imports_function(self):
        """Vérifier que sweep.py importe compute_search_space_stats."""
        from backtest import sweep
        
        # Vérifier que la fonction est disponible ou importée
        import inspect
        source = inspect.getsource(sweep)
        assert 'compute_search_space_stats' in source
    
    def test_sweep_uses_stats_before_grid_generation(self):
        """Vérifier que SweepEngine utilise les stats avant de générer la grille."""
        import inspect
        from backtest.sweep import SweepEngine
        
        # Vérifier que run_sweep utilise compute_search_space_stats
        source = inspect.getsource(SweepEngine.run_sweep)
        assert 'compute_search_space_stats' in source


class TestAgentIntegration:
    """Tests d'intégration avec les agents LLM."""
    
    def test_autonomous_strategist_imports_function(self):
        """Vérifier que autonomous_strategist.py importe compute_search_space_stats."""
        from agents import autonomous_strategist
        
        # Vérifier que la fonction est importée
        import inspect
        source = inspect.getsource(autonomous_strategist)
        assert 'compute_search_space_stats' in source
        assert 'SearchSpaceStats' in source
    
    def test_autonomous_strategist_uses_stats_in_context(self):
        """Vérifier que AutonomousStrategist utilise les stats dans le contexte."""
        import inspect
        from agents.autonomous_strategist import AutonomousStrategist
        
        # Vérifier que _build_iteration_context utilise compute_search_space_stats
        source = inspect.getsource(AutonomousStrategist._build_iteration_context)
        assert 'compute_search_space_stats' in source


class TestConsistency:
    """Tests de cohérence entre les différentes utilisations."""
    
    def test_all_modules_use_same_function(self):
        """Vérifier que tous les modules utilisent la même fonction."""
        from utils.parameters import compute_search_space_stats as utils_func
        
        # Sweep (indirect via import)
        from backtest.sweep import compute_search_space_stats as sweep_func
        assert sweep_func is utils_func
    
    def test_search_space_stats_usage_patterns(self):
        """Vérifier les patterns d'utilisation sont cohérents."""
        # Toutes les utilisations devraient :
        # 1. Calculer les stats avec compute_search_space_stats()
        # 2. Utiliser les attributs stats (total_combinations, per_param_counts, etc.)
        # 3. Gérer les informations d'overflow ou warnings si nécessaire
        
        # CLI pattern
        cli_path = Path(__file__).parent.parent / "cli" / "commands.py"
        if cli_path.exists():
            content = cli_path.read_text(encoding="utf-8")
            # Vérifier l'utilisation des attributs stats
            assert 'stats.total_combinations' in content or 'stats.per_param_counts' in content
        
        # Sweep pattern
        sweep_path = Path(__file__).parent.parent / "backtest" / "sweep.py"
        if sweep_path.exists():
            content = sweep_path.read_text(encoding="utf-8")
            assert 'stats.total_combinations' in content or 'stats.per_param_counts' in content


class TestDocumentation:
    """Tests de documentation."""
    
    def test_function_has_docstring(self):
        """Vérifier que compute_search_space_stats a une docstring."""
        from utils.parameters import compute_search_space_stats
        
        assert compute_search_space_stats.__doc__ is not None
        assert len(compute_search_space_stats.__doc__) > 50
    
    def test_searchspacestats_has_docstring(self):
        """Vérifier que SearchSpaceStats a une docstring."""
        from utils.parameters import SearchSpaceStats
        
        assert SearchSpaceStats.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
