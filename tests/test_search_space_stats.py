"""
Tests pour compute_search_space_stats() et gestion unifiée de l'espace de recherche.

Valide:
- Calcul correct du nombre de combinaisons
- Gestion des différents formats d'entrée
- Warnings et overflow detection
- Continuous vs discrete spaces
"""

import pytest
from unittest.mock import Mock

from utils.parameters import (
    compute_search_space_stats,
    SearchSpaceStats,
    ParameterSpec,
)


class TestSearchSpaceStats:
    """Tests du dataclass SearchSpaceStats."""
    
    def test_summary_continuous(self):
        """Summary pour espace continu."""
        stats = SearchSpaceStats(
            total_combinations=-1,
            per_param_counts={"fast": -1, "slow": -1},
            warnings=[],
            has_overflow=False,
            is_continuous=True,
        )
        assert "continu" in stats.summary().lower()
    
    def test_summary_discrete(self):
        """Summary pour espace discret."""
        stats = SearchSpaceStats(
            total_combinations=120,
            per_param_counts={"fast": 10, "slow": 12},
            warnings=[],
            has_overflow=False,
            is_continuous=False,
        )
        assert "120" in stats.summary()
    
    def test_to_dict(self):
        """Conversion en dictionnaire."""
        stats = SearchSpaceStats(
            total_combinations=100,
            per_param_counts={"fast": 10, "slow": 10},
            warnings=["test warning"],
            has_overflow=False,
            is_continuous=False,
        )
        d = stats.to_dict()
        assert d["total_combinations"] == 100
        assert d["per_param_counts"] == {"fast": 10, "slow": 10}
        assert len(d["warnings"]) == 1


class TestComputeSearchSpaceStats:
    """Tests de la fonction compute_search_space_stats()."""
    
    def test_with_parameter_spec_discrete(self):
        """Test avec ParameterSpec et step défini."""
        param_space = {
            "fast_period": ParameterSpec(
                name="fast_period",
                min_val=5,
                max_val=20,
                default=10,
                step=1,
                param_type="int"
            ),
            "slow_period": ParameterSpec(
                name="slow_period",
                min_val=20,
                max_val=50,
                default=30,
                step=5,
                param_type="int"
            ),
        }
        
        stats = compute_search_space_stats(param_space)
        
        # fast: (20-5)/1 + 1 = 16 valeurs
        # slow: (50-20)/5 + 1 = 7 valeurs
        # total: 16 * 7 = 112
        assert stats.total_combinations == 112
        assert stats.per_param_counts["fast_period"] == 16
        assert stats.per_param_counts["slow_period"] == 7
        assert not stats.is_continuous
        assert not stats.has_overflow
    
    def test_with_parameter_spec_continuous(self):
        """Test avec ParameterSpec sans step (continu)."""
        # Note: ParameterSpec génère automatiquement un step dans __post_init__
        # Pour un vrai test continu, utiliser tuples sans step
        param_space = {
            "threshold": (0.0, 1.0),  # Tuple sans step = continu
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.is_continuous
        assert stats.total_combinations == -1
        assert stats.per_param_counts["threshold"] == -1
        assert len(stats.warnings) > 0
        assert any("continu" in w.lower() for w in stats.warnings)
    
    def test_with_tuples_3_elements(self):
        """Test avec tuples (min, max, step)."""
        param_space = {
            "fast": (5, 20, 1),  # 16 valeurs
            "slow": (20, 50, 5),  # 7 valeurs
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.total_combinations == 112
        assert stats.per_param_counts["fast"] == 16
        assert stats.per_param_counts["slow"] == 7
        assert not stats.is_continuous
    
    def test_with_tuples_2_elements(self):
        """Test avec tuples (min, max) sans step."""
        param_space = {
            "fast": (5, 20),  # Continu
            "slow": (20, 50),  # Continu
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.is_continuous
        assert stats.total_combinations == -1
    
    def test_with_dict_format(self):
        """Test avec dict contenant min/max/step."""
        param_space = {
            "fast_period": {
                "min": 5,
                "max": 20,
                "step": 1,
            },
            "slow_period": {
                "min": 20,
                "max": 50,
                "step": 5,
            },
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.total_combinations == 112
        assert not stats.is_continuous
    
    def test_with_dict_count_override(self):
        """Test avec dict ayant un 'count' pré-calculé."""
        param_space = {
            "custom_param": {
                "min": 0,
                "max": 100,
                "count": 25,  # Override le calcul
            },
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.per_param_counts["custom_param"] == 25
        assert stats.total_combinations == 25
    
    def test_overflow_detection(self):
        """Test détection overflow."""
        param_space = {
            "p1": (0, 100, 1),  # 101 valeurs
            "p2": (0, 100, 1),  # 101 valeurs
            "p3": (0, 10, 1),   # 11 valeurs
        }
        
        # 101 * 101 * 11 = 112,211 > 100,000
        stats = compute_search_space_stats(param_space, max_combinations=100000)
        
        assert stats.has_overflow
        assert len(stats.warnings) > 0
        assert any("Limite dépassée" in w or "dépassée" in w for w in stats.warnings)
    
    def test_no_overflow_below_threshold(self):
        """Test pas d'overflow en dessous du seuil."""
        param_space = {
            "p1": (0, 10, 1),  # 11 valeurs
            "p2": (0, 10, 1),  # 11 valeurs
        }
        
        # 11 * 11 = 121 < 100,000
        stats = compute_search_space_stats(param_space, max_combinations=100000)
        
        assert not stats.has_overflow
        assert stats.total_combinations == 121
    
    def test_mixed_continuous_discrete(self):
        """Test mélange continu/discret."""
        param_space = {
            "discrete_param": (0, 10, 1),  # Discret
            "continuous_param": (0.0, 1.0),  # Continu (pas de step)
        }
        
        stats = compute_search_space_stats(param_space)
        
        # Si un paramètre est continu, tout devient continu
        assert stats.is_continuous
        assert stats.total_combinations == -1
        assert stats.per_param_counts["discrete_param"] == 11
        assert stats.per_param_counts["continuous_param"] == -1
    
    def test_with_granularity(self):
        """Test avec granularité spécifiée."""
        param_space = {
            "fast_period": ParameterSpec(
                name="fast_period",
                min_val=5,
                max_val=20,
                default=10,
                step=1,
                param_type="int"
            ),
        }
        
        # Avec granularité 0.8 (très grossier)
        stats = compute_search_space_stats(param_space, granularity=0.8)
        
        # La granularité réduit le nombre de valeurs
        assert stats.per_param_counts["fast_period"] < 16
        assert stats.per_param_counts["fast_period"] >= 1
    
    def test_empty_param_space(self):
        """Test avec espace de paramètres vide."""
        stats = compute_search_space_stats({})
        
        assert stats.total_combinations == 1  # Produit vide = 1
        assert stats.per_param_counts == {}
        assert not stats.is_continuous
    
    def test_single_parameter(self):
        """Test avec un seul paramètre."""
        param_space = {
            "period": (10, 50, 5),  # 9 valeurs
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.total_combinations == 9
        assert stats.per_param_counts["period"] == 9


class TestIntegrationWithExistingCode:
    """Tests d'intégration avec le code existant."""
    
    def test_compatible_with_generate_param_grid(self):
        """Vérifie compatibilité avec generate_param_grid."""
        from utils.parameters import generate_param_grid
        
        param_specs = {
            "fast": ParameterSpec("fast", 5, 15, 10, step=1, param_type="int"),
            "slow": ParameterSpec("slow", 20, 30, 25, step=1, param_type="int"),
        }
        
        # Calculer stats
        stats = compute_search_space_stats(param_specs, granularity=0.5)
        
        # Générer grille réelle
        grid = generate_param_grid(param_specs, granularity=0.5)
        
        # Le nombre doit correspondre
        assert len(grid) == stats.total_combinations
    
    def test_cli_usage_pattern(self):
        """Test du pattern utilisé dans CLI."""
        # Pattern CLI : dict avec min/max/step
        param_space = {
            "fast_period": {"min": 5, "max": 20, "step": 1},
            "slow_period": {"min": 20, "max": 50, "step": 5},
        }
        
        stats = compute_search_space_stats(param_space)
        
        assert stats.total_combinations == 112
        assert not stats.is_continuous
        
        # Vérifier que le summary est utilisable en CLI
        summary = stats.summary()
        assert "112" in summary or "combinaisons" in summary.lower()
    
    def test_ui_llm_usage_pattern(self):
        """Test du pattern utilisé dans UI mode LLM."""
        # Pattern UI LLM : tuples (min, max, step)
        param_space = {
            "fast_period": (5, 20, 1),
            "slow_period": (20, 50, 5),
        }
        
        stats = compute_search_space_stats(param_space)
        
        # Vérifier qu'on peut afficher dans l'UI
        assert stats.total_combinations > 0
        assert len(stats.per_param_counts) == 2
        
        # Vérifier format affichage
        assert stats.to_dict()["total_combinations"] == 112


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
