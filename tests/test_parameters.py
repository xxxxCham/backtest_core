"""
Tests pour le système de paramètres et de granularité.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parameters import (
    EMA_CROSS_PRESET,
    MINIMAL_PRESET,
    SAFE_RANGES_PRESET,
    ParameterSpec,
    Preset,
    calculate_combinations,
    generate_param_grid,
    parameter_values,
)


class TestParameterValues:
    """Tests pour la fonction parameter_values."""

    def test_granularity_zero_returns_max_four_values(self):
        """Granularité 0 devrait retourner max 4 valeurs."""
        values = parameter_values(10, 50, granularity=0.0)
        assert len(values) <= 4
        assert values[0] == 10  # min
        assert values[-1] == 50  # max

    def test_granularity_one_returns_median(self):
        """Granularité 1 devrait retourner uniquement la médiane."""
        values = parameter_values(10, 50, granularity=1.0)
        assert len(values) == 1
        assert values[0] == 30  # médiane de 10-50

    def test_granularity_half_returns_some_values(self):
        """Granularité 0.5 devrait retourner quelques valeurs."""
        values = parameter_values(10, 50, granularity=0.5)
        assert 1 <= len(values) <= 4

    def test_float_values(self):
        """Test avec valeurs flottantes."""
        values = parameter_values(1.0, 3.0, granularity=0.0)
        assert len(values) <= 4
        assert values[0] == 1.0
        assert values[-1] == 3.0

    def test_small_range(self):
        """Test avec une plage petite."""
        values = parameter_values(1, 3, granularity=0.0)
        assert len(values) >= 1
        assert all(1 <= v <= 3 for v in values)

    def test_sorted_output(self):
        """Les valeurs devraient être triées."""
        values = parameter_values(0, 100, granularity=0.0)
        # Comparer avec numpy
        assert np.all(values[:-1] <= values[1:])

    def test_unique_values(self):
        """Les valeurs devraient être uniques."""
        values = parameter_values(10, 50, granularity=0.0)
        assert len(values) == len(np.unique(values))


class TestParameterSpec:
    """Tests pour la classe ParameterSpec."""

    def test_creation_int(self):
        """Test création d'un ParameterSpec entier."""
        spec = ParameterSpec(
            name="bb_period",
            min_val=10,
            max_val=50,
            default=20,
            param_type="int",
            description="Période des bandes de Bollinger"
        )
        assert spec.name == "bb_period"
        assert spec.min_val == 10
        assert spec.max_val == 50
        assert spec.default == 20
        assert spec.param_type == "int"

    def test_creation_float(self):
        """Test création d'un ParameterSpec flottant."""
        spec = ParameterSpec(
            name="bb_std",
            min_val=1.5,
            max_val=3.0,
            default=2.0,
            param_type="float",
            description="Écart-type"
        )
        assert spec.param_type == "float"
        assert spec.min_val == 1.5
        assert spec.default == 2.0

    def test_step_auto_calculation(self):
        """Le step devrait être calculé automatiquement."""
        spec = ParameterSpec(
            name="test",
            min_val=0,
            max_val=100,
            default=50
        )
        assert spec.step is not None
        assert spec.step > 0

    def test_to_dict(self):
        """Test conversion en dictionnaire."""
        spec = ParameterSpec(
            name="test",
            min_val=0,
            max_val=100,
            default=50,
            param_type="int"
        )
        d = spec.to_dict()
        assert d["name"] == "test"
        assert d["min"] == 0
        assert d["max"] == 100
        assert d["default"] == 50


class TestPreset:
    """Tests pour la classe Preset."""

    def test_preset_creation(self):
        """Test création d'un preset."""
        spec = ParameterSpec(
            name="leverage",
            min_val=1,
            max_val=10,
            default=3,
            param_type="int"
        )
        preset = Preset(
            name="test_preset",
            description="Preset de test",
            parameters={"leverage": spec}
        )
        assert preset.name == "test_preset"
        assert "leverage" in preset.parameters

    def test_safe_ranges_preset_exists(self):
        """SAFE_RANGES_PRESET devrait exister et être valide."""
        assert SAFE_RANGES_PRESET is not None
        assert SAFE_RANGES_PRESET.name == "Safe Ranges"
        assert "bb_period" in SAFE_RANGES_PRESET.parameters

    def test_minimal_preset_exists(self):
        """MINIMAL_PRESET devrait exister."""
        assert MINIMAL_PRESET is not None
        assert MINIMAL_PRESET.name == "Minimal"

    def test_ema_cross_preset_exists(self):
        """EMA_CROSS_PRESET devrait exister."""
        assert EMA_CROSS_PRESET is not None
        assert "fast_period" in EMA_CROSS_PRESET.parameters

    def test_preset_get_default_values(self):
        """Test récupération des valeurs par défaut."""
        defaults = SAFE_RANGES_PRESET.get_default_values()
        assert isinstance(defaults, dict)
        assert "bb_period" in defaults

    def test_preset_estimate_combinations(self):
        """Test estimation du nombre de combinaisons."""
        combos = SAFE_RANGES_PRESET.estimate_combinations(granularity=0.5)
        assert isinstance(combos, int)
        assert combos > 0


class TestCalculateCombinations:
    """Tests pour calculate_combinations."""

    def test_single_param(self):
        """Test avec un seul paramètre."""
        specs = {
            "p1": ParameterSpec("p1", 10, 50, 20, param_type="int")
        }
        combos, values_dict = calculate_combinations(specs, granularity=0.0)
        assert combos <= 4  # max 4 valeurs par param
        assert "p1" in values_dict

    def test_multiple_params(self):
        """Test avec plusieurs paramètres."""
        specs = {
            "p1": ParameterSpec("p1", 10, 50, 20, param_type="int"),
            "p2": ParameterSpec("p2", 1.0, 3.0, 2.0, param_type="float")
        }
        combos, values_dict = calculate_combinations(specs, granularity=0.0)
        # 4 * 4 = 16 max
        assert combos <= 16
        assert len(values_dict) == 2

    def test_high_granularity_fewer_combos(self):
        """Granularité élevée = moins de combinaisons."""
        specs = {
            "p1": ParameterSpec("p1", 10, 50, 20, param_type="int"),
            "p2": ParameterSpec("p2", 1.0, 3.0, 2.0, param_type="float")
        }
        combos_low, _ = calculate_combinations(specs, granularity=0.0)
        combos_high, _ = calculate_combinations(specs, granularity=1.0)

        assert combos_high < combos_low
        assert combos_high == 1  # 1 * 1 = 1 (médianes)


class TestGenerateParamGrid:
    """Tests pour generate_param_grid."""

    def test_generates_list_of_dicts(self):
        """Devrait générer une liste de dictionnaires."""
        specs = {
            "p1": ParameterSpec("p1", 10, 20, 15, param_type="int")
        }
        grid = generate_param_grid(specs, granularity=0.0)

        assert isinstance(grid, list)
        assert all(isinstance(item, dict) for item in grid)

    def test_all_params_present(self):
        """Chaque combinaison devrait avoir tous les paramètres."""
        specs = {
            "p1": ParameterSpec("p1", 10, 20, 15, param_type="int"),
            "p2": ParameterSpec("p2", 1.0, 2.0, 1.5, param_type="float")
        }
        grid = generate_param_grid(specs, granularity=0.0)

        for combo in grid:
            assert "p1" in combo
            assert "p2" in combo

    def test_values_in_range(self):
        """Toutes les valeurs devraient être dans les plages."""
        specs = {
            "p1": ParameterSpec("p1", 10, 50, 20, param_type="int")
        }
        grid = generate_param_grid(specs, granularity=0.0)

        for combo in grid:
            assert 10 <= combo["p1"] <= 50

    def test_int_type_preserved(self):
        """Le type int devrait être préservé."""
        specs = {
            "p1": ParameterSpec("p1", 10, 50, 20, param_type="int")
        }
        grid = generate_param_grid(specs, granularity=0.0)

        for combo in grid:
            # Les valeurs peuvent être int ou np.int64
            assert isinstance(combo["p1"], (int, np.integer))

    def test_empty_specs_returns_single_empty_dict(self):
        """Specs vides = liste avec dict vide."""
        grid = generate_param_grid({}, granularity=0.0)
        assert grid == [{}]

    def test_max_combinations_limit(self):
        """Devrait lever une erreur si trop de combinaisons."""
        # Créer plein de paramètres
        specs = {
            f"p{i}": ParameterSpec(f"p{i}", 1, 100, 50, param_type="int")
            for i in range(10)  # 10 params * 4 values = 4^10 > 1M
        }

        with pytest.raises(ValueError, match="Trop de combinaisons"):
            generate_param_grid(specs, granularity=0.0, max_total_combinations=1000)


class TestIntegrationWithSafeRanges:
    """Tests d'intégration avec le preset SAFE_RANGES."""

    def test_safe_ranges_combinations_reasonable(self):
        """SAFE_RANGES ne devrait pas dépasser un nombre raisonnable de combinaisons."""
        # Avec granularité 0.5, devrait être raisonnable
        combos = SAFE_RANGES_PRESET.estimate_combinations(granularity=0.5)
        assert combos < 10000  # Limite raisonnable

    def test_safe_ranges_has_required_params(self):
        """SAFE_RANGES devrait avoir les paramètres essentiels."""
        required_params = ["bb_period", "bb_std", "atr_period"]
        for param in required_params:
            assert param in SAFE_RANGES_PRESET.parameters

    def test_can_generate_grid_from_preset(self):
        """Devrait pouvoir générer une grille depuis le preset."""
        grid = generate_param_grid(
            SAFE_RANGES_PRESET.parameters,
            granularity=0.8,  # Granularité élevée pour limiter
            max_total_combinations=500
        )

        assert len(grid) > 0
        assert all("bb_period" in combo for combo in grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

