"""
Tests unitaires pour Phase 1 : Infrastructure LLM Grid Search.

Tests:
- RangeProposal création et validation
- normalize_param_ranges() avec différents cas
- AgentContext extension avec sweep_results
"""

import pytest
from utils.parameters import RangeProposal, normalize_param_ranges, ParameterSpec
from agents.base_agent import AgentContext


class TestRangeProposal:
    """Tests pour RangeProposal dataclass."""

    def test_create_range_proposal(self):
        """Test création basique d'un RangeProposal."""
        proposal = RangeProposal(
            ranges={
                "bb_period": {"min": 20, "max": 25, "step": 1},
                "bb_std": {"min": 2.0, "max": 2.5, "step": 0.1}
            },
            rationale="Test correlation bb_period vs bb_std",
            optimize_for="sharpe_ratio",
            max_combinations=50
        )

        assert proposal.ranges["bb_period"]["min"] == 20
        assert proposal.ranges["bb_std"]["max"] == 2.5
        assert proposal.rationale == "Test correlation bb_period vs bb_std"
        assert proposal.optimize_for == "sharpe_ratio"
        assert proposal.max_combinations == 50
        assert proposal.early_stop_threshold is None

    def test_range_proposal_defaults(self):
        """Test valeurs par défaut de RangeProposal."""
        proposal = RangeProposal(
            ranges={"test": {"min": 1, "max": 10, "step": 1}},
            rationale="Test defaults"
        )

        assert proposal.optimize_for == "sharpe_ratio"
        assert proposal.max_combinations == 100
        assert proposal.early_stop_threshold is None

    def test_range_proposal_with_early_stop(self):
        """Test RangeProposal avec early_stop_threshold."""
        proposal = RangeProposal(
            ranges={"test": {"min": 1, "max": 10, "step": 1}},
            rationale="Test early stop",
            early_stop_threshold=2.0
        )

        assert proposal.early_stop_threshold == 2.0


class TestNormalizeParamRanges:
    """Tests pour normalize_param_ranges()."""

    def test_basic_normalization(self):
        """Test normalisation basique."""
        specs = [
            ParameterSpec("bb_period", min_val=10, max_val=50, default=20, step=1, param_type="int"),
            ParameterSpec("bb_std", min_val=1.0, max_val=3.0, default=2.0, step=0.1, param_type="float")
        ]

        ranges = {
            "bb_period": {"min": 20, "max": 25, "step": 1},
            "bb_std": {"min": 2.0, "max": 2.5, "step": 0.1}
        }

        grid = normalize_param_ranges(specs, ranges)

        assert "bb_period" in grid
        assert "bb_std" in grid
        assert grid["bb_period"] == [20, 21, 22, 23, 24, 25]
        # Vérifier bb_std avec tolérance pour float
        assert len(grid["bb_std"]) == 6  # 2.0, 2.1, 2.2, 2.3, 2.4, 2.5
        assert abs(grid["bb_std"][0] - 2.0) < 1e-9
        assert abs(grid["bb_std"][-1] - 2.5) < 1e-9

    def test_clamping_to_spec_bounds(self):
        """Test clamping aux limites du ParameterSpec."""
        specs = [
            ParameterSpec("test", min_val=10, max_val=50, default=20, step=1, param_type="int")
        ]

        # Ranges hors limites
        ranges = {"test": {"min": 5, "max": 60, "step": 1}}

        grid = normalize_param_ranges(specs, ranges)

        # Devrait être clampé à [10, 50]
        assert min(grid["test"]) == 10
        assert max(grid["test"]) == 50

    def test_unknown_parameter_raises_error(self):
        """Test qu'un paramètre inconnu lève une ValueError."""
        specs = [
            ParameterSpec("bb_period", min_val=10, max_val=50, default=20, step=1)
        ]

        ranges = {"unknown_param": {"min": 10, "max": 20, "step": 1}}

        with pytest.raises(ValueError, match="Paramètre inconnu"):
            normalize_param_ranges(specs, ranges)

    def test_missing_min_or_max_raises_error(self):
        """Test que min/max manquants lèvent une ValueError."""
        specs = [
            ParameterSpec("test", min_val=10, max_val=50, default=20, step=1)
        ]

        # Manque 'max'
        ranges = {"test": {"min": 20, "step": 1}}

        with pytest.raises(ValueError, match="'min' et 'max' sont obligatoires"):
            normalize_param_ranges(specs, ranges)

    def test_step_defaults_to_spec_step(self):
        """Test que step par défaut provient du ParameterSpec."""
        specs = [
            ParameterSpec("test", min_val=10, max_val=15, default=12, step=2, param_type="int")
        ]

        # Pas de step fourni
        ranges = {"test": {"min": 10, "max": 14}}

        grid = normalize_param_ranges(specs, ranges)

        # Devrait utiliser step=2 du spec
        assert grid["test"] == [10, 12, 14]

    def test_negative_or_zero_step_raises_error(self):
        """Test que step <= 0 lève une ValueError."""
        specs = [
            ParameterSpec("test", min_val=10, max_val=50, default=20, step=1)
        ]

        ranges = {"test": {"min": 10, "max": 20, "step": 0}}

        with pytest.raises(ValueError, match="step doit être > 0"):
            normalize_param_ranges(specs, ranges)

    def test_min_greater_than_max_after_clamp_raises_error(self):
        """Test que min > max après clamping lève une ValueError."""
        specs = [
            ParameterSpec("test", min_val=30, max_val=50, default=40, step=1)
        ]

        # Range complètement hors limites (inférieure)
        ranges = {"test": {"min": 10, "max": 20, "step": 1}}

        # Après clamping: min=30 (clampé), max=20 (clampé) → min > max
        with pytest.raises(ValueError, match="min.*> max"):
            normalize_param_ranges(specs, ranges)

    def test_int_type_returns_integers(self):
        """Test que param_type='int' retourne des entiers."""
        specs = [
            ParameterSpec("test", min_val=10, max_val=15, default=12, step=1, param_type="int")
        ]

        ranges = {"test": {"min": 10, "max": 15, "step": 1}}

        grid = normalize_param_ranges(specs, ranges)

        for val in grid["test"]:
            assert isinstance(val, int)

    def test_float_type_returns_floats(self):
        """Test que param_type='float' retourne des floats."""
        specs = [
            ParameterSpec("test", min_val=1.0, max_val=2.0, default=1.5, step=0.5, param_type="float")
        ]

        ranges = {"test": {"min": 1.0, "max": 2.0, "step": 0.5}}

        grid = normalize_param_ranges(specs, ranges)

        for val in grid["test"]:
            assert isinstance(val, float)


class TestAgentContextSweepExtension:
    """Tests pour extension AgentContext avec sweep_results."""

    def test_agent_context_has_sweep_fields(self):
        """Test qu'AgentContext possède les nouveaux champs sweep."""
        ctx = AgentContext(strategy_name="test")

        assert hasattr(ctx, "sweep_results")
        assert hasattr(ctx, "sweep_summary")
        assert ctx.sweep_results is None
        assert ctx.sweep_summary == ""

    def test_agent_context_can_set_sweep_results(self):
        """Test qu'on peut assigner sweep_results et sweep_summary."""
        ctx = AgentContext(strategy_name="test")

        sweep_data = {
            "best_params": {"bb_period": 23, "bb_std": 2.2},
            "best_metrics": {"sharpe_ratio": 2.45, "total_return_pct": 8.3},
            "n_combinations": 48
        }

        summary = "Grid Search Results (48 combinations tested):\nTop config: Sharpe=2.45"

        ctx.sweep_results = sweep_data
        ctx.sweep_summary = summary

        assert ctx.sweep_results == sweep_data
        assert ctx.sweep_summary == summary
        assert ctx.sweep_results["n_combinations"] == 48
