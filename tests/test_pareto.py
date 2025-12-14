"""
Tests pour le module Pareto Pruning.
"""

import pytest
import numpy as np

from backtest.pareto import (
    ParetoPoint,
    ParetoFrontier,
    ParetoPruner,
    MultiObjectiveOptimizer,
    pareto_optimize,
)


class TestParetoPoint:
    """Tests pour ParetoPoint."""
    
    def test_creation(self):
        """Test création d'un point."""
        point = ParetoPoint(
            params={"fast": 10, "slow": 20},
            objectives={"sharpe_ratio": 1.5, "max_drawdown": 0.15},
        )
        
        assert point.params == {"fast": 10, "slow": 20}
        assert point.objectives["sharpe_ratio"] == 1.5
        assert point.objectives["max_drawdown"] == 0.15
    
    def test_dominates_clearly_better(self):
        """Test domination claire (meilleur sur tout)."""
        point_a = ParetoPoint(
            params={},
            objectives={"sharpe": 2.0, "return": 0.3},
        )
        point_b = ParetoPoint(
            params={},
            objectives={"sharpe": 1.0, "return": 0.2},
        )
        
        directions = {"sharpe": 1, "return": 1}  # Maximiser les deux
        
        assert point_a.dominates(point_b, directions)
        assert not point_b.dominates(point_a, directions)
    
    def test_dominates_with_mixed_directions(self):
        """Test domination avec directions mixtes."""
        # A: meilleur sharpe (max) et meilleur drawdown (min)
        point_a = ParetoPoint(
            params={},
            objectives={"sharpe": 2.0, "max_drawdown": 0.10},
        )
        # B: pire sur les deux
        point_b = ParetoPoint(
            params={},
            objectives={"sharpe": 1.5, "max_drawdown": 0.20},
        )
        
        directions = {"sharpe": 1, "max_drawdown": -1}
        
        assert point_a.dominates(point_b, directions)
        assert not point_b.dominates(point_a, directions)
    
    def test_no_dominance_tradeoff(self):
        """Test non-domination (trade-off)."""
        # A: meilleur sharpe, pire drawdown
        point_a = ParetoPoint(
            params={},
            objectives={"sharpe": 2.0, "max_drawdown": 0.25},
        )
        # B: pire sharpe, meilleur drawdown
        point_b = ParetoPoint(
            params={},
            objectives={"sharpe": 1.5, "max_drawdown": 0.10},
        )
        
        directions = {"sharpe": 1, "max_drawdown": -1}
        
        assert not point_a.dominates(point_b, directions)
        assert not point_b.dominates(point_a, directions)
    
    def test_to_dict(self):
        """Test conversion en dict."""
        point = ParetoPoint(
            params={"x": 1},
            objectives={"y": 2.0},
            metadata={"info": "test"},
        )
        
        d = point.to_dict()
        assert d["params"] == {"x": 1}
        assert d["objectives"] == {"y": 2.0}
        assert d["metadata"] == {"info": "test"}


class TestParetoFrontier:
    """Tests pour ParetoFrontier."""
    
    def test_empty_frontier(self):
        """Test frontière vide."""
        frontier = ParetoFrontier()
        assert frontier.size() == 0
        assert frontier.get_best("sharpe_ratio") is None
    
    def test_add_first_point(self):
        """Test ajout du premier point."""
        frontier = ParetoFrontier()
        point = ParetoPoint(
            params={"x": 1},
            objectives={"sharpe_ratio": 1.5},
        )
        
        added = frontier.add_point(point)
        
        assert added is True
        assert frontier.size() == 1
    
    def test_add_dominated_point_rejected(self):
        """Test rejet de point dominé."""
        frontier = ParetoFrontier()
        
        # Ajouter un bon point
        good = ParetoPoint(
            params={"x": 1},
            objectives={"sharpe_ratio": 2.0, "total_return": 0.3},
        )
        frontier.add_point(good)
        
        # Tenter d'ajouter un point dominé
        bad = ParetoPoint(
            params={"x": 2},
            objectives={"sharpe_ratio": 1.0, "total_return": 0.1},
        )
        added = frontier.add_point(bad)
        
        assert added is False
        assert frontier.size() == 1
    
    def test_add_dominating_point_removes_old(self):
        """Test ajout point dominant supprime les anciens."""
        frontier = ParetoFrontier()
        
        # Ajouter un point moyen
        medium = ParetoPoint(
            params={"x": 1},
            objectives={"sharpe_ratio": 1.5, "total_return": 0.2},
        )
        frontier.add_point(medium)
        
        # Ajouter un point meilleur
        better = ParetoPoint(
            params={"x": 2},
            objectives={"sharpe_ratio": 2.0, "total_return": 0.3},
        )
        frontier.add_point(better)
        
        assert frontier.size() == 1
        assert frontier.points[0].params["x"] == 2
    
    def test_pareto_optimal_points_kept(self):
        """Test que les points Pareto-optimaux sont conservés."""
        frontier = ParetoFrontier()
        
        # Points non-dominés (trade-off)
        point1 = ParetoPoint(
            params={"x": 1},
            objectives={"sharpe_ratio": 2.0, "max_drawdown": 0.20},
        )
        point2 = ParetoPoint(
            params={"x": 2},
            objectives={"sharpe_ratio": 1.5, "max_drawdown": 0.10},
        )
        
        frontier.add_point(point1)
        frontier.add_point(point2)
        
        assert frontier.size() == 2
    
    def test_get_best_for_objective(self):
        """Test récupération du meilleur pour un objectif."""
        frontier = ParetoFrontier()
        
        frontier.add_point(ParetoPoint(
            params={"x": 1},
            objectives={"sharpe_ratio": 1.0},
        ))
        frontier.add_point(ParetoPoint(
            params={"x": 2},
            objectives={"sharpe_ratio": 2.0},
        ))
        
        best = frontier.get_best("sharpe_ratio")
        assert best.params["x"] == 2
    
    def test_is_dominated(self):
        """Test vérification de domination."""
        frontier = ParetoFrontier()
        
        frontier.add_point(ParetoPoint(
            params={},
            objectives={"sharpe_ratio": 2.0, "total_return": 0.3},
        ))
        
        dominated = ParetoPoint(
            params={},
            objectives={"sharpe_ratio": 1.0, "total_return": 0.1},
        )
        
        assert frontier.is_dominated(dominated) is True


class TestParetoPruner:
    """Tests pour ParetoPruner."""
    
    def test_creation(self):
        """Test création du pruner."""
        pruner = ParetoPruner(
            objectives=["sharpe_ratio", "max_drawdown"],
            prune_threshold=0.8,
        )
        
        assert pruner.objectives == ["sharpe_ratio", "max_drawdown"]
        assert pruner.prune_threshold == 0.8
    
    def test_no_prune_initially(self):
        """Test pas de pruning au début."""
        pruner = ParetoPruner(
            objectives=["sharpe"],
            min_frontier_size=5,
        )
        
        # Pas assez de points sur la frontière
        should_prune = pruner.should_prune({"sharpe": 1.0})
        assert should_prune is False
    
    def test_report_updates_frontier(self):
        """Test que report met à jour la frontière."""
        pruner = ParetoPruner(objectives=["sharpe"])
        
        pruner.report(
            params={"x": 1},
            objectives={"sharpe": 1.5},
        )
        
        assert pruner.frontier.size() == 1
    
    def test_get_best_params(self):
        """Test récupération des meilleurs params."""
        pruner = ParetoPruner(objectives=["sharpe"])
        
        pruner.report({"x": 1}, {"sharpe": 1.0})
        pruner.report({"x": 2}, {"sharpe": 2.0})
        
        best = pruner.get_best_params("sharpe")
        assert best["x"] == 2
    
    def test_get_stats(self):
        """Test statistiques du pruner."""
        pruner = ParetoPruner(objectives=["sharpe"])
        
        pruner.should_prune({"sharpe": 1.0})
        pruner.should_prune({"sharpe": 0.5})
        pruner.report({"x": 1}, {"sharpe": 1.5})
        
        stats = pruner.get_stats()
        
        assert stats["total_evaluated"] == 2
        assert stats["total_reported"] == 1
        assert "frontier_size" in stats


class TestMultiObjectiveOptimizer:
    """Tests pour MultiObjectiveOptimizer."""
    
    def test_creation(self):
        """Test création de l'optimiseur."""
        opt = MultiObjectiveOptimizer(
            objectives=["sharpe", "return"],
            prune_threshold=0.7,
        )
        
        assert opt.objectives == ["sharpe", "return"]
    
    def test_optimize_simple(self):
        """Test optimisation simple."""
        opt = MultiObjectiveOptimizer(
            objectives=["value"],
            min_samples_before_prune=2,
        )
        
        def evaluate(params):
            return {"value": params["x"] * 2}
        
        results = opt.optimize(
            param_grid={"x": [1, 2, 3]},
            evaluate_fn=evaluate,
        )
        
        assert "frontier" in results
        assert "stats" in results
        assert results["stats"]["evaluated"] == 3
    
    def test_optimize_with_constraints(self):
        """Test optimisation avec contraintes."""
        opt = MultiObjectiveOptimizer(objectives=["value"])
        
        def evaluate(params):
            return {"value": params["x"]}
        
        def constraints(params):
            return params["x"] > 1
        
        results = opt.optimize(
            param_grid={"x": [1, 2, 3]},
            evaluate_fn=evaluate,
            constraints_fn=constraints,
        )
        
        # x=1 filtré par contraintes
        assert results["stats"]["evaluated"] == 2
    
    def test_optimize_with_progress_callback(self):
        """Test optimisation avec callback de progression."""
        opt = MultiObjectiveOptimizer(objectives=["value"])
        
        progress_calls = []
        
        def on_progress(current, total, info):
            progress_calls.append((current, total, info))
        
        def evaluate(params):
            return {"value": params["x"]}
        
        opt.optimize(
            param_grid={"x": [1, 2, 3]},
            evaluate_fn=evaluate,
            progress_callback=on_progress,
        )
        
        assert len(progress_calls) > 0


class TestParetoOptimizeFunction:
    """Tests pour la fonction utilitaire pareto_optimize."""
    
    def test_basic_usage(self):
        """Test utilisation basique."""
        def evaluate(params):
            x = params["x"]
            return {
                "sharpe_ratio": x,
                "max_drawdown": 1 / (x + 1),
            }
        
        results = pareto_optimize(
            param_grid={"x": [1, 2, 3, 4, 5]},
            evaluate_fn=evaluate,
            objectives=["sharpe_ratio", "max_drawdown"],
        )
        
        assert len(results["frontier"]) > 0
        assert "best_per_objective" in results
    
    def test_returns_best_per_objective(self):
        """Test que les meilleurs params par objectif sont retournés."""
        def evaluate(params):
            return {"obj": params["x"]}
        
        results = pareto_optimize(
            param_grid={"x": [1, 2, 3]},
            evaluate_fn=evaluate,
            objectives=["obj"],
        )
        
        assert results["best_per_objective"]["obj"]["x"] == 3
