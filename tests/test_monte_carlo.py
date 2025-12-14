"""
Tests pour Monte Carlo Sampling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestParameterSpace:
    """Tests pour ParameterSpace."""
    
    def test_sample_float(self):
        """Test échantillonnage float linéaire."""
        from backtest.monte_carlo import ParameterSpace
        
        space = ParameterSpace("test", 0.0, 10.0, "float")
        
        assert space.sample(0.0) == 0.0
        assert space.sample(1.0) == 10.0
        assert space.sample(0.5) == 5.0
    
    def test_sample_int(self):
        """Test échantillonnage int."""
        from backtest.monte_carlo import ParameterSpace
        
        space = ParameterSpace("test", 5, 20, "int")
        
        result = space.sample(0.5)
        assert isinstance(result, int)
        assert 5 <= result <= 20
    
    def test_sample_log_scale(self):
        """Test échantillonnage log-scale."""
        from backtest.monte_carlo import ParameterSpace
        
        space = ParameterSpace("test", 1.0, 100.0, "float", log_scale=True)
        
        # À 0.5 en log-scale, on devrait être autour de sqrt(1*100) = 10
        result = space.sample(0.5)
        assert 8 < result < 12


class TestMonteCarloSampler:
    """Tests pour MonteCarloSampler."""
    
    def test_random_sampling(self):
        """Test échantillonnage aléatoire."""
        from backtest.monte_carlo import (
            MonteCarloSampler, 
            ParameterSpace, 
            SamplingMethod
        )
        
        spaces = [
            ParameterSpace("p1", 0, 10, "int"),
            ParameterSpace("p2", 0.0, 1.0, "float"),
        ]
        
        sampler = MonteCarloSampler(
            param_spaces=spaces,
            n_samples=20,
            method=SamplingMethod.RANDOM,
            seed=42,
        )
        
        samples = sampler.generate_samples()
        
        assert len(samples) == 20
        for s in samples:
            assert "p1" in s
            assert "p2" in s
            assert 0 <= s["p1"] <= 10
            assert 0.0 <= s["p2"] <= 1.0
    
    def test_latin_hypercube_sampling(self):
        """Test Latin Hypercube Sampling."""
        from backtest.monte_carlo import (
            MonteCarloSampler, 
            ParameterSpace, 
            SamplingMethod
        )
        
        spaces = [
            ParameterSpace("x", 0, 100, "int"),
        ]
        
        sampler = MonteCarloSampler(
            param_spaces=spaces,
            n_samples=10,
            method=SamplingMethod.LATIN_HYPERCUBE,
            seed=42,
        )
        
        samples = sampler.generate_samples()
        
        assert len(samples) == 10
        
        # LHS devrait couvrir tout l'espace
        values = [s["x"] for s in samples]
        assert min(values) < 20  # Couvre le bas
        assert max(values) > 80  # Couvre le haut
    
    def test_reproducibility_with_seed(self):
        """Test que le seed assure la reproductibilité."""
        from backtest.monte_carlo import (
            MonteCarloSampler, 
            ParameterSpace, 
            SamplingMethod
        )
        
        spaces = [ParameterSpace("x", 0, 100, "float")]
        
        sampler1 = MonteCarloSampler(spaces, n_samples=5, seed=123)
        sampler2 = MonteCarloSampler(spaces, n_samples=5, seed=123)
        
        # Reset seed pour sampler2
        np.random.seed(123)
        samples1 = sampler1.generate_samples()
        
        np.random.seed(123)
        samples2 = MonteCarloSampler(spaces, n_samples=5, seed=123).generate_samples()
        
        # Les valeurs devraient être identiques
        for s1, s2 in zip(samples1, samples2):
            assert s1["x"] == s2["x"]


class TestMonteCarloOptimizer:
    """Tests pour MonteCarloOptimizer."""
    
    def test_basic_optimization(self):
        """Test optimisation basique."""
        from backtest.monte_carlo import (
            MonteCarloOptimizer, 
            ParameterSpace, 
            SamplingMethod
        )
        
        spaces = [
            ParameterSpace("x", -10, 10, "float"),
            ParameterSpace("y", -10, 10, "float"),
        ]
        
        optimizer = MonteCarloOptimizer(
            param_spaces=spaces,
            n_samples=50,
            method=SamplingMethod.LATIN_HYPERCUBE,
            seed=42,
        )
        
        # Fonction simple: maximum à (0, 0)
        def evaluate(params):
            return -(params["x"]**2 + params["y"]**2)
        
        result = optimizer.optimize(evaluate)
        
        assert result.n_evaluated > 0
        assert result.best_score > -200  # Pas le pire cas
        assert len(result.convergence_history) > 0
    
    def test_with_constraints(self):
        """Test optimisation avec contraintes."""
        from backtest.monte_carlo import (
            MonteCarloOptimizer, 
            ParameterSpace, 
            SamplingMethod
        )
        
        spaces = [
            ParameterSpace("fast", 5, 20, "int"),
            ParameterSpace("slow", 10, 50, "int"),
        ]
        
        optimizer = MonteCarloOptimizer(
            param_spaces=spaces,
            n_samples=30,
            seed=42,
        )
        
        def evaluate(params):
            return params["slow"] - params["fast"]
        
        def constraints(params):
            return params["slow"] > params["fast"]
        
        result = optimizer.optimize(evaluate, constraints_fn=constraints)
        
        # Tous les résultats doivent respecter la contrainte
        for params in result.samples:
            assert params["slow"] > params["fast"]
    
    def test_early_stopping(self):
        """Test early stopping."""
        from backtest.monte_carlo import (
            MonteCarloOptimizer, 
            ParameterSpace,
        )
        
        spaces = [ParameterSpace("x", 0, 100, "int")]
        
        optimizer = MonteCarloOptimizer(
            param_spaces=spaces,
            n_samples=100,
            early_stop_patience=5,
            seed=42,
        )
        
        call_count = [0]
        
        def evaluate(params):
            call_count[0] += 1
            # Score constant = pas d'amélioration
            return 1.0
        
        result = optimizer.optimize(evaluate)
        
        # Devrait s'arrêter avant 100 évaluations
        assert result.n_evaluated < 100
        assert result.n_evaluated >= 5  # Au moins patience
    
    def test_top_k(self):
        """Test récupération top k résultats."""
        from backtest.monte_carlo import (
            MonteCarloOptimizer, 
            ParameterSpace,
        )
        
        spaces = [ParameterSpace("x", 0, 100, "int")]
        
        optimizer = MonteCarloOptimizer(
            param_spaces=spaces,
            n_samples=20,
            early_stop_patience=100,  # Pas d'early stop
            seed=42,
        )
        
        def evaluate(params):
            return params["x"]  # Plus grand = mieux
        
        result = optimizer.optimize(evaluate)
        
        top_5 = result.top_k(5)
        assert len(top_5) == 5
        
        # Vérifier l'ordre décroissant
        scores = [s for _, s in top_5]
        assert scores == sorted(scores, reverse=True)
    
    def test_progress_callback(self):
        """Test callback de progression."""
        from backtest.monte_carlo import (
            MonteCarloOptimizer, 
            ParameterSpace,
        )
        
        spaces = [ParameterSpace("x", 0, 10, "int")]
        
        optimizer = MonteCarloOptimizer(
            param_spaces=spaces,
            n_samples=10,
            early_stop_patience=100,
            seed=42,
        )
        
        progress_calls = []
        
        def callback(current, total, best):
            progress_calls.append((current, total, best))
        
        result = optimizer.optimize(
            lambda p: p["x"],
            progress_callback=callback,
        )
        
        assert len(progress_calls) == result.n_evaluated
        assert progress_calls[-1][0] == result.n_evaluated


class TestMonteCarloFromStrategy:
    """Tests pour création depuis une stratégie."""
    
    def test_from_strategy(self):
        """Test création depuis une stratégie enregistrée."""
        from backtest.monte_carlo import MonteCarloOptimizer
        
        # ema_cross devrait avoir des parameter_specs
        optimizer = MonteCarloOptimizer.from_strategy(
            "ema_cross",
            n_samples=10,
        )
        
        assert len(optimizer.param_spaces) > 0
        
        # Vérifier que fast_period et slow_period sont présents
        names = [s.name for s in optimizer.param_spaces]
        assert "fast_period" in names
        assert "slow_period" in names
    
    def test_from_invalid_strategy_raises(self):
        """Test erreur si stratégie invalide."""
        from backtest.monte_carlo import MonteCarloOptimizer
        
        with pytest.raises(ValueError):
            MonteCarloOptimizer.from_strategy("strategie_inexistante")


class TestMonteCarloSweep:
    """Tests pour la fonction monte_carlo_sweep."""
    
    @pytest.fixture
    def sample_data(self):
        """Données OHLCV de test."""
        import pandas as pd
        np.random.seed(42)
        n = 500
        price = 100 + np.cumsum(np.random.randn(n) * 0.5)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        return pd.DataFrame({
            "open": price,
            "high": price + np.abs(np.random.randn(n) * 0.3),
            "low": price - np.abs(np.random.randn(n) * 0.3),
            "close": price + np.random.randn(n) * 0.2,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }, index=dates)
    
    def test_monte_carlo_sweep(self, sample_data):
        """Test sweep Monte Carlo complet."""
        from backtest.monte_carlo import monte_carlo_sweep
        
        result = monte_carlo_sweep(
            strategy_name="ema_cross",
            data=sample_data,
            n_samples=10,
            metric="sharpe_ratio",
            seed=42,
            constraints_fn=lambda p: p.get("slow_period", 30) > p.get("fast_period", 10),
        )
        
        assert result.n_evaluated > 0
        assert result.best_params is not None
        assert "fast_period" in result.best_params
    
    def test_monte_carlo_sweep_to_dict(self, sample_data):
        """Test export to_dict."""
        from backtest.monte_carlo import monte_carlo_sweep
        
        result = monte_carlo_sweep(
            strategy_name="ema_cross",
            data=sample_data,
            n_samples=10,
            seed=42,
        )
        
        d = result.to_dict()
        
        assert "n_samples" in d
        assert "best_params" in d
        assert "best_score" in d
        assert "top_5" in d
