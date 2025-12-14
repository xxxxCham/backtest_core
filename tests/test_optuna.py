"""
Tests pour le module d'optimisation Optuna.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import conditionnel pour les tests
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Crée des données OHLCV de test."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": close - np.abs(np.random.randn(n) * 0.2),
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.fixture
def simple_param_space():
    """Espace de paramètres simple pour les tests."""
    return {
        "fast_period": {"type": "int", "low": 5, "high": 20},
        "slow_period": {"type": "int", "low": 20, "high": 50},
    }


@pytest.fixture
def float_param_space():
    """Espace de paramètres avec floats."""
    return {
        "multiplier": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
        "threshold": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
    }


@pytest.fixture
def categorical_param_space():
    """Espace de paramètres avec catégories."""
    return {
        "ma_type": {"type": "categorical", "choices": ["sma", "ema"]},
        "period": {"type": "int", "low": 5, "high": 50},
    }


# ============================================================================
# TESTS PARAMSPEC
# ============================================================================

class TestParamSpec:
    """Tests pour la classe ParamSpec."""
    
    def test_import(self):
        """Test que ParamSpec peut être importé."""
        from backtest.optuna_optimizer import ParamSpec
        assert ParamSpec is not None
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_suggest_int(self):
        """Test suggestion d'un paramètre int."""
        from backtest.optuna_optimizer import ParamSpec
        
        spec = ParamSpec(
            name="period",
            param_type="int",
            low=5,
            high=50,
        )
        
        # Mock du trial Optuna
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 20
        
        value = spec.suggest(mock_trial)
        
        assert value == 20
        mock_trial.suggest_int.assert_called_once_with(
            "period", 5, 50, step=1, log=False
        )
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_suggest_float(self):
        """Test suggestion d'un paramètre float."""
        from backtest.optuna_optimizer import ParamSpec
        
        spec = ParamSpec(
            name="multiplier",
            param_type="float",
            low=1.0,
            high=3.0,
            step=0.1,
        )
        
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 2.5
        
        value = spec.suggest(mock_trial)
        
        assert value == 2.5
        mock_trial.suggest_float.assert_called_once()
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_suggest_float_log_scale(self):
        """Test suggestion float avec échelle logarithmique."""
        from backtest.optuna_optimizer import ParamSpec
        
        spec = ParamSpec(
            name="learning_rate",
            param_type="float",
            low=0.001,
            high=1.0,
            log=True,
        )
        
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.01
        
        value = spec.suggest(mock_trial)
        
        assert value == 0.01
        mock_trial.suggest_float.assert_called_once_with(
            "learning_rate", 0.001, 1.0, step=None, log=True
        )
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_suggest_categorical(self):
        """Test suggestion d'un paramètre catégorique."""
        from backtest.optuna_optimizer import ParamSpec
        
        spec = ParamSpec(
            name="ma_type",
            param_type="categorical",
            choices=["sma", "ema", "wma"],
        )
        
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = "ema"
        
        value = spec.suggest(mock_trial)
        
        assert value == "ema"
        mock_trial.suggest_categorical.assert_called_once_with(
            "ma_type", ["sma", "ema", "wma"]
        )
    
    def test_suggest_invalid_type(self):
        """Test avec un type de paramètre invalide."""
        from backtest.optuna_optimizer import ParamSpec
        
        spec = ParamSpec(
            name="invalid",
            param_type="unknown",
        )
        
        mock_trial = Mock()
        
        with pytest.raises(ValueError, match="Type de paramètre inconnu"):
            spec.suggest(mock_trial)


# ============================================================================
# TESTS OPTIMIZATIONRESULT
# ============================================================================

class TestOptimizationResult:
    """Tests pour la classe OptimizationResult."""
    
    def test_import(self):
        """Test que OptimizationResult peut être importé."""
        from backtest.optuna_optimizer import OptimizationResult
        assert OptimizationResult is not None
    
    def test_creation(self):
        """Test création d'un OptimizationResult."""
        from backtest.optuna_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={"fast": 10, "slow": 30},
            best_value=1.5,
            best_metrics={"sharpe_ratio": 1.5, "total_return": 0.25},
            n_trials=100,
            n_completed=95,
            n_pruned=5,
            total_time=60.0,
            history=[{"params": {"fast": 10}, "value": 1.2}],
        )
        
        assert result.best_params == {"fast": 10, "slow": 30}
        assert result.best_value == 1.5
        assert result.n_trials == 100
        assert result.n_completed == 95
        assert result.n_pruned == 5
    
    def test_to_dataframe(self):
        """Test conversion en DataFrame."""
        from backtest.optuna_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={},
            best_value=0,
            best_metrics={},
            n_trials=3,
            n_completed=3,
            n_pruned=0,
            total_time=1.0,
            history=[
                {"params": {"x": 1}, "value": 0.5},
                {"params": {"x": 2}, "value": 0.8},
                {"params": {"x": 3}, "value": 0.6},
            ],
        )
        
        df = result.to_dataframe()
        
        assert len(df) == 3
        assert "value" in df.columns
    
    def test_get_top_n(self):
        """Test récupération des N meilleurs."""
        from backtest.optuna_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={},
            best_value=0,
            best_metrics={},
            n_trials=5,
            n_completed=5,
            n_pruned=0,
            total_time=1.0,
            history=[
                {"params": {"x": i}, "value": i * 0.1}
                for i in range(5)
            ],
        )
        
        top = result.get_top_n(3, ascending=False)
        
        assert len(top) == 3
        assert top.iloc[0]["value"] == 0.4  # Plus grand
    
    def test_summary(self):
        """Test génération du résumé."""
        from backtest.optuna_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={"x": 10},
            best_value=1.5,
            best_metrics={"sharpe_ratio": 1.5, "total_return": 0.25},
            n_trials=100,
            n_completed=90,
            n_pruned=10,
            total_time=120.0,
        )
        
        summary = result.summary()
        
        assert "100" in summary  # n_trials
        assert "90" in summary   # n_completed
        assert "10" in summary   # n_pruned
        assert "1.5" in summary  # best_value


# ============================================================================
# TESTS OPTUNAOPTIMIZER
# ============================================================================

class TestOptunaOptimizer:
    """Tests pour la classe OptunaOptimizer."""
    
    def test_import(self):
        """Test que OptunaOptimizer peut être importé."""
        from backtest.optuna_optimizer import OptunaOptimizer
        assert OptunaOptimizer is not None
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_initialization(self, sample_ohlcv, simple_param_space):
        """Test initialisation de l'optimiseur."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
        )
        
        assert optimizer.strategy_name == "ema_cross"
        assert len(optimizer.param_specs) == 2
        assert optimizer.initial_capital == 10000.0
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_initialization_with_constraints(self, sample_ohlcv, simple_param_space):
        """Test initialisation avec contraintes."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        assert len(optimizer.constraints) == 1
        assert optimizer.constraints[0] == ("slow_period", ">", "fast_period")
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_check_constraints_valid(self, sample_ohlcv, simple_param_space):
        """Test validation de contraintes valides."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        # Contrainte respectée
        assert optimizer._check_constraints({"fast_period": 10, "slow_period": 30})
        
        # Contrainte violée
        assert not optimizer._check_constraints({"fast_period": 30, "slow_period": 10})
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_check_constraints_operators(self, sample_ohlcv, simple_param_space):
        """Test différents opérateurs de contraintes."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        # Greater than
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        assert optimizer._check_constraints({"fast_period": 10, "slow_period": 20})
        assert not optimizer._check_constraints({"fast_period": 20, "slow_period": 10})
        
        # Less than
        optimizer.constraints = [("fast_period", "<", "slow_period")]
        assert optimizer._check_constraints({"fast_period": 10, "slow_period": 20})
        
        # Greater or equal
        optimizer.constraints = [("slow_period", ">=", "fast_period")]
        assert optimizer._check_constraints({"fast_period": 10, "slow_period": 10})
        
        # Less or equal
        optimizer.constraints = [("fast_period", "<=", "slow_period")]
        assert optimizer._check_constraints({"fast_period": 10, "slow_period": 10})


# ============================================================================
# TESTS OPTIMIZATION (avec mocks)
# ============================================================================

class TestOptimization:
    """Tests d'optimisation avec mocks."""
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_optimize_runs(self, sample_ohlcv, simple_param_space):
        """Test que l'optimisation s'exécute."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        # Test avec très peu de trials pour la vitesse
        result = optimizer.optimize(
            n_trials=3,
            metric="sharpe_ratio",
            show_progress=False,
        )
        
        assert result is not None
        assert result.best_params is not None
        assert result.n_completed >= 0
        assert "fast_period" in result.best_params
        assert "slow_period" in result.best_params
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_optimize_respects_constraints(self, sample_ohlcv, simple_param_space):
        """Test que les contraintes sont respectées."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        result = optimizer.optimize(n_trials=5, show_progress=False)
        
        # Le meilleur résultat doit respecter la contrainte
        best = result.best_params
        assert best["slow_period"] > best["fast_period"]
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_optimize_with_timeout(self, sample_ohlcv, simple_param_space):
        """Test optimisation avec timeout."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
        )
        
        # Timeout très court
        result = optimizer.optimize(
            n_trials=100,  # Beaucoup de trials
            timeout=2,     # Mais timeout 2s
            show_progress=False,
        )
        
        assert result is not None
        # Probablement moins de 100 trials complétés
        assert result.n_completed <= 100


# ============================================================================
# TESTS CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Tests pour les fonctions utilitaires."""
    
    def test_suggest_param_space_known_strategy(self):
        """Test suggestion pour stratégie connue."""
        from backtest.optuna_optimizer import suggest_param_space
        
        space = suggest_param_space("ema_cross")
        
        assert "fast_period" in space
        assert "slow_period" in space
        assert space["fast_period"]["type"] == "int"
    
    def test_suggest_param_space_unknown_strategy(self):
        """Test suggestion pour stratégie inconnue."""
        from backtest.optuna_optimizer import suggest_param_space
        
        space = suggest_param_space("unknown_strategy")
        
        # Devrait retourner un espace générique
        assert len(space) > 0
    
    def test_suggest_param_space_rsi(self):
        """Test suggestion pour RSI reversal."""
        from backtest.optuna_optimizer import suggest_param_space
        
        space = suggest_param_space("rsi_reversal")
        
        assert "rsi_period" in space
        assert "oversold_level" in space
        assert "overbought_level" in space
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_quick_optimize(self, sample_ohlcv, simple_param_space):
        """Test de quick_optimize."""
        from backtest.optuna_optimizer import quick_optimize
        
        result = quick_optimize(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
            n_trials=3,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        assert result is not None
        assert result.best_params is not None


# ============================================================================
# TESTS MULTI-OBJECTIF
# ============================================================================

class TestMultiObjective:
    """Tests pour l'optimisation multi-objectif."""
    
    def test_import_multi_objective_result(self):
        """Test import de MultiObjectiveResult."""
        from backtest.optuna_optimizer import MultiObjectiveResult
        assert MultiObjectiveResult is not None
    
    def test_multi_objective_result_creation(self):
        """Test création d'un MultiObjectiveResult."""
        from backtest.optuna_optimizer import MultiObjectiveResult
        
        result = MultiObjectiveResult(
            pareto_front=[
                {"params": {"x": 1}, "values": [0.5, 0.2]},
                {"params": {"x": 2}, "values": [0.4, 0.1]},
            ],
            n_trials=50,
            total_time=30.0,
        )
        
        assert len(result.pareto_front) == 2
        assert result.n_trials == 50
    
    def test_multi_objective_to_dataframe(self):
        """Test conversion Pareto en DataFrame."""
        from backtest.optuna_optimizer import MultiObjectiveResult
        
        result = MultiObjectiveResult(
            pareto_front=[
                {"params": {"x": 1}, "values": [0.5, 0.2]},
                {"params": {"x": 2}, "values": [0.4, 0.1]},
            ],
            n_trials=50,
            total_time=30.0,
        )
        
        df = result.to_dataframe()
        
        assert len(df) == 2
        assert "params" in df.columns


# ============================================================================
# TESTS EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests des cas limites."""
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_empty_param_space(self, sample_ohlcv):
        """Test avec espace de paramètres vide."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        # Devrait lever une erreur ou fonctionner avec défauts
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space={},
        )
        
        assert optimizer.param_specs == []
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_single_param(self, sample_ohlcv):
        """Test avec un seul paramètre."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space={"period": {"type": "int", "low": 5, "high": 50}},
        )
        
        result = optimizer.optimize(n_trials=3, show_progress=False)
        
        assert "period" in result.best_params
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_zero_trials(self, sample_ohlcv, simple_param_space):
        """Test avec 0 trials."""
        from backtest.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=simple_param_space,
        )
        
        # Optuna lève une erreur avec 0 trials car pas de best_trial
        with pytest.raises(ValueError, match="No trials"):
            optimizer.optimize(n_trials=0, show_progress=False)


# ============================================================================
# TESTS INTÉGRATION BACKTEST
# ============================================================================

class TestBacktestIntegration:
    """Tests d'intégration avec le moteur de backtest."""
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna non installé")
    def test_full_pipeline(self, sample_ohlcv):
        """Test du pipeline complet d'optimisation."""
        from backtest.optuna_optimizer import OptunaOptimizer, suggest_param_space
        
        # Utiliser l'espace suggéré
        param_space = suggest_param_space("ema_cross")
        
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_ohlcv,
            param_space=param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        result = optimizer.optimize(
            n_trials=5,
            metric="sharpe_ratio",
            show_progress=False,
        )
        
        # Vérifications
        assert result.best_params is not None
        assert result.best_metrics is not None
        assert "sharpe_ratio" in result.best_metrics or result.best_value != 0
        assert result.n_completed > 0
        assert result.total_time > 0


# ============================================================================
# TESTS CLI
# ============================================================================

class TestCLIIntegration:
    """Tests pour l'intégration CLI."""
    
    def test_optuna_command_exists(self):
        """Test que la commande optuna existe dans le CLI."""
        from cli import create_parser
        
        parser = create_parser()
        
        # Parser la commande optuna
        args = parser.parse_args(["optuna", "-s", "ema_cross", "-d", "test.parquet"])
        
        assert args.command == "optuna"
        assert args.strategy == "ema_cross"
    
    def test_optuna_command_all_args(self):
        """Test parsing de tous les arguments optuna."""
        from cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "optuna",
            "-s", "ema_cross",
            "-d", "test.parquet",
            "-n", "50",
            "-m", "sharpe",
            "--sampler", "tpe",
            "--pruning",
            "--parallel", "2",
            "--capital", "50000",
            "-o", "output.json",
        ])
        
        assert args.n_trials == 50
        assert args.metric == "sharpe"
        assert args.sampler == "tpe"
        assert args.pruning is True
        assert args.parallel == 2
        assert args.capital == 50000.0
        assert args.output == "output.json"
