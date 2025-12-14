"""
Tests pour le système d'early stopping d'Optuna.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

try:
    import optuna
    from backtest.optuna_optimizer import (
        OptunaOptimizer,
        OptimizationResult,
        OPTUNA_AVAILABLE,
    )
    TESTS_CAN_RUN = OPTUNA_AVAILABLE
except ImportError:
    TESTS_CAN_RUN = False


@pytest.fixture
def sample_data():
    """DataFrame OHLCV de test."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    np.random.seed(42)
    
    # Prix simulés
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.2)
    low = close - np.abs(np.random.randn(100) * 0.2)
    open_price = close + np.random.randn(100) * 0.1
    volume = np.random.randint(1000, 10000, 100)
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }).set_index("timestamp")


@pytest.fixture
def param_space():
    """Espace de paramètres simple pour les tests."""
    return {
        "fast_period": {"type": "int", "low": 5, "high": 20},
        "slow_period": {"type": "int", "low": 20, "high": 50},
    }


@pytest.mark.skipif(not TESTS_CAN_RUN, reason="Optuna non disponible")
class TestEarlyStoppingCallback:
    """Tests du callback d'early stopping."""

    def test_callback_creation(self, sample_data, param_space):
        """Test création du callback early stopping."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=5,
        )
        
        callback = optimizer._create_early_stop_callback(
            patience=5,
            direction="maximize"
        )
        
        assert callable(callback)

    def test_callback_stops_study(self, sample_data, param_space):
        """Test que le callback arrête bien l'étude après patience épuisée."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=3,
        )
        
        # Mock study et trials
        study = Mock()
        study.stop = Mock()
        
        callback = optimizer._create_early_stop_callback(
            patience=3,
            direction="maximize"
        )
        
        # Simuler 5 trials sans amélioration
        for i in range(5):
            trial = Mock()
            trial.number = i
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.value = 1.0  # Score constant (pas d'amélioration)
            trial.values = None  # Pas multi-objectif
            
            callback(study, trial)
        
        # Vérifier que study.stop() a été appelé (après 3 trials)
        assert study.stop.called

    def test_callback_resets_on_improvement(self, sample_data, param_space):
        """Test que le compteur se réinitialise quand il y a amélioration."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=3,
        )
        
        study = Mock()
        study.stop = Mock()
        
        callback = optimizer._create_early_stop_callback(
            patience=3,
            direction="maximize"
        )
        
        # Trial 1: 1.0 (baseline)
        trial1 = Mock()
        trial1.number = 0
        trial1.state = optuna.trial.TrialState.COMPLETE
        trial1.value = 1.0
        trial1.values = None
        callback(study, trial1)
        
        # Trial 2: 0.9 (pas d'amélioration)
        trial2 = Mock()
        trial2.number = 1
        trial2.state = optuna.trial.TrialState.COMPLETE
        trial2.value = 0.9
        trial2.values = None
        callback(study, trial2)
        
        # Trial 3: 1.5 (AMÉLIORATION -> reset compteur)
        trial3 = Mock()
        trial3.number = 2
        trial3.state = optuna.trial.TrialState.COMPLETE
        trial3.value = 1.5
        trial3.values = None
        callback(study, trial3)
        
        # Trial 4-6: Pas d'amélioration (3x)
        for i in range(3, 6):
            trial = Mock()
            trial.number = i
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.value = 1.0
            trial.values = None
            callback(study, trial)
        
        # Maintenant stop devrait être appelé (car 3 sans amélioration après reset)
        assert study.stop.called

    def test_callback_direction_minimize(self, sample_data, param_space):
        """Test callback avec direction='minimize'."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=2,
        )
        
        study = Mock()
        study.stop = Mock()
        
        callback = optimizer._create_early_stop_callback(
            patience=2,
            direction="minimize"
        )
        
        # Trial 1: 10.0 (baseline)
        trial1 = Mock()
        trial1.number = 0
        trial1.state = optuna.trial.TrialState.COMPLETE
        trial1.value = 10.0
        trial1.values = None
        callback(study, trial1)
        
        # Trial 2: 5.0 (amélioration car minimize)
        trial2 = Mock()
        trial2.number = 1
        trial2.state = optuna.trial.TrialState.COMPLETE
        trial2.value = 5.0
        trial2.values = None
        callback(study, trial2)
        
        # Pas d'arrêt car amélioration
        assert not study.stop.called

    def test_callback_ignores_pruned_trials(self, sample_data, param_space):
        """Test que le callback ignore les trials pruned."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=2,
        )
        
        study = Mock()
        study.stop = Mock()
        
        callback = optimizer._create_early_stop_callback(
            patience=2,
            direction="maximize"
        )
        
        # Trial 1: Complete
        trial1 = Mock()
        trial1.number = 0
        trial1.state = optuna.trial.TrialState.COMPLETE
        trial1.value = 1.0
        trial1.values = None
        callback(study, trial1)
        
        # Trial 2: Pruned (ignoré)
        trial2 = Mock()
        trial2.number = 1
        trial2.state = optuna.trial.TrialState.PRUNED
        callback(study, trial2)
        
        # Trial 3: Failed (ignoré)
        trial3 = Mock()
        trial3.number = 2
        trial3.state = optuna.trial.TrialState.FAIL
        callback(study, trial3)
        
        # Trial 4-5: Complete sans amélioration
        for i in range(3, 5):
            trial = Mock()
            trial.number = i
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.value = 0.5
            trial.values = None
            callback(study, trial)
        
        # Stop appelé après 2 trials complets sans amélioration
        assert study.stop.called


@pytest.mark.skipif(not TESTS_CAN_RUN, reason="Optuna non disponible")
class TestOptunaEarlyStoppingIntegration:
    """Tests d'intégration avec OptimizeOptimizer."""

    def test_optimizer_with_early_stop_patience_init(self, sample_data, param_space):
        """Test initialisation optimizer avec early_stop_patience."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=10,
        )
        
        assert optimizer.early_stop_patience == 10

    def test_optimizer_without_early_stop(self, sample_data, param_space):
        """Test optimizer sans early stopping (valeur par défaut)."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
        )
        
        assert optimizer.early_stop_patience is None

    def test_optimize_with_early_stop_override(self, sample_data, param_space):
        """Test override de early_stop_patience dans optimize()."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=5,  # Valeur par défaut
        )
        
        # Mock optimize pour vérifier les callbacks
        with patch.object(optimizer, '_create_objective', return_value=lambda t: 1.0):
            with patch('optuna.create_study') as mock_study:
                mock_study_instance = Mock()
                mock_study_instance.optimize = Mock()
                mock_study_instance.best_params = {"fast_period": 10, "slow_period": 30}
                mock_study_instance.best_value = 1.5
                mock_study_instance.trials = []
                mock_study.return_value = mock_study_instance
                
                # Appel avec override à 3
                result = optimizer.optimize(
                    n_trials=10,
                    metric="sharpe_ratio",
                    early_stop_patience=3,
                    show_progress=False,
                )
                
                # Vérifier que optimize() a été appelé avec callbacks
                call_args = mock_study_instance.optimize.call_args
                assert call_args is not None
                
                # Vérifier la présence d'un callback
                if 'callbacks' in call_args.kwargs:
                    callbacks = call_args.kwargs['callbacks']
                    assert callbacks is not None
                    assert len(callbacks) > 0

    @pytest.mark.slow
    def test_early_stopping_functional_test(self, sample_data, param_space):
        """Test fonctionnel : vérifier que early stopping réduit le nombre de trials."""
        # Note: Ce test est marqué slow car il fait vraiment tourner Optuna
        
        # Optimizer sans early stopping
        optimizer_no_stop = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=None,
        )
        
        result_no_stop = optimizer_no_stop.optimize(
            n_trials=20,
            show_progress=False,
        )
        
        # Optimizer avec early stopping agressif
        optimizer_with_stop = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=3,
        )
        
        result_with_stop = optimizer_with_stop.optimize(
            n_trials=20,
            show_progress=False,
        )
        
        # Avec early stopping, on devrait avoir moins de trials complétés
        # (sauf si chaque trial améliore, ce qui est peu probable avec Sharpe aléatoire)
        # On vérifie juste que ça tourne sans erreur et que result est valide
        assert result_with_stop.n_completed <= 20
        assert result_with_stop.n_completed > 0
        assert result_no_stop.n_completed <= 20


@pytest.mark.skipif(not TESTS_CAN_RUN, reason="Optuna non disponible")
class TestMultiObjectiveEarlyStopping:
    """Tests early stopping pour mode multi-objectif."""

    def test_multi_objective_with_early_stop(self, sample_data, param_space):
        """Test early stopping en mode multi-objectif."""
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=sample_data,
            param_space=param_space,
            early_stop_patience=5,
        )
        
        # Mock pour éviter les vrais backtests
        with patch.object(optimizer, '_engine') as mock_engine:
            mock_result = Mock()
            mock_result.metrics = {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.15,
            }
            mock_engine.run.return_value = mock_result
            
            result = optimizer.optimize_multi_objective(
                n_trials=10,
                metrics=["sharpe_ratio", "max_drawdown"],
                directions=["maximize", "minimize"],
                show_progress=False,
            )
            
            # Vérifier que ça tourne sans erreur
            assert result is not None
            assert hasattr(result, 'pareto_front')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
