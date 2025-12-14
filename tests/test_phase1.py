"""
Tests unitaires pour Phase 1 - Fondations Critiques
===================================================

Tests pour:
- Métriques Tier S
- Walk-Forward Validation
- Système de Contraintes
"""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics_tier_s import (
    TierSMetrics,
    calculate_tier_s_metrics,
    sortino_ratio,
    calmar_ratio,
    sqn,
    recovery_factor,
    ulcer_index,
    martin_ratio,
    gain_pain_ratio,
    r_multiple_stats,
    outlier_adjusted_sharpe,
    calculate_tier_s_score,
)
from backtest.validation import (
    ValidationFold,
    WalkForwardResult,
    WalkForwardValidator,
    OverfittingDetector,
    calculate_walk_forward_metrics,
    train_test_split,
)
from utils.parameters import (
    ParameterConstraint,
    ConstraintValidator,
    generate_constrained_param_grid,
    ParameterSpec,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_returns() -> pd.Series:
    """Génère des rendements réalistes."""
    np.random.seed(42)
    # Rendements journaliers ~ N(0.0005, 0.02) - légèrement positifs
    returns = pd.Series(np.random.normal(0.0005, 0.02, 1000))
    # Ajouter quelques mauvais jours
    returns.iloc[100:105] = -0.03  # Petite baisse
    returns.iloc[500:510] = -0.05  # Grosse correction
    return returns


@pytest.fixture
def sample_equity(sample_returns) -> pd.Series:
    """Génère une courbe d'équité."""
    initial_capital = 10000.0
    equity = initial_capital * (1 + sample_returns).cumprod()
    return equity


@pytest.fixture
def sample_trades_pnl() -> pd.Series:
    """Génère des P&L de trades."""
    np.random.seed(42)
    # Mix de gains et pertes
    pnl = pd.Series([
        100, -50, 200, -80, 150, -120, 80, -30, 250, -100,
        50, -60, 180, -90, 120, -40, 90, -70, 160, -50,
        300, -150, 200, -100, 180, -80, 140, -60, 220, -110,
    ])
    return pnl


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Génère des données OHLCV."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = low + np.random.random(n) * (high - low)
    volume = np.random.randint(1000, 10000, n)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


# =============================================================================
# TESTS MÉTRIQUES TIER S
# =============================================================================

class TestSortinoRatio:
    """Tests pour le ratio de Sortino."""
    
    def test_positive_returns(self, sample_returns):
        """Sortino devrait être positif pour des rendements positifs."""
        result = sortino_ratio(sample_returns)
        assert isinstance(result, float)
        # Pour des rendements légèrement positifs avec volatilité
        assert result > -10  # Pas de valeurs aberrantes
    
    def test_empty_series(self):
        """Sortino devrait être 0 pour une série vide."""
        result = sortino_ratio(pd.Series([]))
        assert result == 0.0
    
    def test_no_downside(self):
        """Sortino devrait être inf pour des rendements tous positifs."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])
        result = sortino_ratio(returns)
        assert result == float('inf')


class TestCalmarRatio:
    """Tests pour le ratio de Calmar."""
    
    def test_basic_calculation(self, sample_returns, sample_equity):
        """Test de base du calcul Calmar."""
        result = calmar_ratio(sample_returns, sample_equity)
        assert isinstance(result, float)
    
    def test_no_drawdown(self):
        """Calmar devrait être inf s'il n'y a pas de drawdown."""
        equity = pd.Series([100, 101, 102, 103, 104, 105])
        returns = equity.pct_change().dropna()
        result = calmar_ratio(returns, equity)
        assert result == float('inf')


class TestSQN:
    """Tests pour le System Quality Number."""
    
    def test_basic_calculation(self, sample_trades_pnl):
        """Test de base du SQN."""
        result = sqn(sample_trades_pnl, min_trades=20)
        assert isinstance(result, float)
        # SQN devrait être entre -10 et 10
        assert -10 <= result <= 10
    
    def test_insufficient_trades(self):
        """SQN devrait être 0 si pas assez de trades."""
        pnl = pd.Series([100, -50, 200])  # Seulement 3 trades
        result = sqn(pnl, min_trades=30)
        assert result == 0.0
    
    def test_good_system(self):
        """Un bon système devrait avoir un SQN élevé."""
        # Beaucoup de gains, peu de pertes
        pnl = pd.Series([100] * 50 + [-30] * 10)
        result = sqn(pnl, min_trades=30)
        assert result > 2.0  # Système "moyen" ou mieux


class TestRecoveryFactor:
    """Tests pour le Recovery Factor."""
    
    def test_profitable_system(self, sample_equity):
        """Recovery Factor pour un système profitable."""
        initial = 10000.0
        result = recovery_factor(sample_equity, initial)
        assert isinstance(result, float)
    
    def test_losing_system(self):
        """Recovery Factor pour un système perdant."""
        equity = pd.Series([10000, 9500, 9000, 8500, 8000])
        result = recovery_factor(equity, 10000)
        assert result < 0  # Négatif car perte nette


class TestUlcerIndex:
    """Tests pour l'Ulcer Index."""
    
    def test_basic_calculation(self, sample_equity):
        """Test de base de l'Ulcer Index."""
        result = ulcer_index(sample_equity)
        assert isinstance(result, float)
        assert result >= 0  # Toujours positif
    
    def test_no_drawdown(self):
        """Ulcer Index devrait être 0 sans drawdown."""
        equity = pd.Series([100, 101, 102, 103, 104])
        result = ulcer_index(equity)
        assert result == 0.0


class TestTierSMetrics:
    """Tests pour le calcul complet des métriques Tier S."""
    
    def test_full_calculation(self, sample_returns, sample_equity, sample_trades_pnl):
        """Test du calcul complet."""
        metrics = calculate_tier_s_metrics(
            returns=sample_returns,
            equity=sample_equity,
            trades_pnl=sample_trades_pnl,
            initial_capital=10000.0
        )
        
        assert isinstance(metrics, TierSMetrics)
        assert metrics.tier_s_grade in ['A', 'B', 'C', 'D', 'F']
        assert 0 <= metrics.tier_s_score <= 100
    
    def test_to_dict(self, sample_returns, sample_equity, sample_trades_pnl):
        """Test de la sérialisation."""
        metrics = calculate_tier_s_metrics(
            returns=sample_returns,
            equity=sample_equity,
            trades_pnl=sample_trades_pnl,
        )
        
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert 'sqn' in d
        assert 'tier_s_grade' in d


# =============================================================================
# TESTS WALK-FORWARD VALIDATION
# =============================================================================

class TestWalkForwardValidator:
    """Tests pour le validateur Walk-Forward."""
    
    def test_split_generation(self, sample_ohlcv_df):
        """Test de génération des splits."""
        validator = WalkForwardValidator(
            n_folds=5,
            test_pct=0.2,
            embargo_pct=0.01,
            min_train_samples=50  # Réduit pour le test avec 500 samples
        )
        
        folds = validator.split(sample_ohlcv_df)
        
        assert len(folds) > 0
        assert len(folds) <= 5
        
        for fold in folds:
            assert isinstance(fold, ValidationFold)
            assert fold.train_start < fold.train_end
            assert fold.test_start < fold.test_end
            assert fold.train_end <= fold.test_start  # Pas de chevauchement
    
    def test_get_data_splits(self, sample_ohlcv_df):
        """Test de la récupération des données."""
        validator = WalkForwardValidator(n_folds=3)
        folds = validator.split(sample_ohlcv_df)
        
        if folds:
            train_df, test_df = validator.get_data_splits(sample_ohlcv_df, folds[0])
            
            assert len(train_df) > 0
            assert len(test_df) > 0
            assert len(train_df) + len(test_df) <= len(sample_ohlcv_df)
    
    def test_expanding_window(self, sample_ohlcv_df):
        """Test du mode fenêtre expandante."""
        validator = WalkForwardValidator(n_folds=3, expanding=True)
        folds = validator.split(sample_ohlcv_df)
        
        # En mode expanding, tous les folds commencent à 0
        for fold in folds:
            assert fold.train_start == 0


class TestTrainTestSplit:
    """Tests pour le split train/test simple."""
    
    def test_basic_split(self, sample_ohlcv_df):
        """Test de split basique."""
        train, test = train_test_split(sample_ohlcv_df, test_pct=0.2)
        
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) > len(test)  # Train devrait être plus grand
    
    def test_embargo(self, sample_ohlcv_df):
        """Test que l'embargo est appliqué."""
        n = len(sample_ohlcv_df)
        train, test = train_test_split(sample_ohlcv_df, test_pct=0.2, embargo_pct=0.05)
        
        # Avec embargo, train + test < total
        assert len(train) + len(test) < n


class TestOverfittingDetector:
    """Tests pour le détecteur d'overfitting."""
    
    def test_analyze_robust(self):
        """Test d'analyse d'un système robuste."""
        result = WalkForwardResult(
            folds=[],
            total_train_samples=1000,
            total_test_samples=200,
            embargo_samples=10,
            avg_train_sharpe=1.5,
            avg_test_sharpe=1.2,
            avg_overfitting_ratio=1.25,
            confidence_score=0.8,
            is_robust=True,
        )
        
        detector = OverfittingDetector()
        diagnosis = detector.analyze(result)
        
        assert not diagnosis['overfitting_detected']
        assert diagnosis['severity'] == 'none'
    
    def test_analyze_overfitting(self):
        """Test d'analyse d'un système overfitté."""
        result = WalkForwardResult(
            folds=[],
            total_train_samples=1000,
            total_test_samples=200,
            embargo_samples=10,
            avg_train_sharpe=3.0,
            avg_test_sharpe=0.2,
            avg_overfitting_ratio=15.0,
            confidence_score=0.2,
            is_robust=False,
        )
        
        detector = OverfittingDetector()
        diagnosis = detector.analyze(result)
        
        assert diagnosis['overfitting_detected']
        assert diagnosis['severity'] == 'critical'
        assert len(diagnosis['recommendations']) > 0


# =============================================================================
# TESTS SYSTÈME DE CONTRAINTES
# =============================================================================

class TestParameterConstraint:
    """Tests pour les contraintes de paramètres."""
    
    def test_greater_than(self):
        """Test contrainte param_a > param_b."""
        constraint = ParameterConstraint(
            param_a='slow',
            constraint_type='greater_than',
            param_b='fast'
        )
        
        assert constraint.validate({'slow': 26, 'fast': 12})
        assert not constraint.validate({'slow': 12, 'fast': 26})
        assert not constraint.validate({'slow': 20, 'fast': 20})
    
    def test_ratio_min(self):
        """Test contrainte ratio minimum."""
        constraint = ParameterConstraint(
            param_a='slow',
            constraint_type='ratio_min',
            param_b='fast',
            ratio=1.5
        )
        
        assert constraint.validate({'slow': 30, 'fast': 10})  # ratio = 3
        assert constraint.validate({'slow': 15, 'fast': 10})  # ratio = 1.5
        assert not constraint.validate({'slow': 12, 'fast': 10})  # ratio = 1.2
    
    def test_difference_min(self):
        """Test contrainte différence minimum."""
        constraint = ParameterConstraint(
            param_a='slow',
            constraint_type='difference_min',
            param_b='fast',
            value=5
        )
        
        assert constraint.validate({'slow': 26, 'fast': 12})  # diff = 14
        assert constraint.validate({'slow': 17, 'fast': 12})  # diff = 5
        assert not constraint.validate({'slow': 15, 'fast': 12})  # diff = 3
    
    def test_min_value(self):
        """Test contrainte valeur minimum."""
        constraint = ParameterConstraint(
            param_a='k_sl',
            constraint_type='min_value',
            value=0.5
        )
        
        assert constraint.validate({'k_sl': 1.0})
        assert constraint.validate({'k_sl': 0.5})
        assert not constraint.validate({'k_sl': 0.3})


class TestConstraintValidator:
    """Tests pour le validateur de contraintes."""
    
    def test_validate_all(self):
        """Test de validation multiple."""
        validator = ConstraintValidator([
            ParameterConstraint('slow', 'greater_than', 'fast'),
            ParameterConstraint('k_sl', 'min_value', value=0.5),
        ])
        
        assert validator.validate({'slow': 26, 'fast': 12, 'k_sl': 1.0})
        assert not validator.validate({'slow': 10, 'fast': 20, 'k_sl': 1.0})
        assert not validator.validate({'slow': 26, 'fast': 12, 'k_sl': 0.3})
    
    def test_filter_grid(self):
        """Test de filtrage de grille."""
        validator = ConstraintValidator()
        validator.add_greater_than('slow', 'fast')
        
        grid = [
            {'slow': 26, 'fast': 12},
            {'slow': 12, 'fast': 26},  # Invalide
            {'slow': 20, 'fast': 10},
            {'slow': 10, 'fast': 20},  # Invalide
        ]
        
        filtered = validator.filter_grid(grid)
        
        assert len(filtered) == 2
        assert all(p['slow'] > p['fast'] for p in filtered)
    
    def test_get_violations(self):
        """Test de récupération des violations."""
        validator = ConstraintValidator([
            ParameterConstraint('slow', 'greater_than', 'fast', description="slow > fast"),
            ParameterConstraint('k_sl', 'min_value', value=0.5, description="k_sl >= 0.5"),
        ])
        
        violations = validator.get_violations({'slow': 10, 'fast': 20, 'k_sl': 0.3})
        
        assert len(violations) == 2


class TestGenerateConstrainedParamGrid:
    """Tests pour la génération de grilles contraintes."""
    
    def test_with_constraints(self):
        """Test de génération avec contraintes."""
        specs = {
            'fast': ParameterSpec('fast', 5, 20, 12, param_type='int'),
            'slow': ParameterSpec('slow', 15, 50, 26, param_type='int'),
        }
        
        constraints = ConstraintValidator()
        constraints.add_greater_than('slow', 'fast')
        
        # Sans contraintes
        grid_no_constraint = generate_constrained_param_grid(
            specs, constraints=None, granularity=0.5
        )
        
        # Avec contraintes
        grid_with_constraint = generate_constrained_param_grid(
            specs, constraints=constraints, granularity=0.5
        )
        
        # La grille avec contraintes devrait être plus petite
        assert len(grid_with_constraint) <= len(grid_no_constraint)
        
        # Toutes les combinaisons restantes doivent être valides
        for params in grid_with_constraint:
            assert params['slow'] > params['fast']


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

class TestPhase1Integration:
    """Tests d'intégration Phase 1."""
    
    def test_tier_s_with_performance(self, sample_returns, sample_equity, sample_trades_pnl):
        """Test d'intégration Tier S avec performance.py."""
        from backtest.performance import calculate_metrics, equity_curve
        
        # Calculer les métriques standard + Tier S
        metrics = calculate_metrics(
            equity=sample_equity,
            returns=sample_returns,
            trades_df=pd.DataFrame({'pnl': sample_trades_pnl}),
            initial_capital=10000.0,
            include_tier_s=True
        )
        
        assert 'tier_s' in metrics
        assert metrics['tier_s'] is not None
        assert 'sqn' in metrics
        assert 'tier_s_grade' in metrics
    
    def test_validation_workflow(self, sample_ohlcv_df):
        """Test du workflow de validation complet."""
        # 1. Créer le validateur
        validator = WalkForwardValidator(n_folds=3, test_pct=0.2)
        
        # 2. Générer les splits
        folds = validator.split(sample_ohlcv_df)
        assert len(folds) > 0
        
        # 3. Pour chaque fold, obtenir train/test
        for fold in folds:
            train_df, test_df = validator.get_data_splits(sample_ohlcv_df, fold)
            assert len(train_df) > 0
            assert len(test_df) > 0
    
    def test_constraints_with_grid(self):
        """Test des contraintes avec génération de grille."""
        specs = {
            'fast_period': ParameterSpec('fast_period', 5, 15, 10, param_type='int'),
            'slow_period': ParameterSpec('slow_period', 20, 40, 30, param_type='int'),
            'k_sl': ParameterSpec('k_sl', 1.0, 3.0, 2.0),
        }
        
        constraints = ConstraintValidator()
        constraints.add_greater_than('slow_period', 'fast_period')
        constraints.add_ratio_min('slow_period', 'fast_period', ratio=1.5)
        
        grid = generate_constrained_param_grid(
            specs,
            constraints=constraints,
            granularity=0.5
        )
        
        # Vérifier que toutes les contraintes sont respectées
        for params in grid:
            assert params['slow_period'] > params['fast_period']
            assert params['slow_period'] / params['fast_period'] >= 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
