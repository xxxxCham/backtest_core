"""
Tests unitaires complets pour la fonction sharpe_ratio.

Couvre:
- Cas normaux avec différentes distributions
- Zero variance (std == 0)
- Différents timeframes (1m, 15m, 1h, 1d)
- Aberrations statistiques
- Numpy arrays et pandas Series
- Méthodes: standard, trading_days, daily_resample
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from backtest.performance import sharpe_ratio


class TestSharpeRatioNormal:
    """Tests avec des cas normaux."""
    
    def test_positive_returns(self):
        """Sharpe positif avec rendements positifs."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.03, 0.01])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe > 0
        assert np.isfinite(sharpe)
    
    def test_negative_returns(self):
        """Sharpe négatif avec rendements négatifs."""
        returns = pd.Series([-0.01, -0.02, -0.015, -0.03, -0.01])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe < 0
        assert np.isfinite(sharpe)
    
    def test_mixed_returns(self):
        """Rendements mixtes (positifs et négatifs)."""
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.02])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_with_risk_free_rate(self):
        """Sharpe avec taux sans risque."""
        returns = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02])
        sharpe_no_rf = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe_with_rf = sharpe_ratio(returns, risk_free=0.02, periods_per_year=252, method="standard")
        
        # Avec taux sans risque, le Sharpe doit être plus bas
        assert sharpe_with_rf < sharpe_no_rf


class TestSharpeRatioZeroVariance:
    """Tests avec variance nulle (std == 0)."""
    
    def test_all_zeros(self):
        """Tous les rendements à zéro."""
        returns = pd.Series(np.zeros(100))
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_constant_positive(self):
        """Rendements constants positifs."""
        returns = pd.Series(np.ones(100) * 0.01)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_constant_negative(self):
        """Rendements constants négatifs."""
        returns = pd.Series(np.ones(100) * -0.01)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_numpy_array_zeros(self):
        """Numpy array avec zéros."""
        returns = np.zeros(100)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0


class TestSharpeRatioTimeframes:
    """Tests avec différents timeframes."""
    
    def test_1_minute_timeframe(self):
        """Timeframe 1 minute (525600 périodes/an)."""
        returns = pd.Series(np.random.normal(0.0001, 0.001, 1000))
        periods_per_year = 365 * 24 * 60  # 525600
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year, method="standard")
        assert np.isfinite(sharpe)
        assert abs(sharpe) < 1000  # Sanity check
    
    def test_15_minute_timeframe(self):
        """Timeframe 15 minutes (35040 périodes/an)."""
        returns = pd.Series(np.random.normal(0.0005, 0.005, 1000))
        periods_per_year = 365 * 24 * 4  # 35040
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year, method="standard")
        assert np.isfinite(sharpe)
        assert abs(sharpe) < 1000
    
    def test_1_hour_timeframe(self):
        """Timeframe 1 heure (8760 périodes/an)."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        periods_per_year = 365 * 24  # 8760
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year, method="standard")
        assert np.isfinite(sharpe)
        assert abs(sharpe) < 500
    
    def test_daily_timeframe(self):
        """Timeframe quotidien (252 périodes/an)."""
        returns = pd.Series(np.random.normal(0.005, 0.02, 250))
        periods_per_year = 252
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year, method="standard")
        assert np.isfinite(sharpe)
        assert abs(sharpe) < 100
    
    def test_weekly_timeframe(self):
        """Timeframe hebdomadaire (52 périodes/an)."""
        returns = pd.Series(np.random.normal(0.01, 0.05, 100))
        periods_per_year = 52
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year, method="standard")
        assert np.isfinite(sharpe)


class TestSharpeRatioAberrations:
    """Tests avec aberrations statistiques."""
    
    def test_single_outlier_positive(self):
        """Un outlier positif extrême."""
        returns = pd.Series([0.01] * 99 + [10.0])  # +1000%
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
        assert sharpe > 0
    
    def test_single_outlier_negative(self):
        """Un outlier négatif extrême."""
        returns = pd.Series([0.01] * 99 + [-0.99])  # -99%
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_alternating_extreme_values(self):
        """Valeurs extrêmes alternées."""
        returns = pd.Series([0.5, -0.4, 0.6, -0.5, 0.4, -0.3] * 10)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_highly_skewed_distribution(self):
        """Distribution très asymétrique."""
        # Distribution log-normale (fortement asymétrique)
        returns = pd.Series(np.random.lognormal(0, 0.5, 1000) - 1)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_very_small_returns(self):
        """Rendements très petits (proche de la précision machine)."""
        returns = pd.Series(np.random.normal(0, 1e-12, 1000))
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        # Devrait retourner 0 car std trop faible (seuil: 1e-10)
        assert sharpe == 0.0
    
    def test_very_large_returns(self):
        """Rendements très grands."""
        returns = pd.Series(np.random.normal(5.0, 10.0, 1000))
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)


class TestSharpeRatioEdgeCases:
    """Tests des cas limites."""
    
    def test_empty_series(self):
        """Série vide."""
        returns = pd.Series([])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_single_value(self):
        """Une seule valeur (< 2 observations)."""
        returns = pd.Series([0.01])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_two_values(self):
        """Deux valeurs (minimum requis)."""
        returns = pd.Series([0.01, 0.02])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_all_nan(self):
        """Tous NaN."""
        returns = pd.Series([np.nan] * 100)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert sharpe == 0.0
    
    def test_some_nan(self):
        """Quelques NaN mélangés."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, 0.015, 0.03])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
        assert sharpe > 0


class TestSharpeRatioInputTypes:
    """Tests avec différents types d'entrée."""
    
    def test_numpy_array_normal(self):
        """Numpy array standard."""
        returns = np.random.normal(0.001, 0.01, 1000)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_numpy_array_with_nan(self):
        """Numpy array avec NaN."""
        returns = np.array([0.01, np.nan, 0.02, 0.015, np.nan, 0.03])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_pandas_series_with_index(self):
        """Pandas Series avec index personnalisé."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)


class TestSharpeRatioMethods:
    """Tests des différentes méthodes de calcul."""
    
    def test_method_standard(self):
        """Méthode standard."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        assert np.isfinite(sharpe)
    
    def test_method_trading_days(self):
        """Méthode trading_days (filtre les zéros)."""
        returns = pd.Series([0.01, 0.0, 0.02, 0.0, 0.015, 0.0, 0.03])
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="trading_days")
        assert np.isfinite(sharpe)
    
    def test_method_daily_resample_with_equity(self):
        """Méthode daily_resample avec equity."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        equity = pd.Series(np.cumsum(np.random.normal(0, 10, 100)) + 10000, index=dates)
        returns = equity.pct_change().dropna()
        
        sharpe = sharpe_ratio(
            returns, 
            risk_free=0.0, 
            periods_per_year=252,
            method="daily_resample",
            equity=equity
        )
        assert np.isfinite(sharpe)
    
    def test_method_daily_resample_without_equity_fallback(self):
        """daily_resample sans equity → fallback sur standard."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        sharpe = sharpe_ratio(
            returns, 
            risk_free=0.0, 
            periods_per_year=252,
            method="daily_resample",
            equity=None
        )
        assert np.isfinite(sharpe)
    
    def test_method_trading_days_all_zeros(self):
        """trading_days avec que des zéros → retourne 0."""
        returns = pd.Series(np.zeros(100))
        sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="trading_days")
        assert sharpe == 0.0


class TestSharpeRatioStatisticalProperties:
    """Tests des propriétés statistiques."""
    
    def test_higher_mean_higher_sharpe(self):
        """Rendement moyen plus élevé → Sharpe plus élevé (même volatilité)."""
        np.random.seed(42)
        returns1 = pd.Series(np.random.normal(0.001, 0.01, 1000))
        returns2 = pd.Series(np.random.normal(0.005, 0.01, 1000))
        
        sharpe1 = sharpe_ratio(returns1, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe2 = sharpe_ratio(returns2, risk_free=0.0, periods_per_year=252, method="standard")
        
        assert sharpe2 > sharpe1
    
    def test_lower_volatility_higher_sharpe(self):
        """Volatilité plus faible → Sharpe plus élevé (même rendement moyen)."""
        np.random.seed(42)
        returns1 = pd.Series(np.random.normal(0.005, 0.02, 1000))
        returns2 = pd.Series(np.random.normal(0.005, 0.01, 1000))
        
        sharpe1 = sharpe_ratio(returns1, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe2 = sharpe_ratio(returns2, risk_free=0.0, periods_per_year=252, method="standard")
        
        assert sharpe2 > sharpe1
    
    def test_symmetry_sign(self):
        """Inverser les signes → inverse le Sharpe."""
        returns = pd.Series(np.random.normal(0.005, 0.01, 1000))
        
        sharpe_pos = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe_neg = sharpe_ratio(-returns, risk_free=0.0, periods_per_year=252, method="standard")
        
        assert np.isclose(sharpe_pos, -sharpe_neg, rtol=0.01)
    
    def test_annualization_scale(self):
        """Différentes périodisations donnent des résultats cohérents."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        
        sharpe_daily = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe_hourly = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252*24, method="standard")
        
        # Les deux doivent être du même ordre de grandeur (annualisés différemment)
        assert np.isfinite(sharpe_daily)
        assert np.isfinite(sharpe_hourly)


class TestSharpeRatioReproducibility:
    """Tests de reproductibilité."""
    
    def test_same_input_same_output(self):
        """Même entrée → même sortie."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        
        sharpe1 = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe2 = sharpe_ratio(returns, risk_free=0.0, periods_per_year=252, method="standard")
        
        assert sharpe1 == sharpe2
    
    def test_copy_vs_original(self):
        """Copie vs original → même résultat."""
        returns_orig = pd.Series(np.random.normal(0.001, 0.01, 1000))
        returns_copy = returns_orig.copy()
        
        sharpe_orig = sharpe_ratio(returns_orig, risk_free=0.0, periods_per_year=252, method="standard")
        sharpe_copy = sharpe_ratio(returns_copy, risk_free=0.0, periods_per_year=252, method="standard")
        
        assert sharpe_orig == sharpe_copy
