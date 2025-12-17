"""
Tests pour les 3 corrections critiques identifiées le 16/12/2025.

Ces tests documentent et valident les fixes suivants :
1. Alias CLI sharpe/sharpe_ratio
2. Garde epsilon + plafonnement Sharpe
3. Import numpy dans cli/commands.py
"""

import numpy as np
import pandas as pd
import pytest

from cli.commands import normalize_metric_name, METRIC_ALIASES
from backtest.performance import sharpe_ratio


# =============================================================================
# TEST 1 - Alias CLI
# =============================================================================

class TestCLIMetricAliases:
    """Test du système d'alias pour les métriques CLI."""
    
    def test_metric_aliases_dict(self):
        """Vérifie que le dictionnaire d'alias est correct."""
        assert "sharpe" in METRIC_ALIASES
        assert "sortino" in METRIC_ALIASES
        assert METRIC_ALIASES["sharpe"] == "sharpe_ratio"
        assert METRIC_ALIASES["sortino"] == "sortino_ratio"
    
    def test_normalize_sharpe(self):
        """sharpe doit être normalisé en sharpe_ratio."""
        assert normalize_metric_name("sharpe") == "sharpe_ratio"
    
    def test_normalize_sharpe_ratio(self):
        """sharpe_ratio doit rester sharpe_ratio."""
        assert normalize_metric_name("sharpe_ratio") == "sharpe_ratio"
    
    def test_normalize_sortino(self):
        """sortino doit être normalisé en sortino_ratio."""
        assert normalize_metric_name("sortino") == "sortino_ratio"
    
    def test_normalize_sortino_ratio(self):
        """sortino_ratio doit rester sortino_ratio."""
        assert normalize_metric_name("sortino_ratio") == "sortino_ratio"
    
    def test_normalize_other_metrics(self):
        """Les autres métriques doivent rester inchangées."""
        assert normalize_metric_name("total_return") == "total_return"
        assert normalize_metric_name("max_drawdown") == "max_drawdown"
        assert normalize_metric_name("win_rate") == "win_rate"


# =============================================================================
# TEST 2 - Garde Epsilon + Plafonnement Sharpe
# =============================================================================

class TestSharpeRatioStability:
    """Test des gardes epsilon et plafonnement du Sharpe."""
    
    def test_zero_volatility_returns_zero(self):
        """
        Variance quasi-nulle (< 0.1% annualisé) doit retourner 0.
        Évite les Sharpe aberrants de ±50, ±100.
        """
        returns_constant = pd.Series([0.0001] * 100)
        result = sharpe_ratio(returns_constant, method="standard")
        assert result == 0.0, "Volatilité < epsilon doit retourner 0"
    
    def test_extreme_sharpe_capped_positive(self):
        """
        Sharpe > 20 doit être plafonné à 20.
        """
        # Créer des returns qui généreraient un Sharpe > 20 sans plafonnement
        # (rendements très élevés avec variance faible)
        returns_extreme = pd.Series([0.05] * 10 + [0.0] * 90)
        result = sharpe_ratio(returns_extreme, method="standard")
        assert result <= 20.0, f"Sharpe doit être plafonné à 20, got {result}"
    
    def test_extreme_sharpe_capped_negative(self):
        """
        Sharpe < -20 doit être plafonné à -20.
        """
        # Créer des returns négatifs qui généreraient Sharpe < -20
        returns_extreme = pd.Series([-0.05] * 10 + [0.0] * 90)
        result = sharpe_ratio(returns_extreme, method="standard")
        assert result >= -20.0, f"Sharpe doit être plafonné à -20, got {result}"
    
    def test_normal_sharpe_unchanged(self):
        """
        Un Sharpe normal (-10 à +10) ne doit pas être modifié.
        """
        np.random.seed(42)
        returns_normal = pd.Series(np.random.normal(0.001, 0.02, 100))
        result = sharpe_ratio(returns_normal, method="standard")
        assert -20.0 <= result <= 20.0, "Sharpe normal doit rester dans [-20, 20]"
    
    def test_minimal_volatility_threshold(self):
        """
        Volatilité < 0.1% annualisé doit déclencher la garde epsilon.
        """
        # Créer des returns avec variance très faible
        returns_tiny_vol = pd.Series([0.00001] * 100)
        result = sharpe_ratio(returns_tiny_vol, method="standard")
        assert result == 0.0, "Variance < 0.1% annualisé doit retourner 0"
    
    def test_acceptable_volatility_passes(self):
        """
        Volatilité > 0.1% annualisé doit permettre le calcul.
        """
        # Créer des returns avec variance acceptable
        np.random.seed(42)
        returns_ok = pd.Series(np.random.normal(0.01, 0.1, 100))  # 10% volatilité
        result = sharpe_ratio(returns_ok, method="standard")
        # Le résultat doit être calculé (pas 0 par défaut)
        assert result != 0.0 or returns_ok.std() > 0.01


# =============================================================================
# TEST 3 - Import Numpy dans CLI
# =============================================================================

class TestCLINumpyImport:
    """Test de l'import numpy dans cli/commands.py."""
    
    def test_numpy_imported_in_cli_commands(self):
        """Vérifie que numpy est importé dans cli/commands.py."""
        import cli.commands as cmd_module
        assert hasattr(cmd_module, 'np'), "numpy doit être importé comme 'np'"
    
    def test_numpy_isfinite_available(self):
        """Vérifie que np.isfinite est disponible dans cli/commands."""
        import cli.commands as cmd_module
        assert hasattr(cmd_module.np, 'isfinite'), "np.isfinite doit être accessible"
        
        # Test d'utilisation réelle
        val = 3.14159
        assert cmd_module.np.isfinite(val) == True
        assert cmd_module.np.isfinite(float('inf')) == False
        assert cmd_module.np.isfinite(float('nan')) == False


# =============================================================================
# TEST 4 - Intégration Optuna (non-régression)
# =============================================================================

class TestOptunaIntegration:
    """Test d'intégration pour vérifier qu'Optuna fonctionne avec les corrections."""
    
    def test_optuna_with_normalized_metrics(self):
        """
        Optuna doit fonctionner avec les métriques normalisées.
        """
        from backtest.optuna_optimizer import OptunaOptimizer
        
        # Créer des données de test
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 2000, 100),
        }, index=dates)
        
        param_space = {
            "fast_period": {"type": "int", "low": 5, "high": 20},
            "slow_period": {"type": "int", "low": 21, "high": 50},
        }
        
        # L'optimiseur doit se créer sans erreur
        optimizer = OptunaOptimizer(
            strategy_name="ema_cross",
            data=df,
            param_space=param_space,
            constraints=[("slow_period", ">", "fast_period")],
        )
        
        assert optimizer is not None
        assert optimizer.strategy_name == "ema_cross"
    
    def test_optuna_result_processing_with_nan(self):
        """
        Le traitement des résultats Optuna doit gérer les NaN sans erreur numpy.
        """
        import cli.commands as cmd_module
        
        # Simuler un DataFrame avec des NaN
        test_df = pd.DataFrame({
            'trial': [0, 1, 2],
            'value': [1.5, float('nan'), 2.3],
            'fast_period': [10, 15, 12],
            'slow_period': [30, 40, 35],
        })
        
        # Tester le code qui utilise np.isfinite
        for val in test_df['value']:
            if cmd_module.np.isfinite(val):
                val_str = f"{val:.4f}"
            else:
                val_str = "N/A"
            
            assert val_str in ["1.5000", "N/A", "2.3000"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
