"""
Tests unitaires pour la correction du calcul du ratio de Sharpe.

Vérifie que le Sharpe ne sort plus toujours ±3.49 et varie selon les données.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.performance import sharpe_ratio, sortino_ratio, calculate_metrics


class TestSharpeFix:
    """Tests pour la correction du Sharpe ratio."""

    def test_sharpe_varies_with_different_returns(self):
        """Vérifie que le Sharpe varie quand on change les returns."""
        np.random.seed(42)

        # Générer 3 séries de returns différentes
        returns1 = pd.Series(np.random.normal(0.001, 0.01, 100))  # Bon rendement
        returns2 = pd.Series(np.random.normal(-0.001, 0.01, 100))  # Mauvais rendement
        returns3 = pd.Series(np.random.normal(0.0, 0.005, 100))  # Rendement neutre, moins volatile

        sharpe1 = sharpe_ratio(returns1, periods_per_year=252, method="standard")
        sharpe2 = sharpe_ratio(returns2, periods_per_year=252, method="standard")
        sharpe3 = sharpe_ratio(returns3, periods_per_year=252, method="standard")

        # Les Sharpe doivent être différents
        assert sharpe1 != sharpe2
        assert sharpe2 != sharpe3
        assert sharpe1 != sharpe3

        # Sharpe1 (bon rendement) doit être > Sharpe2 (mauvais rendement)
        assert sharpe1 > sharpe2

        # Aucun ne devrait être exactement ±3.49
        assert abs(sharpe1 - 3.49) > 0.1
        assert abs(sharpe2 - 3.49) > 0.1
        assert abs(sharpe3 - 3.49) > 0.1
        assert abs(sharpe1 + 3.49) > 0.1
        assert abs(sharpe2 + 3.49) > 0.1
        assert abs(sharpe3 + 3.49) > 0.1

    def test_sharpe_std_zero_returns_zero(self):
        """Vérifie que Sharpe = 0 quand std = 0 (returns constants)."""
        # Returns constants (volatilité nulle)
        returns_constant = pd.Series([0.01] * 100)

        sharpe = sharpe_ratio(returns_constant, periods_per_year=252, method="standard")

        # Sharpe devrait être 0.0 (convention), pas inf
        assert sharpe == 0.0

    def test_sharpe_empty_returns_zero(self):
        """Vérifie que Sharpe = 0 quand returns est vide."""
        returns_empty = pd.Series([], dtype=np.float64)

        sharpe = sharpe_ratio(returns_empty, periods_per_year=252, method="standard")

        assert sharpe == 0.0

    def test_sharpe_sparse_equity_daily_resample(self):
        """
        Test du cas problématique: equity qui ne change qu'aux trades.
        La méthode "daily_resample" devrait donner un résultat correct.
        """
        # Simuler equity sparse sur 30 jours
        n_days = 30
        dates = pd.date_range('2024-01-01', periods=n_days*24*60, freq='1min')
        equity_sparse = pd.Series(10000.0, index=dates)

        # Seulement 30 trades dispersés
        trade_indices = np.random.choice(range(100, len(dates)-100), size=30, replace=False)
        np.random.seed(42)
        for i in trade_indices:
            pnl = np.random.normal(50, 100)
            equity_sparse[i:] += pnl

        # Calculer returns
        returns_sparse = equity_sparse.pct_change().fillna(0)

        # Méthode daily_resample: devrait être raisonnable
        sharpe_daily = sharpe_ratio(
            returns_sparse,
            periods_per_year=252,
            method="daily_resample",
            equity=equity_sparse
        )

        # Le Sharpe devrait être dans une plage réaliste
        # Avec 30 jours de données, on peut avoir des valeurs un peu élevées
        # mais pas astronomiques comme avant
        assert -20 < sharpe_daily < 20, \
            f"Sharpe daily_resample hors plage: {sharpe_daily}"

        # Vérifier que ça ne plante pas
        assert not np.isnan(sharpe_daily)
        assert not np.isinf(sharpe_daily)

    def test_sharpe_sanity_check_random(self):
        """Sanity check: returns aléatoires petits => Sharpe pas ±3.49."""
        np.random.seed(123)

        # 10 runs avec returns aléatoires
        sharpes = []
        for _ in range(10):
            returns = pd.Series(np.random.normal(0, 0.001, 100))
            sharpe = sharpe_ratio(returns, periods_per_year=252, method="trading_days")
            sharpes.append(sharpe)

        # Vérifier que les Sharpe ne sont pas tous identiques
        unique_sharpes = len(set([round(s, 1) for s in sharpes]))
        assert unique_sharpes > 3, \
            f"Sharpe trop constant: {sharpes}"

        # Vérifier qu'aucun n'est exactement ±3.49
        for sharpe in sharpes:
            assert abs(sharpe - 3.49) > 0.05
            assert abs(sharpe + 3.49) > 0.05

    def test_periods_per_year_values(self):
        """Test des valeurs de periods_per_year pour différents timeframes."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))

        # Différents timeframes (ancienne méthode vs nouvelle)
        old_periods = {
            "1m": 365 * 24 * 60,    # 525600
            "15m": 365 * 24 * 4,    # 35040
            "1h": 365 * 24,         # 8760
            "1d": 365,              # 365
        }

        # Nouvelle méthode: toujours 252
        new_periods = 252

        sharpe_new = sharpe_ratio(returns, periods_per_year=new_periods, method="trading_days")

        # Le Sharpe avec 252 devrait être dans une plage raisonnable
        assert -5 < sharpe_new < 5

        # Vérifier que les anciennes valeurs donnaient des résultats très différents
        sharpe_1m = sharpe_ratio(returns, periods_per_year=old_periods["1m"], method="standard")

        # Le ratio devrait être proche de sqrt(525600/252) ≈ 45.6
        # (car sharpe ∝ sqrt(periods_per_year))
        ratio = abs(sharpe_1m / sharpe_new) if sharpe_new != 0 else 0
        expected_ratio = np.sqrt(old_periods["1m"] / new_periods)

        # Le ratio devrait être proche de la valeur attendue (±20%)
        if sharpe_new != 0:
            assert 0.8 * expected_ratio < ratio < 1.2 * expected_ratio

    def test_calculate_metrics_integration(self):
        """Test d'intégration: calculate_metrics avec daily_resample."""
        # Créer des données simulées sur 60 jours
        n_days = 60
        dates = pd.date_range('2024-01-01', periods=n_days*24*60, freq='1min')
        equity = pd.Series(10000.0, index=dates)

        # Simuler quelques trades
        np.random.seed(42)
        trade_indices = np.random.choice(range(100, len(dates)-100), size=30, replace=False)
        for i in trade_indices:
            pnl = np.random.normal(100, 50)
            equity[i:] += pnl

        returns = equity.pct_change().fillna(0)

        # Créer trades_df
        trades_df = pd.DataFrame({
            'pnl': [np.random.normal(100, 50) for _ in range(30)],
            'entry_ts': pd.date_range('2024-01-01', periods=30, freq='2d'),
            'exit_ts': pd.date_range('2024-01-01 01:00', periods=30, freq='2d')
        })

        # Calculer métriques avec daily_resample
        metrics = calculate_metrics(
            equity=equity,
            returns=returns,
            trades_df=trades_df,
            initial_capital=10000.0,
            periods_per_year=252,
            sharpe_method="daily_resample"
        )

        # Vérifier que les métriques sont calculées
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "total_pnl" in metrics

        # Sharpe devrait être dans une plage raisonnable
        sharpe = metrics["sharpe_ratio"]
        # Avec 60 jours de données, on peut avoir des valeurs variées
        assert -30 < sharpe < 30, f"Sharpe hors plage: {sharpe}"
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_sortino_method_parameter(self):
        """Test que sortino_ratio accepte le paramètre method."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1d')
        equity = pd.Series(10000 + np.cumsum(np.random.normal(10, 50, 100)), index=dates)
        returns = equity.pct_change().dropna()

        sortino_std = sortino_ratio(returns, periods_per_year=252, method="standard")
        sortino_daily = sortino_ratio(returns, periods_per_year=252, method="daily_resample", equity=equity)

        # Ne devrait pas planter
        assert not np.isnan(sortino_std)
        assert not np.isnan(sortino_daily)

    def test_sharpe_negative_returns(self):
        """Test que Sharpe est négatif pour returns majoritairement négatifs."""
        # Equity décroissante
        dates = pd.date_range('2024-01-01', periods=100, freq='1d')
        equity_neg = pd.Series(10000 - np.cumsum(np.abs(np.random.normal(50, 20, 100))), index=dates)
        returns_neg = equity_neg.pct_change().dropna()

        sharpe = sharpe_ratio(returns_neg, periods_per_year=252, method="daily_resample", equity=equity_neg)

        # Sharpe devrait être négatif
        assert sharpe < 0, f"Sharpe devrait être négatif: {sharpe}"

    def test_sharpe_few_days(self):
        """Test avec très peu de jours de données."""
        # Seulement 3 jours de données
        dates = pd.date_range('2024-01-01', periods=3*24*60, freq='1min')
        equity = pd.Series([10000, 10100, 10050], index=dates[::24*60])
        returns = equity.pct_change().dropna()

        sharpe = sharpe_ratio(returns, periods_per_year=252, method="daily_resample", equity=equity)

        # Avec seulement 2 returns, devrait être 0 ou une valeur calculée
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_sharpe_all_zero_returns(self):
        """Test avec tous les returns à zéro."""
        returns = pd.Series([0.0] * 100)

        sharpe_td = sharpe_ratio(returns, periods_per_year=252, method="trading_days")
        sharpe_std = sharpe_ratio(returns, periods_per_year=252, method="standard")

        # Devrait retourner 0.0 (std = 0 ou < 2 observations)
        assert sharpe_td == 0.0
        assert sharpe_std == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
