"""
Tests pour le module de visualisation.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_ohlcv():
    """Génère des données OHLCV de test."""
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Générer des prix réalistes
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * (1 + np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = (high + low) / 2 + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


@pytest.fixture
def sample_trades():
    """Génère des trades de test."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    return [
        {
            'entry_ts': base_time + timedelta(hours=5),
            'exit_ts': base_time + timedelta(hours=15),
            'side': 'LONG',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'price_entry': 100.0,
            'price_exit': 105.0,
            'size': 1.0,
            'pnl': 5.0,
            'return_pct': 0.05,
            'exit_reason': 'signal',
        },
        {
            'entry_ts': base_time + timedelta(hours=20),
            'exit_ts': base_time + timedelta(hours=30),
            'side': 'SHORT',
            'entry_price': 103.0,
            'exit_price': 100.0,
            'price_entry': 103.0,
            'price_exit': 100.0,
            'size': 1.0,
            'pnl': 3.0,
            'return_pct': 0.03,
            'exit_reason': 'signal',
        },
        {
            'entry_ts': base_time + timedelta(hours=35),
            'exit_ts': base_time + timedelta(hours=45),
            'side': 'LONG',
            'entry_price': 102.0,
            'exit_price': 98.0,
            'price_entry': 102.0,
            'price_exit': 98.0,
            'size': 1.0,
            'pnl': -4.0,
            'return_pct': -0.04,
            'exit_reason': 'stop_loss',
        },
    ]


@pytest.fixture
def sample_metrics():
    """Métriques de test."""
    return {
        'pnl': 4.0,
        'total_return': 0.04,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'max_drawdown': -5.0,
        'win_rate': 0.67,
        'num_trades': 3,
        'profit_factor': 2.0,
        'initial_capital': 10000,
    }


@pytest.fixture
def sample_equity_curve():
    """Equity curve de test."""
    equity = [10000]
    for _ in range(99):
        equity.append(equity[-1] * (1 + np.random.randn() * 0.01))
    return equity


# ==============================================================================
# TESTS D'IMPORT
# ==============================================================================

class TestImports:
    """Tests d'import du module."""
    
    def test_import_visualization(self):
        """Test import principal."""
        from utils import visualization
        assert visualization is not None
    
    def test_import_functions(self):
        """Test import des fonctions."""
        from utils.visualization import (
            plot_trades,
            plot_equity_curve,
            plot_drawdown,
            visualize_backtest,
            load_and_visualize,
        )
        assert callable(plot_trades)
        assert callable(plot_equity_curve)
        assert callable(plot_drawdown)
        assert callable(visualize_backtest)
        assert callable(load_and_visualize)
    
    def test_import_classes(self):
        """Test import des classes."""
        from utils.visualization import TradeMarker, BacktestVisualData
        assert TradeMarker is not None
        assert BacktestVisualData is not None
    
    def test_plotly_available(self):
        """Test détection Plotly."""
        from utils.visualization import PLOTLY_AVAILABLE
        assert isinstance(PLOTLY_AVAILABLE, bool)


# ==============================================================================
# TESTS DES FONCTIONS DE PLOT
# ==============================================================================

class TestPlotTrades:
    """Tests pour plot_trades."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_basic_plot(self, sample_ohlcv, sample_trades):
        """Test création graphique basique."""
        from utils.visualization import plot_trades
        
        fig = plot_trades(sample_ohlcv, sample_trades)
        
        assert fig is not None
        assert len(fig.data) > 0  # Au moins le candlestick
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_plot_without_volume(self, sample_ohlcv, sample_trades):
        """Test sans volume."""
        from utils.visualization import plot_trades
        
        fig = plot_trades(sample_ohlcv, sample_trades, show_volume=False)
        assert fig is not None
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_plot_empty_trades(self, sample_ohlcv):
        """Test avec liste de trades vide."""
        from utils.visualization import plot_trades
        
        fig = plot_trades(sample_ohlcv, [])
        assert fig is not None
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_plot_max_candles(self, sample_trades):
        """Test limitation du nombre de bougies."""
        from utils.visualization import plot_trades
        
        # Créer un grand DataFrame
        n = 5000
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            'open': np.random.randn(n) + 100,
            'high': np.random.randn(n) + 101,
            'low': np.random.randn(n) + 99,
            'close': np.random.randn(n) + 100,
            'volume': np.random.randint(1000, 10000, n),
        }, index=dates)
        
        fig = plot_trades(df, [], max_candles=1000)
        assert fig is not None


class TestPlotEquityCurve:
    """Tests pour plot_equity_curve."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_basic_equity_curve(self, sample_equity_curve):
        """Test equity curve basique."""
        from utils.visualization import plot_equity_curve
        
        fig = plot_equity_curve(sample_equity_curve)
        assert fig is not None
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_equity_curve_with_capital(self, sample_equity_curve):
        """Test avec capital personnalisé."""
        from utils.visualization import plot_equity_curve
        
        fig = plot_equity_curve(sample_equity_curve, initial_capital=50000)
        assert fig is not None


class TestPlotDrawdown:
    """Tests pour plot_drawdown."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_basic_drawdown(self, sample_equity_curve):
        """Test drawdown basique."""
        from utils.visualization import plot_drawdown
        
        fig = plot_drawdown(sample_equity_curve)
        assert fig is not None


# ==============================================================================
# TESTS DES HELPERS HTML
# ==============================================================================

class TestHTMLHelpers:
    """Tests pour les helpers HTML."""
    
    def test_performance_cards(self, sample_metrics):
        """Test génération des cartes de performance."""
        from utils.visualization import create_performance_cards
        
        html = create_performance_cards(sample_metrics)
        
        assert isinstance(html, str)
        assert 'PnL' in html
        assert 'Sharpe' in html
        assert 'Max Drawdown' in html
    
    def test_trades_table(self, sample_trades):
        """Test génération de la table des trades."""
        from utils.visualization import create_trades_table
        
        html = create_trades_table(sample_trades)
        
        assert isinstance(html, str)
        assert 'LONG' in html
        assert 'SHORT' in html
        assert 'table' in html.lower()
    
    def test_trades_table_max_rows(self, sample_trades):
        """Test limitation des lignes."""
        from utils.visualization import create_trades_table
        
        # Dupliquer les trades
        many_trades = sample_trades * 20
        html = create_trades_table(many_trades, max_rows=10)
        
        assert 'et' in html  # "... et X trades de plus"


# ==============================================================================
# TESTS DE VISUALISATION COMPLÈTE
# ==============================================================================

class TestVisualizeBacktest:
    """Tests pour visualize_backtest."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_visualize_without_output(self, sample_ohlcv, sample_trades, sample_metrics):
        """Test visualisation sans export."""
        from utils.visualization import visualize_backtest
        
        figures = visualize_backtest(
            df=sample_ohlcv,
            trades=sample_trades,
            metrics=sample_metrics,
            show=False,
        )
        
        assert 'trades' in figures
        assert figures['trades'] is not None
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_visualize_with_equity(self, sample_ohlcv, sample_trades, sample_metrics, sample_equity_curve):
        """Test avec equity curve."""
        from utils.visualization import visualize_backtest
        
        figures = visualize_backtest(
            df=sample_ohlcv,
            trades=sample_trades,
            metrics=sample_metrics,
            equity_curve=sample_equity_curve,
            show=False,
        )
        
        assert 'trades' in figures
        assert 'equity' in figures
        assert 'drawdown' in figures
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_visualize_with_output(
        self, sample_ohlcv, sample_trades, sample_metrics, sample_equity_curve, tmp_path
    ):
        """Test avec export HTML."""
        from utils.visualization import visualize_backtest
        
        output_path = tmp_path / "report.html"
        
        figures = visualize_backtest(
            df=sample_ohlcv,
            trades=sample_trades,
            metrics=sample_metrics,
            equity_curve=sample_equity_curve,
            output_path=output_path,
            show=False,
        )
        
        assert output_path.exists()
        content = output_path.read_text()
        assert 'Backtest' in content
        assert 'Plotly' in content


# ==============================================================================
# TESTS LOAD AND VISUALIZE
# ==============================================================================

class TestLoadAndVisualize:
    """Tests pour load_and_visualize."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_load_backtest_results(self, sample_trades, sample_metrics, tmp_path):
        """Test chargement et visualisation."""
        from utils.visualization import load_and_visualize
        
        # Créer fichier de résultats
        results = {
            'strategy': 'test_strategy',
            'params': {'param1': 10},
            'trades': sample_trades,
            'metrics': sample_metrics,
        }
        
        results_path = tmp_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, default=str)
        
        # Charger et visualiser (sans données OHLCV)
        result = load_and_visualize(
            results_path=results_path,
            show=False,
        )
        
        assert result is not None


# ==============================================================================
# TESTS DES DATACLASSES
# ==============================================================================

class TestDataclasses:
    """Tests pour les dataclasses."""
    
    def test_trade_marker(self):
        """Test TradeMarker."""
        from utils.visualization import TradeMarker
        
        marker = TradeMarker(
            timestamp=pd.Timestamp("2024-01-01"),
            price=100.0,
            side="LONG",
            action="entry",
            pnl=5.0,
            trade_id=1,
        )
        
        assert marker.timestamp == pd.Timestamp("2024-01-01")
        assert marker.price == 100.0
        assert marker.side == "LONG"
    
    def test_backtest_visual_data(self, sample_ohlcv, sample_trades):
        """Test BacktestVisualData."""
        from utils.visualization import BacktestVisualData
        
        data = BacktestVisualData(
            ohlcv=sample_ohlcv,
            trades=sample_trades,
            strategy_name="test",
            symbol="BTCUSDT",
        )
        
        assert len(data.ohlcv) == len(sample_ohlcv)
        assert len(data.trades) == len(sample_trades)
        assert data.strategy_name == "test"


# ==============================================================================
# TESTS EDGE CASES
# ==============================================================================

class TestEdgeCases:
    """Tests pour les cas limites."""
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        from utils.visualization import plot_trades
        
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        fig = plot_trades(df, [])
        assert fig is not None
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_single_candle(self):
        """Test avec une seule bougie."""
        from utils.visualization import plot_trades
        
        df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000],
        }, index=[pd.Timestamp("2024-01-01")])
        
        fig = plot_trades(df, [])
        assert fig is not None
    
    def test_metrics_with_missing_values(self):
        """Test métriques avec valeurs manquantes."""
        from utils.visualization import create_performance_cards
        
        metrics = {'pnl': 100}  # Minimal
        html = create_performance_cards(metrics)
        assert 'PnL' in html
    
    @pytest.mark.skipif(
        not __import__('utils.visualization', fromlist=['PLOTLY_AVAILABLE']).PLOTLY_AVAILABLE,
        reason="Plotly non disponible"
    )
    def test_trades_outside_range(self, sample_ohlcv):
        """Test trades hors de la plage de données."""
        from utils.visualization import plot_trades
        
        # Trade avant la plage de données
        trades = [{
            'entry_ts': pd.Timestamp("2020-01-01"),
            'exit_ts': pd.Timestamp("2020-01-02"),
            'side': 'LONG',
            'entry_price': 50,
            'exit_price': 55,
            'pnl': 5,
        }]
        
        fig = plot_trades(sample_ohlcv, trades)
        assert fig is not None
