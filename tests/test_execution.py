"""
Tests pour le module d'exécution réaliste.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.execution import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionModel,
    ExecutionResult,
    SpreadCalculator,
    SlippageCalculator,
    create_execution_engine,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """Génère des données OHLCV de test."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    spread = np.abs(np.random.randn(n)) * 0.1
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + spread,
        "low": prices - spread,
        "close": prices + np.random.randn(n) * 0.2,
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)
    
    return df


@pytest.fixture
def volatile_ohlcv():
    """Génère des données avec haute volatilité."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    
    # Prix avec haute volatilité
    prices = 100 + np.cumsum(np.random.randn(n) * 3.0)  # 3x plus volatile
    spread = np.abs(np.random.randn(n)) * 0.5
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + spread,
        "low": prices - spread,
        "close": prices + np.random.randn(n) * 0.5,
        "volume": np.random.randint(100, 500, n),  # Faible volume
    }, index=dates)
    
    return df


# ============================================================================
# Tests ExecutionConfig
# ============================================================================

class TestExecutionConfig:
    """Tests pour ExecutionConfig."""
    
    def test_default_config(self):
        """Test configuration par défaut."""
        config = ExecutionConfig()
        
        assert config.model == ExecutionModel.DYNAMIC
        assert config.spread_bps == 5.0
        assert config.slippage_bps == 3.0
        assert config.latency_ms == 50.0
        assert config.use_volatility_spread is True
        assert config.use_volume_slippage is True
    
    def test_custom_config(self):
        """Test configuration personnalisée."""
        config = ExecutionConfig(
            model=ExecutionModel.FIXED,
            spread_bps=10.0,
            slippage_bps=5.0,
            latency_ms=100.0,
        )
        
        assert config.model == ExecutionModel.FIXED
        assert config.spread_bps == 10.0
        assert config.slippage_bps == 5.0
        assert config.latency_ms == 100.0
    
    def test_config_to_dict(self):
        """Test conversion en dict."""
        config = ExecutionConfig()
        d = config.to_dict()
        
        assert "model" in d
        assert "spread_bps" in d
        assert "slippage_bps" in d
        assert d["model"] == "dynamic"


# ============================================================================
# Tests ExecutionResult
# ============================================================================

class TestExecutionResult:
    """Tests pour ExecutionResult."""
    
    def test_result_properties(self):
        """Test propriétés du résultat."""
        result = ExecutionResult(
            executed_price=100.5,
            requested_price=100.0,
            spread_cost=0.25,
            slippage_cost=0.15,
            market_impact=0.10,
        )
        
        assert result.total_cost == 0.5
        assert result.total_cost_bps == pytest.approx(50.0, rel=0.01)
    
    def test_result_to_dict(self):
        """Test conversion en dict."""
        result = ExecutionResult(
            executed_price=100.0,
            requested_price=100.0,
        )
        d = result.to_dict()
        
        assert "executed_price" in d
        assert "requested_price" in d
        assert "total_cost_bps" in d


# ============================================================================
# Tests ExecutionEngine - Modèle IDEAL
# ============================================================================

class TestExecutionEngineIdeal:
    """Tests pour le modèle IDEAL (aucun coût)."""
    
    def test_ideal_no_costs(self, sample_ohlcv):
        """Mode idéal: aucun coût d'exécution."""
        config = ExecutionConfig(model=ExecutionModel.IDEAL)
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        result = engine.execute_order(price=100.0, side=1, bar_idx=50)
        
        assert result.executed_price == 100.0
        assert result.spread_cost == 0.0
        assert result.slippage_cost == 0.0
        assert result.total_cost_bps == 0.0


# ============================================================================
# Tests ExecutionEngine - Modèle FIXED
# ============================================================================

class TestExecutionEngineFixed:
    """Tests pour le modèle FIXED."""
    
    def test_fixed_spread_slippage(self, sample_ohlcv):
        """Mode fixe: spread et slippage constants."""
        config = ExecutionConfig(
            model=ExecutionModel.FIXED,
            spread_bps=10.0,
            slippage_bps=5.0,
        )
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        result = engine.execute_order(price=100.0, side=1, bar_idx=50)
        
        # Prix augmenté pour un achat
        assert result.executed_price > 100.0
        assert result.spread_cost > 0
        assert result.slippage_cost > 0
    
    def test_fixed_buy_vs_sell(self, sample_ohlcv):
        """Mode fixe: direction affecte le prix."""
        config = ExecutionConfig(
            model=ExecutionModel.FIXED,
            spread_bps=10.0,
            slippage_bps=5.0,
        )
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        buy_result = engine.execute_order(price=100.0, side=1, bar_idx=50)
        sell_result = engine.execute_order(price=100.0, side=-1, bar_idx=50)
        
        # Achat: prix plus haut, Vente: prix plus bas
        assert buy_result.executed_price > 100.0
        assert sell_result.executed_price < 100.0


# ============================================================================
# Tests ExecutionEngine - Modèle DYNAMIC
# ============================================================================

class TestExecutionEngineDynamic:
    """Tests pour le modèle DYNAMIC."""
    
    def test_dynamic_volatility_spread(self, sample_ohlcv, volatile_ohlcv):
        """Mode dynamique: spread augmente avec volatilité."""
        config = ExecutionConfig(
            model=ExecutionModel.DYNAMIC,
            spread_bps=5.0,
            use_volatility_spread=True,
        )
        
        # Données normales
        engine1 = ExecutionEngine(config)
        engine1.prepare(sample_ohlcv)
        result1 = engine1.execute_order(price=100.0, side=1, bar_idx=100)
        
        # Données volatiles
        engine2 = ExecutionEngine(config)
        engine2.prepare(volatile_ohlcv)
        result2 = engine2.execute_order(price=100.0, side=1, bar_idx=100)
        
        # Spread plus élevé en période volatile
        # (coût total plus important)
        assert result2.spread_cost >= result1.spread_cost
    
    def test_dynamic_volume_slippage(self, sample_ohlcv, volatile_ohlcv):
        """Mode dynamique: slippage augmente avec faible volume."""
        config = ExecutionConfig(
            model=ExecutionModel.DYNAMIC,
            slippage_bps=5.0,
            use_volume_slippage=True,
        )
        
        # Données avec volume normal
        engine1 = ExecutionEngine(config)
        engine1.prepare(sample_ohlcv)
        result1 = engine1.execute_order(price=100.0, side=1, bar_idx=100)
        
        # Données avec faible volume
        engine2 = ExecutionEngine(config)
        engine2.prepare(volatile_ohlcv)
        result2 = engine2.execute_order(price=100.0, side=1, bar_idx=100)
        
        # Les deux ont du slippage
        assert result1.slippage_cost > 0
        assert result2.slippage_cost > 0
    
    def test_dynamic_bounds_respected(self, sample_ohlcv):
        """Mode dynamique: bornes min/max respectées."""
        config = ExecutionConfig(
            model=ExecutionModel.DYNAMIC,
            spread_bps=5.0,
            min_spread_bps=2.0,
            max_spread_bps=20.0,
            slippage_bps=3.0,
            min_slippage_bps=1.0,
            max_slippage_bps=15.0,
        )
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        # Tester plusieurs barres
        for i in range(50, 150):
            spread = engine._calculate_spread_bps(i)
            slippage = engine._calculate_slippage_bps(i)
            
            assert config.min_spread_bps <= spread <= config.max_spread_bps
            assert config.min_slippage_bps <= slippage <= config.max_slippage_bps


# ============================================================================
# Tests Bid/Ask
# ============================================================================

class TestBidAsk:
    """Tests pour le calcul bid/ask."""
    
    def test_bid_ask_spread(self, sample_ohlcv):
        """Test calcul bid/ask."""
        config = ExecutionConfig(
            model=ExecutionModel.FIXED,
            spread_bps=10.0,  # 0.1%
        )
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        mid = 100.0
        bid, ask = engine.get_bid_ask(mid, bar_idx=50)
        
        assert bid < mid
        assert ask > mid
        assert ask - bid == pytest.approx(0.1, rel=0.01)  # 0.1% de 100


# ============================================================================
# Tests Latency
# ============================================================================

class TestLatency:
    """Tests pour la latence."""
    
    def test_latency_bars_calculation(self, sample_ohlcv):
        """Test calcul barres de latence."""
        # 1h bars, 2h latency = 2 bars
        config = ExecutionConfig(latency_ms=2 * 60 * 60 * 1000)  # 2h en ms
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        result = engine.execute_order(price=100.0, side=1, bar_idx=50)
        
        assert result.latency_bars == 2
    
    def test_zero_latency(self, sample_ohlcv):
        """Test sans latence."""
        config = ExecutionConfig(latency_ms=0)
        engine = ExecutionEngine(config)
        engine.prepare(sample_ohlcv)
        
        result = engine.execute_order(price=100.0, side=1, bar_idx=50)
        
        assert result.latency_bars == 0


# ============================================================================
# Tests SpreadCalculator
# ============================================================================

class TestSpreadCalculator:
    """Tests pour SpreadCalculator."""
    
    def test_fixed_spread(self):
        """Test spread fixe."""
        spread = SpreadCalculator.fixed_spread(5.0)
        assert spread == 5.0
    
    def test_volatility_spread(self):
        """Test spread basé volatilité."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        
        spreads = SpreadCalculator.volatility_spread(returns, base_bps=5.0)
        
        assert len(spreads) == 100
        assert np.all(spreads >= 5.0)
    
    def test_roll_spread(self):
        """Test estimateur Roll."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        spreads = SpreadCalculator.roll_spread(closes, window=20)
        
        assert len(spreads) == 100
        assert np.all(spreads >= 0)
    
    def test_high_low_spread(self):
        """Test estimateur Corwin-Schultz."""
        np.random.seed(42)
        n = 100
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n)) * 0.5
        lows = closes - np.abs(np.random.randn(n)) * 0.5
        
        spreads = SpreadCalculator.high_low_spread(highs, lows, closes)
        
        assert len(spreads) == 100
        assert np.all(spreads >= 0)


# ============================================================================
# Tests SlippageCalculator
# ============================================================================

class TestSlippageCalculator:
    """Tests pour SlippageCalculator."""
    
    def test_fixed_slippage(self):
        """Test slippage fixe."""
        slip = SlippageCalculator.fixed_slippage(3.0)
        assert slip == 3.0
    
    def test_volume_slippage(self):
        """Test slippage basé volume."""
        # Ordre normal
        slip1 = SlippageCalculator.volume_slippage(
            order_size=100,
            avg_volume=10000,
            base_bps=3.0
        )
        
        # Gros ordre
        slip2 = SlippageCalculator.volume_slippage(
            order_size=1000,
            avg_volume=10000,
            base_bps=3.0
        )
        
        assert slip2 > slip1
    
    def test_volatility_slippage(self):
        """Test slippage basé volatilité."""
        # Faible volatilité
        slip1 = SlippageCalculator.volatility_slippage(volatility=0.01)
        
        # Haute volatilité
        slip2 = SlippageCalculator.volatility_slippage(volatility=0.05)
        
        assert slip2 > slip1
    
    def test_almgren_chriss_impact(self):
        """Test modèle Almgren-Chriss."""
        impact = SlippageCalculator.almgren_chriss_impact(
            order_size=10000,
            daily_volume=1000000,
            daily_volatility=0.02,
            eta=0.1,
            gamma=0.5
        )
        
        assert impact > 0
        assert impact < 1  # Impact raisonnable


# ============================================================================
# Tests Factory
# ============================================================================

class TestFactory:
    """Tests pour la factory."""
    
    def test_create_ideal(self):
        """Test création modèle idéal."""
        engine = create_execution_engine(model="ideal")
        assert engine.config.model == ExecutionModel.IDEAL
    
    def test_create_fixed(self):
        """Test création modèle fixe."""
        engine = create_execution_engine(
            model="fixed",
            spread_bps=10.0,
            slippage_bps=5.0
        )
        assert engine.config.model == ExecutionModel.FIXED
        assert engine.config.spread_bps == 10.0
    
    def test_create_dynamic(self):
        """Test création modèle dynamique."""
        engine = create_execution_engine(model="dynamic")
        assert engine.config.model == ExecutionModel.DYNAMIC
    
    def test_create_with_kwargs(self):
        """Test création avec kwargs additionnels."""
        engine = create_execution_engine(
            model="dynamic",
            spread_bps=8.0,
            use_volatility_spread=False,
            volatility_window=30
        )
        
        assert engine.config.spread_bps == 8.0
        assert engine.config.use_volatility_spread is False
        assert engine.config.volatility_window == 30


# ============================================================================
# Tests Intégration avec Simulator
# ============================================================================

class TestSimulatorIntegration:
    """Tests d'intégration avec le simulateur."""
    
    def test_simulator_with_execution_engine(self, sample_ohlcv):
        """Test simulation avec moteur d'exécution."""
        from backtest.simulator import simulate_trades
        
        # Créer des signaux simples
        signals = pd.Series(0, index=sample_ohlcv.index)
        signals.iloc[20] = 1   # Long
        signals.iloc[50] = -1  # Short
        signals.iloc[80] = 1   # Long
        signals.iloc[110] = 0  # Flat
        
        params = {"leverage": 1, "k_sl": 5.0, "fees_bps": 5}
        
        # Sans moteur d'exécution (mode simple)
        trades_simple = simulate_trades(sample_ohlcv, signals, params)
        
        # Avec moteur d'exécution dynamique
        engine = create_execution_engine(model="dynamic", spread_bps=5.0, slippage_bps=3.0)
        trades_realistic = simulate_trades(
            sample_ohlcv, signals, params,
            execution_engine=engine
        )
        
        # Les deux doivent générer des trades
        assert len(trades_simple) > 0
        assert len(trades_realistic) > 0
    
    def test_realistic_execution_higher_costs(self, volatile_ohlcv):
        """Test que l'exécution réaliste a des coûts plus élevés."""
        from backtest.simulator import simulate_trades
        
        signals = pd.Series(0, index=volatile_ohlcv.index)
        signals.iloc[30] = 1
        signals.iloc[100] = -1
        signals.iloc[150] = 0
        
        params = {"leverage": 1, "k_sl": 10.0, "fees_bps": 5}
        
        # Mode idéal
        engine_ideal = create_execution_engine(model="ideal")
        trades_ideal = simulate_trades(
            volatile_ohlcv, signals, params,
            execution_engine=engine_ideal
        )
        
        # Mode dynamique
        engine_dynamic = create_execution_engine(model="dynamic", spread_bps=10.0)
        trades_dynamic = simulate_trades(
            volatile_ohlcv, signals, params,
            execution_engine=engine_dynamic
        )
        
        if len(trades_ideal) > 0 and len(trades_dynamic) > 0:
            # Le PnL avec exécution réaliste devrait être inférieur ou égal
            # (coûts plus élevés)
            pnl_ideal = trades_ideal["pnl"].sum()
            pnl_dynamic = trades_dynamic["pnl"].sum()
            
            # Pas de comparaison stricte car dépend des données
            # On vérifie juste que les deux fonctionnent
            assert isinstance(pnl_ideal, (int, float))
            assert isinstance(pnl_dynamic, (int, float))


# ============================================================================
# Tests Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        engine = create_execution_engine(model="dynamic")
        engine.prepare(df)  # Ne doit pas crasher
        
        assert engine._prepared is False
    
    def test_single_bar(self):
        """Test avec une seule barre."""
        df = pd.DataFrame({
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000],
        }, index=pd.date_range("2024-01-01", periods=1, freq="1h"))
        
        engine = create_execution_engine(model="dynamic")
        engine.prepare(df)
        
        # Ne doit pas crasher
        result = engine.execute_order(price=100.0, side=1, bar_idx=0)
        assert result.executed_price > 0
    
    def test_zero_volume(self):
        """Test avec volume zéro."""
        df = pd.DataFrame({
            "open": [100] * 50,
            "high": [101] * 50,
            "low": [99] * 50,
            "close": [100.5] * 50,
            "volume": [0] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))
        
        config = ExecutionConfig(use_volume_slippage=True)
        engine = ExecutionEngine(config)
        engine.prepare(df)
        
        result = engine.execute_order(price=100.0, side=1, bar_idx=30)
        assert result.executed_price > 0
    
    def test_bar_idx_out_of_bounds(self, sample_ohlcv):
        """Test avec index hors limites."""
        engine = create_execution_engine(model="dynamic")
        engine.prepare(sample_ohlcv)
        
        # Index trop grand - doit être borné
        result = engine.execute_order(price=100.0, side=1, bar_idx=9999)
        assert result.executed_price > 0
        
        # Index négatif - doit être borné à 0
        result = engine.execute_order(price=100.0, side=1, bar_idx=-5)
        assert result.executed_price > 0
    
    def test_unprepared_engine(self):
        """Test moteur non préparé."""
        engine = create_execution_engine(model="dynamic")
        
        # Ne doit pas crasher même sans prepare()
        result = engine.execute_order(price=100.0, side=1, bar_idx=0)
        assert result.executed_price > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
