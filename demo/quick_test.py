"""
Backtest Core - Quick Test Demo
===============================

Script de démonstration pour tester le moteur de backtest.
Génère des données synthétiques et exécute un backtest complet.

Usage:
    python demo/quick_test.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
import pandas as pd

# Imports du moteur
from backtest.engine import BacktestEngine, quick_backtest
from indicators.registry import IndicatorRegistry
from strategies.bollinger_atr import BollingerATRStrategy
from strategies.ema_cross import EMACrossStrategy


def generate_synthetic_ohlcv(
    n_bars: int = 10000,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    start_date: str = "2024-01-01"
) -> pd.DataFrame:
    """
    Génère des données OHLCV synthétiques pour les tests.

    Args:
        n_bars: Nombre de barres à générer
        start_price: Prix de départ
        volatility: Volatilité (écart-type des rendements)
        trend: Drift par période (tendance)
        start_date: Date de début

    Returns:
        DataFrame OHLCV avec colonnes (open, high, low, close, volume)
    """
    np.random.seed(42)  # Reproductibilité

    # Générer les rendements log-normaux
    returns = np.random.normal(trend, volatility, n_bars)

    # Construire les prix
    prices = start_price * np.exp(np.cumsum(returns))

    # Construire OHLCV
    df = pd.DataFrame()

    # Close prices
    df["close"] = prices

    # Open = previous close + noise
    df["open"] = df["close"].shift(1).fillna(start_price)
    df["open"] += np.random.normal(0, volatility * start_price * 0.1, n_bars)

    # High/Low avec spread réaliste
    spread = np.random.uniform(0.001, 0.01, n_bars) * prices
    df["high"] = np.maximum(df["open"], df["close"]) + spread
    df["low"] = np.minimum(df["open"], df["close"]) - spread

    # Volume
    df["volume"] = np.random.exponential(1000, n_bars) * (1 + np.abs(returns) * 10)

    # Index datetime
    start = pd.Timestamp(start_date, tz="UTC")
    df.index = pd.date_range(start=start, periods=n_bars, freq="1min")
    df.index.name = "timestamp"

    return df


def test_indicators():
    """Test les indicateurs techniques."""
    print("\n" + "=" * 60)
    print("TEST DES INDICATEURS")
    print("=" * 60)

    # Générer données
    df = generate_synthetic_ohlcv(n_bars=500)
    print(f"✅ Données générées: {len(df)} barres")

    # Test Bollinger
    from indicators.bollinger import bollinger_bands
    upper, middle, lower = bollinger_bands(df["close"], period=20, std_dev=2.0)
    print(f"✅ Bollinger Bands: upper={upper[-1]:.2f}, middle={middle[-1]:.2f}, lower={lower[-1]:.2f}")

    # Test ATR
    from indicators.atr import atr
    atr_vals = atr(df["high"], df["low"], df["close"], period=14)
    print(f"✅ ATR(14): {atr_vals[-1]:.4f}")

    # Test RSI
    from indicators.rsi import rsi
    rsi_vals = rsi(df["close"], period=14)
    print(f"✅ RSI(14): {rsi_vals[-1]:.2f}")

    # Test EMA
    from indicators.ema import ema, sma
    ema_vals = ema(df["close"], period=20)
    sma_vals = sma(df["close"], period=20)
    print(f"✅ EMA(20): {ema_vals[-1]:.2f}, SMA(20): {sma_vals[-1]:.2f}")

    # Test Registry
    registry = IndicatorRegistry()
    indicators = registry.calculate_multiple(df, {
        "bollinger": {"period": 20, "std_dev": 2.0},
        "atr": {"period": 14},
        "rsi": {"period": 14}
    })
    print(f"✅ Registry: calculé {len(indicators)} indicateurs")

    return True


def test_strategies():
    """Test les stratégies de trading."""
    print("\n" + "=" * 60)
    print("TEST DES STRATÉGIES")
    print("=" * 60)

    # Générer données
    df = generate_synthetic_ohlcv(n_bars=1000)
    print(f"✅ Données générées: {len(df)} barres")

    # Test Bollinger ATR Strategy
    strategy = BollingerATRStrategy()
    print(f"✅ Stratégie: {strategy.name}")
    print(f"   Indicateurs requis: {strategy.required_indicators}")

    # Calculer indicateurs
    from indicators.registry import calculate_indicator
    indicators = {
        "bollinger": calculate_indicator("bollinger", df, {"period": 20, "std_dev": 2.0}),
        "atr": calculate_indicator("atr", df, {"period": 14})
    }

    # Générer signaux
    params = {"entry_z": 2.0, "k_sl": 1.5, "leverage": 3}
    signals = strategy.generate_signals(df, indicators, params)

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    print(f"✅ Signaux générés: {n_long} longs, {n_short} shorts")

    # Test EMA Cross Strategy
    strategy2 = EMACrossStrategy()
    signals2 = strategy2.generate_signals(df, {}, {"fast_period": 12, "slow_period": 26})
    n_cross = (signals2 != 0).sum()
    print(f"✅ EMA Cross signaux: {n_cross} croisements")

    return True


def test_backtest_engine():
    """Test le moteur de backtest complet."""
    print("\n" + "=" * 60)
    print("TEST DU MOTEUR DE BACKTEST")
    print("=" * 60)

    # Générer données
    df = generate_synthetic_ohlcv(n_bars=5000, volatility=0.01)
    print(f"✅ Données générées: {len(df)} barres")

    # Créer le moteur
    engine = BacktestEngine(initial_capital=10000)
    print("✅ Moteur initialisé")

    # Test avec Bollinger ATR
    print("\n--- Backtest: Bollinger + ATR ---")
    result = engine.run(
        df=df,
        strategy="bollinger_atr",
        params={
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "k_sl": 1.5,
            "leverage": 3
        },
        symbol="SYNTHETIC",
        timeframe="1m"
    )

    print(f"✅ Backtest terminé en {result.meta['duration_sec']:.2f}s")
    print(f"   Trades: {result.metrics['total_trades']}")
    print(f"   P&L: ${result.metrics['total_pnl']:,.2f}")
    print(f"   Sharpe: {result.metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result.metrics['max_drawdown']:.1f}%")
    print(f"   Win Rate: {result.metrics['win_rate']:.1f}%")

    # Test avec EMA Cross
    print("\n--- Backtest: EMA Crossover ---")
    result2 = engine.run(
        df=df,
        strategy="ema_cross",
        params={
            "fast_period": 12,
            "slow_period": 26,
            "leverage": 2
        },
        symbol="SYNTHETIC",
        timeframe="1m"
    )

    print("✅ Backtest terminé")
    print(f"   Trades: {result2.metrics['total_trades']}")
    print(f"   P&L: ${result2.metrics['total_pnl']:,.2f}")
    print(f"   Sharpe: {result2.metrics['sharpe_ratio']:.2f}")

    return True


def test_quick_backtest():
    """Test la fonction de backtest rapide."""
    print("\n" + "=" * 60)
    print("TEST QUICK BACKTEST")
    print("=" * 60)

    df = generate_synthetic_ohlcv(n_bars=2000)

    result = quick_backtest(
        df,
        strategy_name="bollinger_atr",
        leverage=3,
        k_sl=2.0
    )

    print(f"✅ Quick backtest: {result.metrics['total_trades']} trades, P&L=${result.metrics['total_pnl']:,.2f}")

    return True


def main():
    """Exécute tous les tests."""
    print("=" * 60)
    print("  BACKTEST CORE - SUITE DE TESTS")
    print("=" * 60)

    tests = [
        ("Indicateurs", test_indicators),
        ("Stratégies", test_strategies),
        ("Moteur Backtest", test_backtest_engine),
        ("Quick Backtest", test_quick_backtest),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ ERREUR dans {name}: {e}")
            results.append((name, False))

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)

    for name, success in results:
        status = "✅ OK" if success else "❌ ÉCHOUÉ"
        print(f"  {name}: {status}")

    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)

    print(f"\n📊 {n_passed}/{n_total} tests réussis")

    if n_passed == n_total:
        print("\n🎉 Tous les tests sont passés! Le moteur est fonctionnel.")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
