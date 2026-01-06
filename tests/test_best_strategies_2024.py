#!/usr/bin/env python
"""Test des meilleures stratégies sur année complète 2024."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
from data.loader import load_ohlcv


def test_ema_cross_21_38():
    """Test EMA Cross (21/38) - Meilleur résultat du sweep."""
    print("\n" + "="*80)
    print("🏆 TEST #1: EMA CROSS (21/38)")
    print("="*80)

    # Charger données
    df = load_ohlcv(
        symbol="BTCUSDC",
        timeframe="30m",
        start="2024-01-01",
        end="2024-12-31"
    )
    print(f"📊 Données: {len(df)} barres")

    # Paramètres optimaux
    params = {
        "fast_period": 21,
        "slow_period": 38,
        "leverage": 1,
        "fees_bps": 10,
        "slippage_bps": 5
    }

    print(f"⚙️  Paramètres: {params}")

    # Backtest
    engine = BacktestEngine(initial_capital=10000.0)
    result = engine.run(
        df=df,
        strategy="ema_cross",
        params=params,
        symbol="BTCUSDC",
        timeframe="30m"
    )

    # Afficher résultats
    print("\n📈 RÉSULTATS:")
    print(f"   PnL: ${result.metrics['total_pnl']:.2f}")
    print(f"   Return: {result.metrics['total_return_pct']:.2f}%")
    print(f"   Sharpe: {result.metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result.metrics['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {result.metrics['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"   Trades: {result.metrics['total_trades']}")
    print(f"   CAGR: {result.metrics.get('annualized_return', 0):.2f}%")

    return result


def test_bollinger_atr_optimized():
    """Test BollingerATR avec paramètres optimisés (run 0150267a)."""
    print("\n" + "="*80)
    print("⭐ TEST #2: BOLLINGER ATR (Optimisé)")
    print("="*80)

    # Charger données
    df = load_ohlcv(
        symbol="BTCUSDC",
        timeframe="30m",
        start="2024-01-01",
        end="2024-12-31"
    )
    print(f"📊 Données: {len(df)} barres")

    # Paramètres optimaux du run 0150267a
    params = {
        "bb_period": 38,
        "bb_std": 2.55,
        "atr_period": 17,
        "atr_percentile": 50,
        "entry_z": 2.6,
        "k_sl": 2.6,
        "leverage": 1,
        "fees_bps": 10,
        "slippage_bps": 5
    }

    print(f"⚙️  Paramètres: {params}")

    # Backtest
    engine = BacktestEngine(initial_capital=10000.0)
    result = engine.run(
        df=df,
        strategy="bollinger_atr",
        params=params,
        symbol="BTCUSDC",
        timeframe="30m"
    )

    # Afficher résultats
    print("\n📈 RÉSULTATS:")
    print(f"   PnL: ${result.metrics['total_pnl']:.2f}")
    print(f"   Return: {result.metrics['total_return_pct']:.2f}%")
    print(f"   Sharpe: {result.metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {result.metrics['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {result.metrics['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"   Trades: {result.metrics['total_trades']}")
    print(f"   CAGR: {result.metrics.get('annualized_return', 0):.2f}%")

    return result


def test_top_5_ema():
    """Test top 5 configs EMA Cross."""
    print("\n" + "="*80)
    print("📊 TEST #3: TOP 5 EMA CROSS")
    print("="*80)

    configs = [
        (21, 38, "Config #1 - Meilleur"),
        (23, 35, "Config #2"),
        (19, 41, "Config #3"),
        (23, 38, "Config #4"),
        (17, 44, "Config #5"),
    ]

    df = load_ohlcv(
        symbol="BTCUSDC",
        timeframe="30m",
        start="2024-01-01",
        end="2024-12-31"
    )

    results = []

    for fast, slow, label in configs:
        print(f"\n🔹 {label}: EMA({fast}/{slow})")

        params = {
            "fast_period": fast,
            "slow_period": slow,
            "leverage": 1,
            "fees_bps": 10,
            "slippage_bps": 5
        }

        engine = BacktestEngine(initial_capital=10000.0)
        result = engine.run(
            df=df,
            strategy="ema_cross",
            params=params,
            symbol="BTCUSDC",
            timeframe="30m"
        )
        m = result.metrics

        print(f"   PnL: ${m['total_pnl']:>10.2f} | Return: {m['total_return_pct']:>6.2f}% | "
              f"Sharpe: {m['sharpe_ratio']:>4.2f} | Trades: {m['total_trades']:>3}")

        results.append((fast, slow, label, result))

    # Classement
    print("\n" + "="*80)
    print("🏆 CLASSEMENT FINAL (par Sharpe ratio):")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x[3].metrics['sharpe_ratio'], reverse=True)

    for i, (fast, slow, label, result) in enumerate(sorted_results, 1):
        m = result.metrics
        print(f"{i}. EMA({fast:2}/{slow:2}) - PnL: ${m['total_pnl']:>8.2f} | "
              f"Sharpe: {m['sharpe_ratio']:>5.2f} | Return: {m['total_return_pct']:>6.2f}%")

    return sorted_results


def main():
    print("\n" + "="*80)
    print("🎯 TEST DES MEILLEURES STRATÉGIES - ANNÉE COMPLÈTE 2024")
    print("="*80)
    print("Symbole: BTCUSDC")
    print("Timeframe: 30m")
    print("Période: 2024-01-01 → 2024-12-31")
    print("Capital initial: $10,000")
    print("="*80)

    # Test 1: EMA Cross 21/38
    result_ema = test_ema_cross_21_38()

    # Test 2: BollingerATR optimisé
    result_bollinger = test_bollinger_atr_optimized()

    # Test 3: Top 5 EMA
    test_top_5_ema()

    # Comparaison finale
    print("\n" + "="*80)
    print("📊 COMPARAISON GLOBALE")
    print("="*80)

    ema_m = result_ema.metrics
    boll_m = result_bollinger.metrics

    print(f"\n{'Métrique':<20} {'EMA(21/38)':<15} {'BollingerATR':<15} {'Gagnant'}")
    print("-" * 65)
    print(f"{'PnL':<20} ${ema_m['total_pnl']:<14.2f} ${boll_m['total_pnl']:<14.2f} "
          f"{'EMA' if ema_m['total_pnl'] > boll_m['total_pnl'] else 'Bollinger'}")
    print(f"{'Return %':<20} {ema_m['total_return_pct']:<14.2f} {boll_m['total_return_pct']:<14.2f} "
          f"{'EMA' if ema_m['total_return_pct'] > boll_m['total_return_pct'] else 'Bollinger'}")
    print(f"{'Sharpe':<20} {ema_m['sharpe_ratio']:<14.2f} {boll_m['sharpe_ratio']:<14.2f} "
          f"{'EMA' if ema_m['sharpe_ratio'] > boll_m['sharpe_ratio'] else 'Bollinger'}")
    print(f"{'Win Rate %':<20} {ema_m['win_rate_pct']:<14.2f} {boll_m['win_rate_pct']:<14.2f} "
          f"{'EMA' if ema_m['win_rate_pct'] > boll_m['win_rate_pct'] else 'Bollinger'}")
    print(f"{'Profit Factor':<20} {ema_m['profit_factor']:<14.2f} {boll_m['profit_factor']:<14.2f} "
          f"{'EMA' if ema_m['profit_factor'] > boll_m['profit_factor'] else 'Bollinger'}")
    print(f"{'Max DD %':<20} {ema_m['max_drawdown_pct']:<14.2f} {boll_m['max_drawdown_pct']:<14.2f} "
          f"{'EMA' if abs(ema_m['max_drawdown_pct']) < abs(boll_m['max_drawdown_pct']) else 'Bollinger'}")
    print(f"{'Trades':<20} {ema_m['total_trades']:<14} {boll_m['total_trades']:<14} "
          f"{'EMA' if ema_m['total_trades'] > boll_m['total_trades'] else 'Bollinger'}")

    print("\n" + "="*80)
    print("✅ TESTS TERMINÉS")
    print("="*80)


if __name__ == "__main__":
    main()
