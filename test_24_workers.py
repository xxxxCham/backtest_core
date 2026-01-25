"""
Test de performance avec 24 workers.

Mesure le débit réel avec configuration optimisée 9950X.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from data.loader import load_ohlcv
from backtest.engine import BacktestEngine


def run_single_backtest(args):
    """Execute un backtest (appelé par worker)."""
    df, strategy, params, symbol, timeframe = args

    engine = BacktestEngine(initial_capital=10000)
    try:
        result = engine.run(
            df=df,
            strategy=strategy,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            silent_mode=True,
            fast_metrics=True
        )
        return True
    except Exception:
        return False


def main():
    print("=" * 80)
    print("TEST PERFORMANCE - 24 WORKERS (9950X Optimized)")
    print("=" * 80)

    # Charger données
    print("\n[1/3] Chargement BTCUSDC/30m...")
    df = load_ohlcv("BTCUSDC", "30m")
    print(f"✓ {len(df):,} barres chargées")

    # Générer 100 combinaisons
    print("\n[2/3] Génération 100 combinaisons...")
    param_combos = []
    for entry_z in [1.5, 1.75, 2.0, 2.25, 2.5]:
        for k_sl in [1.0, 1.25, 1.5, 1.75, 2.0]:
            for leverage in [1, 2, 3, 5]:
                param_combos.append({
                    "entry_z": entry_z,
                    "k_sl": k_sl,
                    "leverage": leverage,
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "atr_period": 14,
                })

    print(f"✓ {len(param_combos)} combinaisons prêtes")

    # Préparer arguments pour workers
    args_list = [
        (df, "bollinger_atr", params, "BTCUSDC", "30m")
        for params in param_combos
    ]

    # Test avec différents nombres de workers
    for n_workers in [8, 16, 24, 32]:
        print(f"\n{'='*80}")
        print(f"[3/3] TEST AVEC {n_workers} WORKERS")
        print(f"{'='*80}")

        t0 = time.perf_counter()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(run_single_backtest, args_list))

        t1 = time.perf_counter()
        elapsed = t1 - t0

        success_count = sum(results)
        bt_per_sec = success_count / elapsed if elapsed > 0 else 0

        print(f"\n📊 RÉSULTATS ({n_workers} workers):")
        print(f"   Réussis: {success_count}/{len(param_combos)}")
        print(f"   Temps total: {elapsed:.2f}s")
        print(f"   🎯 DÉBIT: {bt_per_sec:.1f} backtests/sec")
        print(f"   Efficacité: {(bt_per_sec / (19.2 * n_workers)) * 100:.1f}%")

        # Pause entre tests
        if n_workers < 32:
            time.sleep(2)

    print(f"\n{'='*80}")
    print("✅ Tests terminés")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
