"""Test complet ProcessPoolExecutor avec BacktestEngine."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine  # noqa: E402
from data.loader import load_ohlcv  # noqa: E402


# Worker défini au niveau MODULE pour être picklable par ProcessPoolExecutor
def _worker_backtest(args):
    """Worker pour exécuter un backtest (défini au niveau module pour pickle)."""
    df_worker, strategy_name, params, tf, capital = args
    try:
        engine = BacktestEngine(initial_capital=capital)
        result = engine.run(df=df_worker, strategy=strategy_name, params=params, timeframe=tf)
        return {
            'params': params,
            'sharpe': result.metrics.get('sharpe_ratio', 0),
            'total_pnl': result.metrics.get('total_pnl', 0),
            'trades': result.metrics.get('total_trades', 0),
        }
    except Exception as e:
        return {'params': params, 'error': str(e)}


def test_parallel_backtests():
    """Teste l'exécution parallèle avec la vraie fonction worker."""
    print("=" * 70)
    print("TEST: ProcessPoolExecutor avec BacktestEngine")
    print("=" * 70)

    # Charger données
    df = load_ohlcv("BTCUSDC", "30m")
    print(f"\n✅ Données chargées: {len(df)} barres")

    # Paramètres de test
    strategy_key = "ema_cross"
    timeframe = "30m"
    initial_capital = 10000

    # Grille de paramètres
    param_grid = [
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 15, "slow_period": 40},
        {"fast_period": 20, "slow_period": 50},
    ]

    # Préparer arguments worker (utilise _worker_backtest défini au niveau module)
    backtest_args = [
        (df, strategy_key, combo, timeframe, initial_capital)
        for combo in param_grid
    ]

    print(f"\n📊 Grille: {len(param_grid)} combinaisons")
    print("🚀 Lancement ProcessPoolExecutor (4 workers)...\n")

    start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker_backtest, args) for args in backtest_args]

        results = []
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)

            # Afficher résultat
            if "error" not in result:
                print(f"  ✅ Run {i+1}/{len(param_grid)}: "
                      f"Sharpe={result['sharpe']:.2f}, "
                      f"PnL=${result['total_pnl']:.2f}, "
                      f"Trades={result['trades']}")
            else:
                print(f"  ❌ Run {i+1}/{len(param_grid)}: ERREUR - {result['error'][:50]}")

    duration = time.time() - start

    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"  Durée totale: {duration:.2f}s")
    print(f"  Durée par run: {duration/len(param_grid):.2f}s")
    print(f"  Runs réussis: {sum(1 for r in results if 'error' not in r)}/{len(results)}")

    # Meilleur résultat
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r["sharpe"])
        print(f"\n🏆 Meilleur Sharpe: {best['sharpe']:.2f}")
        print(f"   Paramètres: {best['params']}")
        print(f"   PnL: ${best['total_pnl']:.2f}")
        print(f"   Trades: {best['trades']}")

    print("\n✅ Test terminé avec succès!")
    return True


if __name__ == "__main__":
    try:
        test_parallel_backtests()
    except Exception as e:
        import traceback
        print(f"\n❌ ERREUR: {e}")
        print(traceback.format_exc())
        sys.exit(1)
