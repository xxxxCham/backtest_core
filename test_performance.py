"""
Script de test rapide pour vérifier les performances du backtest.

Usage:
    python test_performance.py

Vérifie que le système atteint au moins 15 bt/sec en séquentiel.
Projection parallèle (8 workers): 8 × 15 = 120 bt/sec minimum.
"""
import time
import sys
from data.loader import load_ohlcv
from backtest.engine import BacktestEngine


def test_performance(n_backtests=30, min_expected_btps=15):
    """
    Teste les performances du moteur de backtest.

    Args:
        n_backtests: Nombre de backtests à exécuter
        min_expected_btps: Débit minimum attendu (backtests/sec)

    Returns:
        True si performance acceptable, False sinon
    """
    print("=" * 70)
    print("TEST DE PERFORMANCE - Moteur Backtest")
    print("=" * 70)

    # Charger données
    print("\n[1/3] Chargement données BTCUSDC/30m...")
    df = load_ohlcv("BTCUSDC", "30m")
    print(f"✓ {len(df):,} barres chargées")

    # Préparer combinaisons
    print(f"\n[2/3] Préparation {n_backtests} combinaisons...")
    params_list = []
    for entry_z in [1.5, 2.0, 2.5]:
        for k_sl in [1.0, 1.5, 2.0]:
            for leverage in [1, 3]:
                params_list.append({
                    "entry_z": entry_z,
                    "k_sl": k_sl,
                    "leverage": leverage,
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "atr_period": 14,
                })
                if len(params_list) >= n_backtests:
                    break
            if len(params_list) >= n_backtests:
                break
        if len(params_list) >= n_backtests:
            break

    print(f"✓ {len(params_list)} combinaisons prêtes")

    # Exécuter backtests
    print(f"\n[3/3] Exécution {n_backtests} backtests...")
    engine = BacktestEngine(initial_capital=10000)

    t0 = time.perf_counter()
    success_count = 0
    errors = []

    for i, params in enumerate(params_list, 1):
        try:
            result = engine.run(
                df=df,
                strategy="bollinger_atr",
                params=params,
                symbol="BTCUSDC",
                timeframe="30m",
                silent_mode=True,
                fast_metrics=True
            )
            success_count += 1
        except Exception as e:
            errors.append(f"Backtest {i}: {str(e)[:50]}")

        # Progress indicator
        if i % 10 == 0 or i == n_backtests:
            elapsed = time.perf_counter() - t0
            bt_per_sec = i / elapsed if elapsed > 0 else 0
            print(f"  {i}/{n_backtests} • {bt_per_sec:.1f} bt/sec", end="\r")

    t1 = time.perf_counter()
    elapsed = t1 - t0

    print()  # Nouvelle ligne après progress

    # Résultats
    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)

    bt_per_sec = success_count / elapsed if elapsed > 0 else 0

    print(f"Backtests réussis: {success_count}/{n_backtests}")
    if errors:
        print(f"Erreurs: {len(errors)}")
        for err in errors[:3]:  # Afficher max 3 erreurs
            print(f"  • {err}")

    print(f"\nTemps total: {elapsed:.2f}s")
    print(f"Temps moyen: {elapsed/n_backtests:.3f}s/backtest")
    print(f"\n🎯 DÉBIT SÉQUENTIEL: {bt_per_sec:.1f} backtests/sec")

    # Projection parallèle
    parallel_8w = bt_per_sec * 8 * 0.8  # 8 workers, 80% efficiency
    print(f"📊 Projection 8 workers: {parallel_8w:.0f} backtests/sec")

    # Verdict
    print("\n" + "=" * 70)
    if bt_per_sec >= min_expected_btps:
        print(f"✅ PERFORMANCE OK (>= {min_expected_btps} bt/sec attendu)")
        print("=" * 70)
        return True
    else:
        print(f"❌ PERFORMANCE INSUFFISANTE (< {min_expected_btps} bt/sec)")
        print(f"   Attendu: {min_expected_btps:.1f} bt/sec")
        print(f"   Obtenu: {bt_per_sec:.1f} bt/sec")
        print(f"   Écart: {(min_expected_btps - bt_per_sec) / min_expected_btps * 100:.0f}% trop lent")
        print("=" * 70)
        return False


if __name__ == "__main__":
    # Test avec 30 backtests
    success = test_performance(n_backtests=30, min_expected_btps=15)

    # Exit code
    sys.exit(0 if success else 1)
