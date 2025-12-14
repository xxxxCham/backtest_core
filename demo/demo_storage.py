"""
Démonstration du système de stockage des résultats de backtests.

Ce script montre comment :
1. Exécuter des backtests avec sauvegarde automatique
2. Charger des résultats précédents
3. Rechercher et filtrer les résultats
4. Gérer les sweeps
"""

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.storage import get_storage
from backtest.sweep import SweepEngine
from data.sample_data.generate_sample import generate_trending_data
from strategies.bollinger_atr import BollingerATRStrategy
from strategies.ema_cross import EMACrossStrategy


def demo_basic_save_and_load():
    """Démo 1: Sauvegarde et chargement basiques."""
    print("\n" + "=" * 70)
    print("DÉMO 1: SAUVEGARDE ET CHARGEMENT BASIQUES")
    print("=" * 70)

    # Générer des données
    print("\n📊 Génération des données...")
    df = generate_trending_data(n_bars=500, trend_strength=0.3)

    # Exécuter un backtest avec auto_save=True (par défaut)
    print("\n🚀 Exécution du backtest avec sauvegarde automatique...")
    engine = BacktestEngine(initial_capital=10000, auto_save=True)
    result = engine.run(
        df=df,
        strategy=BollingerATRStrategy(),
        params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 2},
        symbol="BTCUSDT",
        timeframe="1h",
    )

    run_id = result.meta["run_id"]
    print(f"\n✅ Backtest terminé et sauvegardé!")
    print(f"   Run ID: {run_id}")
    print(f"   Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Total P&L: ${result.metrics.get('total_pnl', 0):,.2f}")

    # Charger le résultat
    print("\n📂 Chargement du résultat depuis le stockage...")
    storage = get_storage()
    loaded_result = storage.load_result(run_id)

    print(f"\n✅ Résultat chargé avec succès!")
    print(f"   Stratégie: {loaded_result.meta['strategy']}")
    print(f"   Symbole: {loaded_result.meta['symbol']}")
    print(f"   Nombre de trades: {len(loaded_result.trades)}")


def demo_search_and_filter():
    """Démo 2: Recherche et filtrage des résultats."""
    print("\n" + "=" * 70)
    print("DÉMO 2: RECHERCHE ET FILTRAGE")
    print("=" * 70)

    # Exécuter plusieurs backtests
    print("\n🚀 Exécution de plusieurs backtests...")
    df = generate_trending_data(n_bars=500)

    strategies = [
        ("bollinger_atr", BollingerATRStrategy(), {"entry_z": 2.0}),
        ("ema_cross", EMACrossStrategy(), {"fast_period": 10, "slow_period": 30}),
    ]

    for name, strategy, params in strategies:
        engine = BacktestEngine(auto_save=True)
        result = engine.run(df=df, strategy=strategy, params=params, symbol="ETHUSDT")
        print(f"   ✓ {name}: Sharpe={result.metrics.get('sharpe_ratio', 0):.2f}")

    # Rechercher les résultats
    print("\n🔍 Recherche des résultats...")
    storage = get_storage()

    # Tous les résultats
    all_results = storage.list_results(limit=10)
    print(f"\n📊 {len(all_results)} résultats au total")

    # Filtrer par stratégie
    bollinger_results = storage.search_results(strategy="bollinger_atr")
    print(f"   - Bollinger ATR: {len(bollinger_results)} résultats")

    # Filtrer par Sharpe minimum
    good_results = storage.search_results(min_sharpe=0.5)
    print(f"   - Sharpe > 0.5: {len(good_results)} résultats")

    # Meilleurs résultats
    best = storage.get_best_results(n=3, metric="sharpe_ratio")
    print("\n🏆 Top 3 des meilleurs Sharpe Ratios:")
    for i, meta in enumerate(best, 1):
        sharpe = meta.metrics.get("sharpe_ratio", 0)
        strategy = meta.strategy
        print(f"   {i}. {strategy}: {sharpe:.2f}")


def demo_sweep_storage():
    """Démo 3: Stockage des sweeps."""
    print("\n" + "=" * 70)
    print("DÉMO 3: STOCKAGE DES SWEEPS")
    print("=" * 70)

    # Générer des données
    print("\n📊 Génération des données...")
    df = generate_trending_data(n_bars=300)

    # Exécuter un sweep avec sauvegarde automatique
    print("\n🔄 Exécution d'un sweep avec sauvegarde automatique...")
    engine = SweepEngine(max_workers=4, auto_save=True)

    param_grid = {
        "entry_z": [1.5, 2.0, 2.5],
        "k_sl": [1.0, 1.5, 2.0],
    }

    sweep_results = engine.run_sweep(
        df=df,
        strategy=BollingerATRStrategy(),
        param_grid=param_grid,
        show_progress=True,
    )

    print(f"\n✅ Sweep terminé et sauvegardé!")
    print(f"   Combinaisons testées: {sweep_results.n_completed}")
    print(f"   Meilleur Sharpe: {sweep_results.best_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Meilleurs paramètres: {sweep_results.best_params}")


def demo_result_management():
    """Démo 4: Gestion des résultats."""
    print("\n" + "=" * 70)
    print("DÉMO 4: GESTION DES RÉSULTATS")
    print("=" * 70)

    storage = get_storage()

    # Afficher les statistiques
    all_results = storage.list_results()
    print(f"\n📊 Statistiques du stockage:")
    print(f"   Total résultats: {len(all_results)}")

    # Grouper par stratégie
    strategies = {}
    for meta in all_results:
        strat = meta.strategy
        strategies[strat] = strategies.get(strat, 0) + 1

    print(f"\n📈 Résultats par stratégie:")
    for strat, count in strategies.items():
        print(f"   - {strat}: {count}")

    # Afficher quelques métadonnées
    if all_results:
        print(f"\n🔍 Détails du dernier résultat:")
        last = all_results[0]
        print(f"   Run ID: {last.run_id}")
        print(f"   Date: {last.timestamp}")
        print(f"   Stratégie: {last.strategy}")
        print(f"   Symbole: {last.symbol}")
        print(f"   Sharpe: {last.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Trades: {last.n_trades}")

    # Option de nettoyage (commenté pour la sécurité)
    # print("\n🧹 Pour nettoyer les anciens résultats:")
    # print("   storage._cleanup_old_results(keep_last=100)")
    # print("   storage.clear_all()  # ATTENTION: Supprime TOUT!")


def demo_load_and_analyze():
    """Démo 5: Charger et analyser un résultat."""
    print("\n" + "=" * 70)
    print("DÉMO 5: CHARGEMENT ET ANALYSE")
    print("=" * 70)

    storage = get_storage()
    all_results = storage.list_results(limit=1)

    if not all_results:
        print("\n⚠️ Aucun résultat disponible. Exécutez d'abord la démo 1.")
        return

    # Charger le dernier résultat
    meta = all_results[0]
    print(f"\n📂 Chargement du résultat: {meta.run_id}")

    result = storage.load_result(meta.run_id)

    # Analyser
    print(f"\n📊 Analyse du résultat:")
    print(f"   Période: {meta.period_start} → {meta.period_end}")
    print(f"   Barres: {meta.n_bars}")
    print(f"   Trades: {meta.n_trades}")
    print(f"   Durée exec: {meta.duration_sec:.2f}s")

    print(f"\n💰 Métriques de performance:")
    metrics = result.metrics
    print(f"   Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"   Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}%")

    # Afficher quelques trades
    if len(result.trades) > 0:
        print(f"\n📝 Premiers trades:")
        print(result.trades.head(3).to_string())


def main():
    """Fonction principale."""
    print("\n" + "=" * 70)
    print("🚀 DÉMONSTRATION DU SYSTÈME DE STOCKAGE")
    print("=" * 70)

    try:
        # Exécuter toutes les démos
        demo_basic_save_and_load()
        demo_search_and_filter()
        demo_sweep_storage()
        demo_result_management()
        demo_load_and_analyze()

        print("\n" + "=" * 70)
        print("✅ TOUTES LES DÉMOS TERMINÉES!")
        print("=" * 70)

        # Afficher l'emplacement des résultats
        storage = get_storage()
        print(f"\n📁 Les résultats sont stockés dans:")
        print(f"   {storage.storage_dir.absolute()}")

        print("\n💡 Conseils:")
        print("   - Les résultats sont sauvegardés automatiquement par défaut")
        print("   - Utilisez auto_save=False pour désactiver la sauvegarde")
        print("   - Utilisez storage.search_results() pour filtrer les résultats")
        print("   - Utilisez storage.get_best_results() pour les meilleurs runs")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
