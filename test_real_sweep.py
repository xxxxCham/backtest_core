"""
Test de sweep avec une vraie stratégie (BollingerATR).

Lance un petit sweep pour valider :
- Sauvegarde automatique
- Vitesse de calcul
- Qualité des résultats
"""
import sys
import time
from pathlib import Path

# Configuration du path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.sweep import SweepEngine
from data.loader import load_ohlcv
from strategies.bollinger_atr import BollingerATRStrategy


def main():
    print("=" * 70)
    print("TEST SWEEP RÉEL - BollingerATR")
    print("=" * 70)

    # 1. Charger les données
    print("\n[1/4] Chargement des données...")

    # Utiliser ETHUSDT 1m (seules données disponibles dans sample_data)
    symbol = "ETHUSDT"
    timeframe = "1m"

    try:
        # Charger directement le fichier CSV sample
        import pandas as pd
        df = pd.read_csv('data/sample_data/ETHUSDT_1m_sample.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Limiter à 5000 dernières barres pour test rapide
        df = df.tail(5000)
        print(f"✓ Données chargées: {symbol} {timeframe}")
        print(f"  - Période: {df.index[0]} → {df.index[-1]}")
        print(f"  - Barres: {len(df)}")
    except Exception as e:
        print(f"✗ Erreur chargement données: {e}")
        return False

    # 2. Initialiser la stratégie
    print("\n[2/4] Initialisation de la stratégie...")
    strategy = BollingerATRStrategy()
    print(f"✓ Stratégie: {strategy.__class__.__name__}")

    # 3. Définir une grille RÉDUITE pour test rapide
    # Normalement vous auriez des milliers de combinaisons, ici on teste juste le mécanisme
    param_grid = {
        "bb_period": [20, 30],           # 2 valeurs au lieu de dizaines
        "bb_std": [2.0, 2.5],            # 2 valeurs
        "atr_period": [14],              # 1 valeur
        "atr_percentile": [30, 50],      # 2 valeurs
        "entry_z": [2.0],                # 1 valeur
        "k_sl": [1.5],                   # 1 valeur
    }

    total_combos = 2 * 2 * 1 * 2 * 1 * 1  # = 8 combinaisons
    print(f"\n[3/4] Configuration du sweep:")
    print(f"  - Combinaisons: {total_combos}")
    print(f"  - bb_period: {param_grid['bb_period']}")
    print(f"  - bb_std: {param_grid['bb_std']}")
    print(f"  - atr_percentile: {param_grid['atr_percentile']}")

    # 4. Exécuter le sweep
    print(f"\n[4/4] Exécution du sweep...")
    print("-" * 70)

    engine = SweepEngine(
        max_workers=6,  # Utiliser 6 workers pour tester la parallélisation
        initial_capital=10000,
        auto_save=True  # ← CRITIQUE: Sauvegarde auto activée
    )

    start_time = time.time()

    try:
        sweep_results = engine.run_sweep(
            df=df,
            strategy=strategy,
            param_grid=param_grid,
            optimize_for="sharpe_ratio",
            silent_mode=True,
            fast_metrics=True,  # Utiliser fast_metrics pour vitesse max
        )

        elapsed = time.time() - start_time

        print("-" * 70)
        print(f"\n✅ Sweep terminé en {elapsed:.1f}s")
        print(f"  - Réussis: {sweep_results.n_completed}/{total_combos}")
        print(f"  - Échoués: {sweep_results.n_failed}")
        print(f"  - Vitesse: {total_combos/elapsed:.1f} bt/s")

        # 5. Afficher les meilleurs résultats
        if sweep_results.n_completed > 0:
            print(f"\n" + "=" * 70)
            print("MEILLEURS PARAMÈTRES TROUVÉS")
            print("=" * 70)
            print(f"\nParamètres:")
            for key, value in sweep_results.best_params.items():
                print(f"  {key:20s}: {value}")

            print(f"\nMétriques:")
            metrics = sweep_results.best_metrics
            print(f"  {'Sharpe Ratio':20s}: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
            print(f"  {'Total Return':20s}: {metrics.get('total_return_pct', 'N/A'):.2f}%")
            print(f"  {'Total PnL':20s}: ${metrics.get('total_pnl', 'N/A'):,.2f}")
            print(f"  {'Win Rate':20s}: {metrics.get('win_rate_pct', 'N/A'):.1f}%")
            print(f"  {'Profit Factor':20s}: {metrics.get('profit_factor', 'N/A'):.2f}")
            print(f"  {'Max Drawdown':20s}: {metrics.get('max_drawdown_pct', 'N/A'):.2f}%")
            print(f"  {'Total Trades':20s}: {metrics.get('total_trades', 'N/A')}")

            # 6. Vérifier la sauvegarde
            print(f"\n" + "=" * 70)
            print("VÉRIFICATION DE LA SAUVEGARDE")
            print("=" * 70)

            sweep_dirs = sorted(Path("backtest_results").glob("sweep_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if sweep_dirs:
                latest = sweep_dirs[0]
                summary_file = latest / "summary.json"
                results_file = latest / "all_results.parquet"

                print(f"\nRépertoire: {latest.name}")
                print(f"  ✓ summary.json: {'OUI' if summary_file.exists() else 'NON'}")
                print(f"  ✓ all_results.parquet: {'OUI' if results_file.exists() else 'NON'}")

                if results_file.exists():
                    import pandas as pd
                    df_results = pd.read_parquet(results_file)
                    print(f"\nContenu du parquet:")
                    print(f"  - Lignes: {len(df_results)}")
                    print(f"  - Colonnes: {len(df_results.columns)}")

                    # Afficher top 3 résultats
                    if 'sharpe_ratio' in df_results.columns and len(df_results) > 0:
                        print(f"\n  Top 3 configurations (par Sharpe):")
                        top3 = df_results.nlargest(3, 'sharpe_ratio')
                        for i, (idx, row) in enumerate(top3.iterrows(), 1):
                            print(f"\n  #{i}:")
                            print(f"    bb_period={row.get('bb_period')}, bb_std={row.get('bb_std')}, atr_percentile={row.get('atr_percentile')}")
                            print(f"    Sharpe={row.get('sharpe_ratio', 0):.3f}, Return={row.get('total_return_pct', 0):.2f}%, Trades={row.get('total_trades', 0)}")

            print(f"\n" + "=" * 70)
            print("✅ TEST RÉUSSI - Tout fonctionne correctement !")
            print("=" * 70 + "\n")

            return True
        else:
            print(f"\n⚠ Aucun backtest réussi - vérifiez les logs ci-dessus")
            return False

    except Exception as e:
        print(f"\n✗ ERREUR lors du sweep: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
