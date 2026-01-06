"""
Simulation du grid search UI pour tester le debug logging.
Reproduit exactement le comportement de ui/app.py.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from backtest.engine import BacktestEngine

# Configurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("🧪 SIMULATION GRID SEARCH UI - TEST DEBUG LOGGING")
print("=" * 80)
print()

# Charger données - utiliser fichier existant ou générer synthétique
DATA_FILE = 'data/sample_data/BTCUSDT_1h_6months.csv'
df = None

try:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    logger.info(f"Données chargées: {len(df)} barres")
except FileNotFoundError:
    logger.warning(f"Fichier {DATA_FILE} introuvable - génération données synthétiques")
    import numpy as np
    np.random.seed(42)
    n_bars = 4320  # 6 mois en 1h
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    close = 40000 + np.cumsum(np.random.randn(n_bars) * 100)
    df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 50,
        'high': close + np.abs(np.random.randn(n_bars) * 100),
        'low': close - np.abs(np.random.randn(n_bars) * 100),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    logger.info(f"Données synthétiques générées: {len(df)} barres")

# Créer une petite grille de test
param_grid = [
    {'atr_period': 10, 'atr_mult': 1.5, 'leverage': 1},
    {'atr_period': 12, 'atr_mult': 1.8, 'leverage': 1},
    {'atr_period': 14, 'atr_mult': 2.0, 'leverage': 1},
    {'atr_period': 16, 'atr_mult': 2.2, 'leverage': 1},
    {'atr_period': 18, 'atr_mult': 2.5, 'leverage': 1},
    {'atr_period': 20, 'atr_mult': 2.8, 'leverage': 1},
    {'atr_period': 24, 'atr_mult': 3.0, 'leverage': 1},
    {'atr_period': 28, 'atr_mult': 3.5, 'leverage': 1},
    {'atr_period': 30, 'atr_mult': 4.0, 'leverage': 1},
]

logger.info(f"Grille: {len(param_grid)} combinaisons")
print()

# Fonction run_single_backtest (comme dans l'UI)


def run_single_backtest(param_combo):
    """Exécute un seul backtest et retourne le résultat."""
    try:
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(
            df=df,
            strategy='atr_channel',
            params=param_combo,
            timeframe='1h'
        )

        params_str = str(param_combo)

        if result:
            return {
                "params": params_str,
                "params_dict": param_combo,
                "total_pnl": result.metrics["total_pnl"],
                "sharpe": result.metrics["sharpe_ratio"],
                "max_dd": result.metrics["max_drawdown"],
                "win_rate": result.metrics["win_rate"],
                "trades": result.metrics["total_trades"],
                "profit_factor": result.metrics["profit_factor"]
            }
        else:
            return {
                "params": params_str,
                "params_dict": param_combo,
                "error": "No result"
            }
    except Exception as e:
        return {
            "params": str(param_combo),
            "params_dict": param_combo,
            "error": str(e)
        }


# Exécution parallèle (comme dans l'UI)
results_list = []
n_workers = 4

logger.info(f"Lancement de {len(param_grid)} backtests en parallèle (workers={n_workers})")
print()

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    future_to_params = {
        executor.submit(run_single_backtest, combo): combo
        for combo in param_grid
    }

    completed = 0
    for future in as_completed(future_to_params):
        completed += 1
        result = future.result()

        # Retirer params_dict du résultat final
        result_clean = {k: v for k, v in result.items() if k != "params_dict"}
        results_list.append(result_clean)

        if completed % 3 == 0:
            logger.info(f"Progression: {completed}/{len(param_grid)} ({completed/len(param_grid)*100:.0f}%)")

logger.info(f"Optimisation terminée: {len(results_list)} tests")
print()

# Créer DataFrame (comme dans l'UI)
results_df = pd.DataFrame(results_list)

print("=" * 80)
print("🔍 DEBUG GRID SEARCH - Analyse de la colonne 'trades'")
print("=" * 80)
print()

# 🔍 DEBUG: Logging identique à l'UI
if "trades" in results_df.columns:
    logger.info("=" * 80)
    logger.info("🔍 DEBUG GRID SEARCH - Analyse de la colonne 'trades'")
    logger.info(f"   Type: {results_df['trades'].dtype}")
    logger.info(f"   Shape: {results_df['trades'].shape}")
    logger.info(f"   Premières valeurs: {results_df['trades'].head(10).tolist()}")
    logger.info(f"   Stats: min={results_df['trades'].min()}, max={results_df['trades'].max()}, mean={results_df['trades'].mean():.2f}")

    # Vérifier si il y a des floats
    trades_values = results_df['trades'].values
    fractional = [x for x in trades_values if isinstance(x, float) and not x.is_integer()]
    if fractional:
        logger.warning(f"   ⚠️  {len(fractional)} valeurs fractionnaires détectées: {fractional[:5]}")
    else:
        logger.info("   ✅ Toutes les valeurs sont des entiers")
    logger.info("=" * 80)

print()

# Filtrer erreurs
error_column = results_df.get("error")
if error_column is not None:
    valid_results = results_df[error_column.isna()]
else:
    valid_results = results_df

if not valid_results.empty:
    valid_results = valid_results.sort_values("sharpe", ascending=False)

    print("=" * 80)
    print("📊 STATISTIQUES DES RÉSULTATS")
    print("=" * 80)
    print()
    print(f"Nombre de résultats: {len(valid_results)}")
    print()
    print("Types des colonnes:")
    print(valid_results.dtypes)
    print()

    if "trades" in valid_results.columns:
        print("Statistiques 'trades':")
        print(f"  Type: {valid_results['trades'].dtype}")
        print(f"  Min: {valid_results['trades'].min()}")
        print(f"  Max: {valid_results['trades'].max()}")
        print(f"  Mean: {valid_results['trades'].mean():.2f}")
        print()

        # Afficher toutes les valeurs
        print("Toutes les valeurs de 'trades':")
        print(valid_results['trades'].tolist())

    print()
    print("=" * 80)
    print("🏆 TOP 10 COMBINAISONS")
    print("=" * 80)
    print()
    print(valid_results[['params', 'sharpe', 'total_pnl', 'trades', 'win_rate']].head(10).to_string())
    print()

    best = valid_results.iloc[0]
    print(f"🥇 Meilleure: {best['params']}")
    print(f"   Sharpe: {best['sharpe']:.2f}")
    print(f"   P&L: ${best['total_pnl']:,.2f}")
    print(f"   Trades: {best['trades']}")
    print()

print("=" * 80)
print("✅ SIMULATION TERMINÉE")
print("=" * 80)
