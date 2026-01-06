"""
Profiler de Performance - Backtest Core

Outils pour profiler et analyser les performances du code.
Identifie les goulots d'étranglement (bottlenecks) pour optimisation.

Usage:
    # Profiler un backtest simple
    python tools/profiler.py simple --strategy ema_cross

    # Profiler une optimisation Grid Search
    python tools/profiler.py grid --strategy ema_cross --combinations 50

    # Analyser un rapport de profiling
    python tools/profiler.py analyze --report profiling_results/report_20250101_120000.prof
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from backtest.engine import BacktestEngine  # noqa: E402
from data.loader import load_ohlcv  # noqa: E402
from strategies import list_strategies  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.parameters import generate_param_grid, ParameterSpec  # noqa: E402


class PerformanceProfiler:
    """Profiler de performance pour backtests."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = cProfile.Profile()
        self.start_time = None
        self.end_time = None

    def start(self):
        """Démarre le profiling."""
        self.start_time = time.perf_counter()
        self.profiler.enable()
        print(f"🔍 Profiling démarré à {datetime.now().strftime('%H:%M:%S')}")

    def stop(self):
        """Arrête le profiling."""
        self.profiler.disable()
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        print(f"✅ Profiling terminé en {duration:.2f}s")

    def save_report(self, name: str = "profile") -> Path:
        """Sauvegarde le rapport de profiling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{name}_{timestamp}.prof"

        self.profiler.dump_stats(str(report_path))
        print(f"💾 Rapport sauvegardé: {report_path}")
        return report_path

    def print_stats(self, top_n: int = 30, sort_by: str = "cumulative"):
        """Affiche les statistiques de profiling."""
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(sort_by)

        print(f"\n{'='*80}")
        print(f"TOP {top_n} FONCTIONS LES PLUS LENTES (tri: {sort_by})")
        print(f"{'='*80}\n")

        stats.print_stats(top_n)
        print(stream.getvalue())

    def print_callers(self, top_n: int = 10):
        """Affiche qui appelle les fonctions les plus lentes."""
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")

        print(f"\n{'='*80}")
        print(f"TOP {top_n} APPELANTS (Qui appelle les fonctions lentes)")
        print(f"{'='*80}\n")

        stats.print_callers(top_n)
        print(stream.getvalue())


def profile_single_backtest(
    strategy_name: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Profile un backtest simple.

    Args:
        strategy_name: Nom de la stratégie
        symbol: Symbole à tester
        timeframe: Timeframe
        start: Date de début
        end: Date de fin
        params: Paramètres de la stratégie (optionnel)

    Returns:
        Résultats du backtest
    """
    print(f"\n{'='*80}")
    print("PROFILING BACKTEST SIMPLE")
    print(f"{'='*80}")
    print(f"Stratégie: {strategy_name}")
    print(f"Symbole: {symbol} ({timeframe})")
    print(f"Période: {start} → {end}")
    print(f"{'='*80}\n")

    # Charger les données
    print("📊 Chargement des données...")
    df = load_ohlcv(symbol, timeframe, start=start, end=end)
    print(f"   ✅ {len(df)} barres chargées\n")

    # Créer le moteur
    config = Config(initial_capital=10000.0)
    engine = BacktestEngine(config)

    # Obtenir les paramètres par défaut si non fournis
    if params is None:
        from strategies import create_strategy
        strategy_instance = create_strategy(strategy_name)
        params = {
            spec.name: spec.default
            for spec in strategy_instance.parameter_specs.values()
        }

    # Profiler l'exécution
    profiler = PerformanceProfiler()
    profiler.start()

    result = engine.run(strategy_name, params, df)

    profiler.stop()

    # Afficher les résultats
    if result:
        print(f"\n{'='*80}")
        print("RÉSULTATS BACKTEST")
        print(f"{'='*80}")
        print(f"PnL Total: ${result.metrics['total_pnl']:.2f}")
        print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {result.metrics['win_rate']:.2f}%")
        print(f"Total Trades: {result.metrics['total_trades']}")
        print(f"{'='*80}\n")

    # Afficher les stats de profiling
    profiler.print_stats(top_n=30, sort_by="cumulative")
    profiler.print_callers(top_n=10)

    # Sauvegarder le rapport
    profiler.save_report(f"backtest_{strategy_name}_{symbol}")

    return {
        "result": result,
        "profiler": profiler,
    }


def profile_grid_search(
    strategy_name: str,
    n_combinations: int = 50,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start: str = "2024-01-01",
    end: str = "2024-12-31",
) -> Dict[str, Any]:
    """
    Profile une optimisation Grid Search.

    Args:
        strategy_name: Nom de la stratégie
        n_combinations: Nombre de combinaisons à tester
        symbol: Symbole à tester
        timeframe: Timeframe
        start: Date de début
        end: Date de fin

    Returns:
        Résultats de l'optimisation
    """
    print(f"\n{'='*80}")
    print("PROFILING GRID SEARCH")
    print(f"{'='*80}")
    print(f"Stratégie: {strategy_name}")
    print(f"Symbole: {symbol} ({timeframe})")
    print(f"Période: {start} → {end}")
    print(f"Combinaisons: {n_combinations}")
    print(f"{'='*80}\n")

    # Charger les données
    print("📊 Chargement des données...")
    df = load_ohlcv(symbol, timeframe, start=start, end=end)
    print(f"   ✅ {len(df)} barres chargées\n")

    # Créer le moteur
    config = Config(initial_capital=10000.0)
    engine = BacktestEngine(config)

    # Générer la grille de paramètres
    from strategies import create_strategy
    strategy_instance = create_strategy(strategy_name)

    # Créer des specs avec ranges pour optimisation
    param_specs: Dict[str, ParameterSpec] = {}
    for name, spec in strategy_instance.parameter_specs.items():
        # Utiliser un range autour de la valeur par défaut
        if spec.param_type in ("int", "float"):
            min_val = max(spec.min_value, spec.default * 0.5)
            max_val = min(spec.max_value, spec.default * 1.5)
            param_specs[name] = ParameterSpec(
                name=name,
                param_type=spec.param_type,
                default=spec.default,
                min_value=min_val,
                max_value=max_val,
                step=spec.step,
            )
        else:
            param_specs[name] = spec

    print("🔢 Génération de la grille de paramètres...")
    param_grid = generate_param_grid(param_specs, max_combinations=n_combinations)
    print(f"   ✅ {len(param_grid)} combinaisons générées\n")

    # Profiler l'optimisation
    profiler = PerformanceProfiler()
    profiler.start()

    results = []
    for i, params in enumerate(param_grid):
        result = engine.run(strategy_name, params, df)
        if result:
            results.append({
                "params": params,
                "sharpe": result.metrics["sharpe_ratio"],
                "pnl": result.metrics["total_pnl"],
                "trades": result.metrics["total_trades"],
            })

        # Afficher la progression
        if (i + 1) % 10 == 0:
            print(f"   Progression: {i+1}/{len(param_grid)} ({(i+1)/len(param_grid)*100:.1f}%)")

    profiler.stop()

    # Afficher les meilleurs résultats
    if results:
        results_df = pd.DataFrame(results)
        best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]

        print(f"\n{'='*80}")
        print("MEILLEURS RÉSULTATS")
        print(f"{'='*80}")
        print(f"Meilleur Sharpe: {best_sharpe['sharpe']:.3f}")
        print(f"PnL: ${best_sharpe['pnl']:.2f}")
        print(f"Trades: {best_sharpe['trades']:.0f}")
        print(f"Paramètres: {best_sharpe['params']}")
        print(f"{'='*80}\n")

    # Afficher les stats de profiling
    profiler.print_stats(top_n=30, sort_by="cumulative")
    profiler.print_callers(top_n=10)

    # Sauvegarder le rapport
    profiler.save_report(f"grid_{strategy_name}_{n_combinations}combos")

    return {
        "results": results,
        "profiler": profiler,
    }


def analyze_profile_report(report_path: str):
    """
    Analyse un rapport de profiling existant.

    Args:
        report_path: Chemin vers le fichier .prof
    """
    print(f"\n{'='*80}")
    print("ANALYSE RAPPORT DE PROFILING")
    print(f"{'='*80}")
    print(f"Fichier: {report_path}")
    print(f"{'='*80}\n")

    stats = pstats.Stats(report_path)
    stats.strip_dirs()

    # Top fonctions par temps cumulé
    print(f"\n{'='*80}")
    print("TOP 30 FONCTIONS PAR TEMPS CUMULÉ")
    print(f"{'='*80}\n")
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    # Top fonctions par temps propre
    print(f"\n{'='*80}")
    print("TOP 30 FONCTIONS PAR TEMPS PROPRE (sans appels)")
    print(f"{'='*80}\n")
    stats.sort_stats("time")
    stats.print_stats(30)

    # Top fonctions par nombre d'appels
    print(f"\n{'='*80}")
    print("TOP 20 FONCTIONS PAR NOMBRE D'APPELS")
    print(f"{'='*80}\n")
    stats.sort_stats("calls")
    stats.print_stats(20)


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Profiler de performance pour Backtest Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Commande: simple
    parser_simple = subparsers.add_parser("simple", help="Profiler un backtest simple")
    parser_simple.add_argument("--strategy", required=True, help="Nom de la stratégie")
    parser_simple.add_argument("--symbol", default="BTCUSDT", help="Symbole (défaut: BTCUSDT)")
    parser_simple.add_argument("--timeframe", default="1h", help="Timeframe (défaut: 1h)")
    parser_simple.add_argument("--start", default="2024-01-01", help="Date de début")
    parser_simple.add_argument("--end", default="2024-12-31", help="Date de fin")

    # Commande: grid
    parser_grid = subparsers.add_parser("grid", help="Profiler une optimisation Grid Search")
    parser_grid.add_argument("--strategy", required=True, help="Nom de la stratégie")
    parser_grid.add_argument("--combinations", type=int, default=50, help="Nombre de combinaisons")
    parser_grid.add_argument("--symbol", default="BTCUSDT", help="Symbole (défaut: BTCUSDT)")
    parser_grid.add_argument("--timeframe", default="1h", help="Timeframe (défaut: 1h)")
    parser_grid.add_argument("--start", default="2024-01-01", help="Date de début")
    parser_grid.add_argument("--end", default="2024-12-31", help="Date de fin")

    # Commande: analyze
    parser_analyze = subparsers.add_parser("analyze", help="Analyser un rapport de profiling")
    parser_analyze.add_argument("--report", required=True, help="Chemin vers le fichier .prof")

    # Commande: list
    subparsers.add_parser("list", help="Lister les stratégies disponibles")

    args = parser.parse_args()

    if args.command == "simple":
        profile_single_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
        )

    elif args.command == "grid":
        profile_grid_search(
            strategy_name=args.strategy,
            n_combinations=args.combinations,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
        )

    elif args.command == "analyze":
        analyze_profile_report(args.report)

    elif args.command == "list":
        print("\n📋 Stratégies disponibles:")
        for strategy in list_strategies():
            print(f"   - {strategy}")
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
