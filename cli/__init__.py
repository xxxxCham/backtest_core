"""
Module-ID: cli.__init__

Purpose: Package CLI - parser argparse, routing commands, entry point.

Role in pipeline: CLI interface

Key components: create_parser(), add_subcommands(), main()

Inputs: sys.argv command-line args

Outputs: Dispatched to cmd_* functions

Dependencies: argparse, .commands

Conventions: Sous-commandes via add_parser(); --verbose/-v global; help auto-generated.

Read-if: Ajout/modification sous-commande ou argument structure.

Skip-if: Vous appelez main() depuis __main__.py.
"""

import argparse
from typing import Optional

from .commands import (
    cmd_analyze,
    cmd_backtest,
    cmd_check_gpu,
    cmd_export,
    cmd_grid_backtest,
    cmd_indicators,
    cmd_info,
    cmd_list,
    cmd_llm_optimize,
    cmd_optuna,
    cmd_sweep,
    cmd_validate,
    cmd_visualize,
)


def create_parser() -> argparse.ArgumentParser:
    """Crée le parser principal avec toutes les sous-commandes."""

    parser = argparse.ArgumentParser(
        prog="backtest_core",
        description="Moteur de backtesting pour stratégies de trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s list strategies              Lister toutes les stratégies
  %(prog)s list indicators              Lister tous les indicateurs
  %(prog)s info strategy bollinger_atr  Détails d'une stratégie
  %(prog)s backtest -s ema_cross -d data.parquet
  %(prog)s sweep -s ema_cross -d data.parquet --granularity 0.3
        """
    )

    # Parser parent avec arguments communs
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbose (debug)"
    )
    common_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux"
    )
    common_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Désactiver les couleurs"
    )
    common_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )
    common_parser.add_argument(
        "--config",
        type=str,
        help="Fichier de configuration TOML"
    )

    # Sous-commandes
    subparsers = parser.add_subparsers(
        title="Commandes",
        dest="command",
        description="Commandes disponibles"
    )

    # === LIST ===
    list_parser = subparsers.add_parser(
        "list",
        parents=[common_parser],
        help="Lister les ressources disponibles",
        description="Liste les stratégies, indicateurs, données ou presets"
    )
    list_parser.add_argument(
        "resource",
        choices=["strategies", "indicators", "data", "presets"],
        help="Type de ressource à lister"
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie au format JSON"
    )

    # === INDICATORS (alias list indicators) ===
    indicators_parser = subparsers.add_parser(
        "indicators",
        parents=[common_parser],
        help="Lister les indicateurs disponibles",
        description="Alias de: list indicators"
    )
    indicators_parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie au format JSON"
    )

    # === INFO ===
    info_parser = subparsers.add_parser(
        "info",
        parents=[common_parser],
        help="Informations détaillées sur une ressource",
        description="Affiche les paramètres et documentation d'une stratégie ou indicateur"
    )
    info_parser.add_argument(
        "resource_type",
        choices=["strategy", "indicator"],
        help="Type de ressource"
    )
    info_parser.add_argument(
        "name",
        help="Nom de la ressource"
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie au format JSON"
    )

    # === BACKTEST ===
    backtest_parser = subparsers.add_parser(
        "backtest",
        parents=[common_parser],
        help="Exécuter un backtest",
        description="Lance un backtest avec une stratégie et des données"
    )
    backtest_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Nom de la stratégie"
    )
    backtest_parser.add_argument(
        "-d", "--data",
        required=True,
        help="Chemin vers le fichier de données OHLCV"
    )
    backtest_parser.add_argument(
        "--start",
        type=str,
        help="Date de debut (format ISO)"
    )
    backtest_parser.add_argument(
        "--end",
        type=str,
        help="Date de fin (format ISO)"
    )
    backtest_parser.add_argument(
        "--symbol",
        type=str,
        help="Symbole (override si non present dans le nom du fichier)"
    )
    backtest_parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe (override si non present dans le nom du fichier)"
    )
    backtest_parser.add_argument(
        "-p", "--params",
        type=str,
        default="{}",
        help="Paramètres stratégie en JSON (défaut: {})"
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    backtest_parser.add_argument(
        "--fees-bps",
        type=int,
        default=10,
        help="Frais en basis points (défaut: 10 = 0.1%%)"
    )
    backtest_parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Slippage en basis points (defaut: config)"
    )
    backtest_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour les résultats"
    )
    backtest_parser.add_argument(
        "--format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Format de sortie (défaut: json)"
    )

    # === SWEEP ===
    sweep_parser = subparsers.add_parser(
        "sweep",
        parents=[common_parser],
        help="Optimisation paramétrique",
        description="Lance une optimisation sur grille de paramètres",
        aliases=["optimize"]
    )
    sweep_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Nom de la stratégie"
    )
    sweep_parser.add_argument(
        "-d", "--data",
        required=True,
        help="Chemin vers le fichier de données OHLCV"
    )
    sweep_parser.add_argument(
        "--start",
        type=str,
        help="Date de debut (format ISO)"
    )
    sweep_parser.add_argument(
        "--end",
        type=str,
        help="Date de fin (format ISO)"
    )
    sweep_parser.add_argument(
        "--symbol",
        type=str,
        help="Symbole (override si non present dans le nom du fichier)"
    )
    sweep_parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe (override si non present dans le nom du fichier)"
    )
    sweep_parser.add_argument(
        "-g", "--granularity",
        type=float,
        default=0.5,
        help="Granularité (0.0=fin, 1.0=grossier, défaut: 0.5)"
    )
    sweep_parser.add_argument(
        "--include-optional-params",
        action="store_true",
        help="Inclure les paramètres optionnels (ex: leverage) dans la grille"
    )
    sweep_parser.add_argument(
        "--max-combinations",
        type=int,
        default=10000,
        help="Limite de combinaisons (défaut: 10000)"
    )
    sweep_parser.add_argument(
        "-m", "--metric",
        choices=["sharpe", "sharpe_ratio", "sortino", "sortino_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor"],
        default="sharpe",
        help="Métrique d'optimisation. Accepte sharpe/sharpe_ratio, sortino/sortino_ratio (défaut: sharpe)"
    )
    sweep_parser.add_argument(

        "--parallel",
        type=int,
        default=4,
        help="Nombre de workers parallèles (défaut: 12)"
    )
    sweep_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour les résultats"
    )
    sweep_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    sweep_parser.add_argument(
        "--fees-bps",
        type=int,
        default=10,
        help="Frais en basis points (défaut: 10)"
    )
    sweep_parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Slippage en basis points (defaut: config)"
    )
    sweep_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Nombre de meilleurs résultats à afficher (défaut: 10)"
    )

    # === VALIDATE ===
    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help="Valider configuration",
        description="Vérifie l'intégrité des stratégies, indicateurs et données"
    )
    validate_parser.add_argument(
        "--strategy",
        type=str,
        help="Valider une stratégie spécifique"
    )
    validate_parser.add_argument(
        "--data",
        type=str,
        help="Valider un fichier de données"
    )
    validate_parser.add_argument(
        "--all",
        action="store_true",
        help="Valider tout le système"
    )

    # === EXPORT ===
    export_parser = subparsers.add_parser(
        "export",
        parents=[common_parser],
        help="Exporter résultats",
        description="Exporte les résultats dans différents formats"
    )
    export_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Fichier de résultats à exporter"
    )
    export_parser.add_argument(
        "-f", "--format",
        choices=["html", "excel", "csv"],
        default="html",
        help="Format d'export (défaut: html)"
    )
    export_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie"
    )
    export_parser.add_argument(
        "--template",
        type=str,
        help="Template de rapport personnalisé"
    )

    # === OPTUNA ===
    optuna_parser = subparsers.add_parser(
        "optuna",
        parents=[common_parser],
        help="Optimisation bayésienne via Optuna",
        description="Lance une optimisation intelligente des paramètres (10-100x plus rapide que sweep)"
    )
    optuna_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Nom de la stratégie"
    )
    optuna_parser.add_argument(
        "-d", "--data",
        required=True,
        help="Chemin vers le fichier de données OHLCV"
    )
    optuna_parser.add_argument(
        "--start",
        type=str,
        help="Date de debut (format ISO)"
    )
    optuna_parser.add_argument(
        "--end",
        type=str,
        help="Date de fin (format ISO)"
    )
    optuna_parser.add_argument(
        "--symbol",
        type=str,
        help="Symbole (override si non present dans le nom du fichier)"
    )
    optuna_parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe (override si non present dans le nom du fichier)"
    )
    optuna_parser.add_argument(
        "-n", "--n-trials",
        type=int,
        default=100,
        help="Nombre de trials (défaut: 100)"
    )
    optuna_parser.add_argument(
        "-m", "--metric",
        default="sharpe",
        help="Métrique à optimiser. Multi-objectif: 'sharpe,max_drawdown' (défaut: sharpe)"
    )
    optuna_parser.add_argument(
        "--sampler",
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Algorithme de sampling (défaut: tpe)"
    )
    optuna_parser.add_argument(
        "--pruning",
        action="store_true",
        help="Activer le pruning (arrêt précoce des trials peu prometteurs)"
    )
    optuna_parser.add_argument(
        "--pruner",
        choices=["median", "hyperband"],
        default="median",
        help="Type de pruner (défaut: median)"
    )
    optuna_parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="Mode multi-objectif (Pareto). Utiliser -m 'sharpe,max_drawdown'"
    )
    optuna_parser.add_argument(
        "--param-space",
        type=str,
        help="Espace de paramètres en JSON (sinon auto-détecté)"
    )
    optuna_parser.add_argument(
        "-c", "--constraints",
        nargs="*",
        help="Contraintes: 'slow_period,>,fast_period' (param1,op,param2)"
    )
    optuna_parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout en secondes (optionnel)"
    )
    optuna_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Nombre de jobs parallèles (défaut: 1, utiliser prudemment)"
    )
    optuna_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    optuna_parser.add_argument(
        "--fees-bps",
        type=int,
        default=10,
        help="Frais en basis points (défaut: 10)"
    )
    optuna_parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Slippage en basis points (defaut: config)"
    )
    optuna_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Nombre de meilleurs résultats à afficher (défaut: 10)"
    )
    optuna_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour les résultats"
    )
    optuna_parser.add_argument(
        "--early-stop-patience",
        type=int,
        help="Arrêt anticipé après N trials sans amélioration (None = désactivé)"
    )

    # === VISUALIZE ===
    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=[common_parser],
        help="Visualiser les résultats de backtest",
        description="Génère des graphiques interactifs (candlesticks + trades)"
    )
    visualize_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Fichier de résultats à visualiser (JSON)"
    )
    visualize_parser.add_argument(
        "-d", "--data",
        type=str,
        help="Fichier de données OHLCV pour les candlesticks"
    )
    visualize_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier HTML de sortie"
    )
    visualize_parser.add_argument(
        "--html",
        action="store_true",
        help="Générer automatiquement un fichier HTML"
    )
    visualize_parser.add_argument(
        "-m", "--metric",
        type=str,
        help="Métrique pour sélectionner le meilleur (pour sweep/optuna)"
    )
    visualize_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    visualize_parser.add_argument(
        "--fees-bps",
        type=int,
        default=10,
        help="Frais en basis points (défaut: 10)"
    )
    visualize_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Ne pas ouvrir le graphique dans le navigateur"
    )

    # === CHECK-GPU ===
    check_gpu_parser = subparsers.add_parser(
        "check-gpu",
        parents=[common_parser],
        help="Diagnostic GPU et benchmark",
        description="Vérifie CuPy, CUDA, GPUs disponibles et benchmark CPU vs GPU"
    )
    check_gpu_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Exécuter un benchmark CPU vs GPU (EMA 10k points)"
    )

    # === LLM-OPTIMIZE ===
    llm_optimize_parser = subparsers.add_parser(
        "llm-optimize",
        parents=[common_parser],
        help="Optimisation LLM multi-agents",
        description="Lance l'orchestrateur multi-agents (Analyst/Strategist/Critic/Validator) pour optimisation intelligente",
        aliases=["orchestrate"]
    )
    llm_optimize_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Nom de la stratégie"
    )
    llm_optimize_parser.add_argument(
        "--symbol",
        required=True,
        help="Symbole (ex: BTCUSDC)"
    )
    llm_optimize_parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe (ex: 1h, 30m, 1d)"
    )
    llm_optimize_parser.add_argument(
        "--start",
        type=str,
        help="Date de début (format ISO)"
    )
    llm_optimize_parser.add_argument(
        "--end",
        type=str,
        help="Date de fin (format ISO)"
    )
    llm_optimize_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    llm_optimize_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Nombre max d'itérations LLM (défaut: 10)"
    )
    llm_optimize_parser.add_argument(
        "--model",
        default="deepseek-r1-distill:14b",
        help="Modèle LLM à utiliser (défaut: deepseek-r1-distill:14b)"
    )
    llm_optimize_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température LLM (défaut: 0.7)"
    )
    llm_optimize_parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens LLM (défaut: 4096)"
    )
    llm_optimize_parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout LLM en secondes (défaut: 900 = 15min)"
    )
    llm_optimize_parser.add_argument(
        "--min-sharpe",
        type=float,
        default=1.0,
        help="Sharpe ratio minimum requis (défaut: 1.0)"
    )
    llm_optimize_parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.20,
        help="Max drawdown limite (fraction, défaut: 0.20 = 20%%)"
    )
    llm_optimize_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour les résultats"
    )

    # === GRID-BACKTEST ===
    grid_backtest_parser = subparsers.add_parser(
        "grid-backtest",
        parents=[common_parser],
        help="Backtest en mode grille",
        description="Exécute un backtest sur une grille de paramètres (différent de sweep)",
        aliases=["grid"]
    )
    grid_backtest_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Nom de la stratégie"
    )
    grid_backtest_parser.add_argument(
        "--symbol",
        required=True,
        help="Symbole (ex: BTCUSDC)"
    )
    grid_backtest_parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe (ex: 1h, 30m, 1d)"
    )
    grid_backtest_parser.add_argument(
        "--start",
        type=str,
        help="Date de début (format ISO)"
    )
    grid_backtest_parser.add_argument(
        "--end",
        type=str,
        help="Date de fin (format ISO)"
    )
    grid_backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital initial (défaut: 10000)"
    )
    grid_backtest_parser.add_argument(
        "--fees-bps",
        type=int,
        default=10,
        help="Frais en basis points (défaut: 10)"
    )
    grid_backtest_parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Slippage en basis points (défaut: config)"
    )
    grid_backtest_parser.add_argument(
        "--param-grid",
        type=str,
        help="Grille de paramètres en JSON (ex: '{\"atr_period\": [10, 14, 20]}'). Si omis, grille auto depuis param_ranges"
    )
    grid_backtest_parser.add_argument(
        "--include-optional-params",
        action="store_true",
        help="Inclure les paramètres optionnels (ex: leverage) dans la grille auto"
    )
    grid_backtest_parser.add_argument(
        "--max-combinations",
        type=int,
        default=1000,
        help="Limite de combinaisons (défaut: 1000)"
    )
    grid_backtest_parser.add_argument(
        "-m", "--metric",
        choices=["sharpe_ratio", "sortino_ratio", "total_return_pct", "max_drawdown", "win_rate", "profit_factor"],
        default="sharpe_ratio",
        help="Métrique pour trier les résultats (défaut: sharpe_ratio)"
    )
    grid_backtest_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Nombre de meilleurs résultats à afficher (défaut: 10)"
    )
    grid_backtest_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour les résultats"
    )

    # === ANALYZE ===
    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common_parser],
        help="Analyser les résultats de backtests",
        description="Analyse les résultats de backtests stockés dans backtest_results/"
    )
    analyze_parser.add_argument(
        "--results-dir",
        type=str,
        default="backtest_results",
        help="Répertoire des résultats (défaut: backtest_results)"
    )
    analyze_parser.add_argument(
        "--profitable-only",
        action="store_true",
        help="Afficher uniquement les runs profitables"
    )
    analyze_parser.add_argument(
        "--sort-by",
        type=str,
        default="total_pnl",
        help="Métrique de tri (défaut: total_pnl)"
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Nombre de runs à afficher (défaut: 10)"
    )
    analyze_parser.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques globales"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie pour l'analyse"
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """Point d'entrée principal du CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Si aucune commande, afficher l'aide
    if parsed.command is None:
        parser.print_help()
        return 0

    # Configuration globale
    import numpy as np
    np.random.seed(parsed.seed)

    # Dispatcher vers la commande appropriée
    commands = {
        "list": cmd_list,
        "indicators": cmd_indicators,
        "info": cmd_info,
        "backtest": cmd_backtest,
        "sweep": cmd_sweep,
        "optimize": cmd_sweep,
        "optuna": cmd_optuna,
        "validate": cmd_validate,
        "export": cmd_export,
        "visualize": cmd_visualize,
        "check-gpu": cmd_check_gpu,
        "llm-optimize": cmd_llm_optimize,
        "orchestrate": cmd_llm_optimize,
        "grid-backtest": cmd_grid_backtest,
        "grid": cmd_grid_backtest,
        "analyze": cmd_analyze,
    }

    try:
        handler = commands.get(parsed.command)
        if handler:
            return handler(parsed)
        else:
            print(f"Commande inconnue: {parsed.command}")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu par l'utilisateur")
        return 130
    except Exception as e:
        if parsed.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Erreur: {e}")
        return 1


__all__ = ["main", "create_parser"]
