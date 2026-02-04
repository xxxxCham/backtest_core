"""
Module-ID: cli.commands

Purpose: Implémentation CLI commands - backtest, sweep, optuna, validate, export, visualize.

Role in pipeline: CLI interface

Key components: cmd_backtest(), cmd_sweep(), cmd_optuna(), Colors, normalize_metric_name(), METRIC_ALIASES

Inputs: argparse parsed args (strategy, data, params, etc.)

Outputs: Console output, JSON results, HTML/CSV exports, visualization

Dependencies: argparse, json, pathlib, pandas, numpy

Conventions: Metric aliases (sharpe → sharpe_ratio); couleurs ANSI (désactivable --no-color); progress bars.

Read-if: Ajout commande CLI ou modification format output.

Skip-if: Vous appelez cmd_backtest(args) depuis main.
"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# =============================================================================
# UTILITAIRES
# =============================================================================

# Système d'alias pour métriques CLI
METRIC_ALIASES = {
    "sharpe": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "total_return": "total_return_pct",
    # Accepter aussi les formes complètes
    "sharpe_ratio": "sharpe_ratio",
    "sortino_ratio": "sortino_ratio",
    "total_return_pct": "total_return_pct",
}


def normalize_metric_name(metric: str) -> str:
    """Normalise le nom d'une métrique CLI en nom interne."""
    return METRIC_ALIASES.get(metric, metric)


class Colors:
    """Codes couleurs ANSI pour le terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @classmethod
    def disable(cls):
        """Désactive les couleurs."""
        cls.RESET = cls.BOLD = cls.RED = cls.GREEN = ""
        cls.YELLOW = cls.BLUE = cls.MAGENTA = cls.CYAN = ""


def print_header(text: str, char: str = "="):
    """Affiche un en-tête formaté."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(Colors.CYAN + char * len(text) + Colors.RESET)


def print_success(text: str):
    """Affiche un message de succès."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    """Affiche un message d'erreur."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    """Affiche un avertissement."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """Affiche une information."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def format_table(headers: List[str], rows: List[List[str]], padding: int = 2) -> str:
    """Formate une table en texte."""
    if not rows:
        return "  (aucune donnée)"

    # Calculer largeurs
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Header
    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(f"  {Colors.BOLD}{header_line}{Colors.RESET}")
    lines.append("  " + "  ".join("-" * w for w in widths))

    # Rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(f"  {row_line}")

    return "\n".join(lines)


def format_bytes(bytes_count: float) -> str:
    """Formate un nombre de bytes en unité lisible."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


def _apply_date_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Filtre un DataFrame OHLCV par date (UTC)."""
    if not start and not end:
        return df

    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[df.index <= end_ts]

    if df.empty:
        raise ValueError(f"Aucune donnée dans la période {start} - {end}")

    return df


# =============================================================================
# COMMANDE: LIST
# =============================================================================

def cmd_list(args) -> int:
    """Liste les ressources disponibles."""
    if args.no_color:
        Colors.disable()

    resource = args.resource

    if resource == "strategies":
        return _list_strategies(args)
    elif resource == "indicators":
        return _list_indicators(args)
    elif resource == "data":
        return _list_data(args)
    elif resource == "presets":
        return _list_presets(args)

    return 0


def cmd_indicators(args) -> int:
    """Alias: list indicators."""
    args.resource = "indicators"
    return cmd_list(args)


def _list_strategies(args) -> int:
    """Liste les stratégies."""
    from strategies import get_strategy, list_strategies

    strategies = list_strategies()

    if args.json:
        data = []
        for name in strategies:
            strat = get_strategy(name)
            if strat:
                instance = strat()
                data.append({
                    "name": name,
                    "description": getattr(instance, "description", ""),
                    "indicators": getattr(instance, "required_indicators", []),
                })
        print(json.dumps(data, indent=2))
        return 0

    print_header(f"Stratégies disponibles ({len(strategies)})")

    rows = []
    for name in sorted(strategies):
        strat = get_strategy(name)
        if strat:
            instance = strat()
            desc = getattr(instance, "description", "")[:50]
            indicators = ", ".join(instance.required_indicators[:3])
            if len(instance.required_indicators) > 3:
                indicators += "..."
            rows.append([name, desc, indicators])

    print(format_table(["Nom", "Description", "Indicateurs"], rows))
    return 0


def _list_indicators(args) -> int:
    """Liste les indicateurs."""
    from indicators.registry import get_indicator, list_indicators

    indicators = list_indicators()

    if args.json:
        data = []
        for name in indicators:
            info = get_indicator(name)
            if info:
                data.append({
                    "name": name,
                    "description": info.description,
                    "required_columns": list(info.required_columns),
                })
        print(json.dumps(data, indent=2))
        return 0

    print_header(f"Indicateurs disponibles ({len(indicators)})")

    rows = []
    for name in sorted(indicators):
        info = get_indicator(name)
        if info:
            cols = ", ".join(info.required_columns)
            desc = info.description[:40] if info.description else ""
            rows.append([name, cols, desc])

    print(format_table(["Nom", "Colonnes", "Description"], rows))
    return 0


def _list_data(args) -> int:
    """Liste les fichiers de données."""
    import os

    from data.loader import discover_available_data

    # Récupérer tokens et timeframes
    tokens, timeframes = discover_available_data()

    # Chercher les fichiers via variable d'environnement ou répertoire par défaut
    env_data_dir = os.environ.get("BACKTEST_DATA_DIR")
    if env_data_dir:
        data_dir = Path(env_data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "sample_data"

    data_files = []
    if data_dir.exists():
        # Format parquet uniquement (selon variable d'environnement)
        data_files.extend(data_dir.glob("*.parquet"))
        # Aussi chercher CSV et Feather comme fallback
        data_files.extend(data_dir.glob("*.csv"))
        data_files.extend(data_dir.glob("*.feather"))

    if args.json:
        print(json.dumps({
            "data_dir": str(data_dir),
            "tokens": tokens,
            "timeframes": timeframes,
            "files": [str(f) for f in data_files]
        }, indent=2))
        return 0

    print_header(f"Fichiers de données ({len(data_files)})")

    if env_data_dir:
        print_info(f"Répertoire: {data_dir} (via $BACKTEST_DATA_DIR)")
    else:
        print_info(f"Répertoire: {data_dir}")

    if not data_files:
        print_warning("Aucun fichier de données trouvé")
        if not env_data_dir:
            print_info("Définissez $env:BACKTEST_DATA_DIR ou placez des fichiers .parquet dans data/sample_data/")
        return 0

    rows = []
    for f in sorted(data_files):
        path = Path(f)
        size = path.stat().st_size / 1024 if path.exists() else 0
        rows.append([path.name, f"{size:.1f} KB", path.suffix])

    print(format_table(["Fichier", "Taille", "Format"], rows))

    # Afficher aussi les tokens et timeframes
    if tokens:
        print(f"\n{Colors.BOLD}Tokens:{Colors.RESET} {', '.join(tokens)}")
    if timeframes:
        print(f"{Colors.BOLD}Timeframes:{Colors.RESET} {', '.join(timeframes)}")

    return 0


def _list_presets(args) -> int:
    """Liste les presets."""
    from utils.parameters import EMA_CROSS_PRESET, MINIMAL_PRESET, SAFE_RANGES_PRESET

    presets = [SAFE_RANGES_PRESET, MINIMAL_PRESET, EMA_CROSS_PRESET]

    if args.json:
        data = [p.to_dict() for p in presets]
        print(json.dumps(data, indent=2))
        return 0

    print_header(f"Presets disponibles ({len(presets)})")

    rows = []
    for p in presets:
        n_params = len(p.parameters)
        combos = p.estimate_combinations()
        rows.append([p.name, p.description[:40], str(n_params), f"~{combos:,}"])

    print(format_table(["Nom", "Description", "Params", "Combinaisons"], rows))
    return 0


# =============================================================================
# COMMANDE: INFO
# =============================================================================

def cmd_info(args) -> int:
    """Affiche les informations détaillées d'une ressource."""
    if args.no_color:
        Colors.disable()

    if args.resource_type == "strategy":
        return _info_strategy(args)
    elif args.resource_type == "indicator":
        return _info_indicator(args)

    return 0


def _info_strategy(args) -> int:
    """Affiche les infos d'une stratégie."""
    from strategies import get_strategy, list_strategies

    name = args.name.lower()
    strat_class = get_strategy(name)

    if not strat_class:
        print_error(f"Stratégie '{name}' non trouvée")
        print_info(f"Disponibles: {', '.join(list_strategies())}")
        return 1

    strat = strat_class()

    if getattr(args, "include_optional_params", False):
        strat._include_optional_params = True

    optional_skipped: List[str] = []
    if hasattr(strat, "parameter_specs") and strat.parameter_specs:
        optional_skipped = [
            name
            for name, spec in strat.parameter_specs.items()
            if not getattr(spec, "optimize", True)
        ]

    if optional_skipped and not getattr(strat, "_include_optional_params", False):
        if not args.quiet:
            skipped = ", ".join(optional_skipped)
            print_info(
                f"Paramètres optionnels ignorés: {skipped} (ajoutez --include-optional-params ou BACKTEST_INCLUDE_OPTIONAL_PARAMS=1)"
            )

    if args.json:
        data = {
            "name": name,
            "description": getattr(strat, "description", ""),
            "required_indicators": strat.required_indicators,
            "default_params": strat.default_params,
            "param_ranges": strat.param_ranges,
        }
        print(json.dumps(data, indent=2))
        return 0

    print_header(f"Stratégie: {name}")
    print(f"  Description: {getattr(strat, 'description', 'N/A')}")
    print(f"  Indicateurs: {', '.join(strat.required_indicators)}")

    print(f"\n{Colors.BOLD}Paramètres par défaut:{Colors.RESET}")
    for k, v in strat.default_params.items():
        print(f"    {k}: {v}")

    print(f"\n{Colors.BOLD}Plages d'optimisation:{Colors.RESET}")
    for k, (min_v, max_v) in strat.param_ranges.items():
        print(f"    {k}: [{min_v}, {max_v}]")

    return 0


def _info_indicator(args) -> int:
    """Affiche les infos d'un indicateur."""
    from indicators.registry import get_indicator, list_indicators

    name = args.name.lower()
    info = get_indicator(name)

    if not info:
        print_error(f"Indicateur '{name}' non trouvé")
        print_info(f"Disponibles: {', '.join(list_indicators())}")
        return 1

    if args.json:
        data = {
            "name": info.name,
            "description": info.description,
            "required_columns": list(info.required_columns),
            "settings_class": info.settings_class.__name__ if info.settings_class else None,
        }
        print(json.dumps(data, indent=2))
        return 0

    print_header(f"Indicateur: {name}")
    print(f"  Description: {info.description}")
    print(f"  Colonnes requises: {', '.join(info.required_columns)}")

    if info.settings_class:
        print(f"\n{Colors.BOLD}Paramètres (Settings):{Colors.RESET}")
        import inspect
        sig = inspect.signature(info.settings_class)
        for param_name, param in sig.parameters.items():
            if param_name != "self":
                default = param.default if param.default != inspect.Parameter.empty else "requis"
                print(f"    {param_name}: {default}")

    return 0


# =============================================================================
# COMMANDE: BACKTEST
# =============================================================================

def cmd_backtest(args) -> int:
    """Exécute un backtest."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    import os
    from pathlib import Path

    from backtest.engine import BacktestEngine
    from strategies import get_strategy, list_strategies

    # Validation stratégie
    strategy_name = args.strategy.lower()
    strat_class = get_strategy(strategy_name)

    if not strat_class:
        print_error(f"Stratégie '{strategy_name}' non trouvée")
        print_info(f"Disponibles: {', '.join(list_strategies())}")
        return 1

    # Validation données - chercher dans BACKTEST_DATA_DIR si chemin relatif
    data_path = Path(args.data)
    if not data_path.exists():
        # Essayer avec le répertoire de données
        env_data_dir = os.environ.get("BACKTEST_DATA_DIR")
        if env_data_dir:
            data_path = Path(env_data_dir) / args.data
        else:
            data_path = Path(__file__).parent.parent / "data" / "sample_data" / args.data

    if not data_path.exists():
        print_error(f"Fichier non trouvé: {args.data}")
        print_info(f"Répertoire de recherche: {data_path.parent}")
        return 1

    # Parser paramètres JSON
    try:
        params = json_module.loads(args.params)
    except json_module.JSONDecodeError as e:
        print_error(f"Paramètres JSON invalides: {e}")
        return 1

    if not args.quiet:
        print_header("Backtest")
        print(f"  Stratégie: {strategy_name}")
        print(f"  Données: {data_path}")
        print(f"  Capital: {args.capital:,.0f}")
        print(f"  Frais: {args.fees_bps} bps ({args.fees_bps/100:.2f}%)")
        if params:
            print(f"  Paramètres: {params}")
        print()

    # Chargement données
    if not args.quiet:
        print_info("Chargement des données...")

    # Utiliser les fonctions internes pour charger directement depuis le fichier
    from data.loader import _normalize_ohlcv, _read_file
    df = _read_file(data_path)
    df = _normalize_ohlcv(df)
    try:
        df = _apply_date_filter(df, args.start, args.end)
    except ValueError as e:
        print_error(str(e))
        return 1

    if not args.quiet:
        print_success(f"Données chargées: {len(df)} barres")

    # Exécution backtest
    if not args.quiet:
        print_info("Exécution du backtest...")

    # Créer la configuration avec les frais
    from utils.config import Config
    config_kwargs = {"fees_bps": args.fees_bps}
    if args.slippage_bps is not None:
        config_kwargs["slippage_bps"] = args.slippage_bps
    config = Config(**config_kwargs)

    engine = BacktestEngine(
        initial_capital=args.capital,
        config=config,
    )

    # Extraire symbol et timeframe du nom de fichier (ex: BTCUSDC_1h.parquet)
    stem = data_path.stem
    parts = stem.split("_")
    symbol = args.symbol or (parts[0] if parts else "UNKNOWN")
    timeframe = args.timeframe or (parts[1] if len(parts) > 1 else "1h")

    result = engine.run(
        df=df,
        strategy=strategy_name,
        params=params,
        symbol=symbol,
        timeframe=timeframe
    )

    # Affichage résultats
    if not args.quiet:
        print()
        print_header("Résultats")
        _print_metrics(result.metrics)
        if result.meta.get("period_start") and result.meta.get("period_end"):
            print(f"    Period: {result.meta['period_start']} -> {result.meta['period_end']}")
        print(f"\n  Trades: {len(result.trades)}")

    # Export si demandé
    if args.output:
        output_path = Path(args.output)

        # Gérer metrics qui peut être un dict ou un objet avec to_dict()
        if hasattr(result.metrics, 'to_dict'):
            metrics_dict = result.metrics.to_dict()
        elif isinstance(result.metrics, dict):
            metrics_dict = result.metrics
        else:
            metrics_dict = vars(result.metrics)

        # Gérer trades qui est un DataFrame
        if isinstance(result.trades, pd.DataFrame):
            trades_list = result.trades.to_dict('records')
        elif result.trades and len(result.trades) > 0:
            first_trade = result.trades[0]
            if hasattr(first_trade, 'to_dict'):
                trades_list = [t.to_dict() for t in result.trades]
            elif hasattr(first_trade, '__dict__'):
                trades_list = [vars(t) for t in result.trades]
            else:
                trades_list = list(result.trades)
        else:
            trades_list = []

        output_data = {
            "strategy": strategy_name,
            "params": params,
            "capital": args.capital,
            "fees_bps": args.fees_bps,
            "meta": result.meta,
            "metrics": metrics_dict,
            "n_trades": len(result.trades),
            "trades": trades_list,
        }

        if args.format == "json":
            with open(output_path, "w") as f:
                json_module.dump(output_data, f, indent=2, default=str)
        elif args.format == "csv":
            pd.DataFrame(result.trades).to_csv(output_path, index=False)
        elif args.format == "parquet":
            pd.DataFrame(result.trades).to_parquet(output_path)

        if not args.quiet:
            print_success(f"Résultats exportés: {output_path}")

    return 0


def _print_metrics(metrics):
    """Affiche les métriques de performance."""
    m = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics

    total_return_pct = m.get("total_return_pct")
    if total_return_pct is None:
        total_return_pct = m.get("total_return", 0) * 100
    max_drawdown_pct = m.get("max_drawdown", 0)
    win_rate_pct = m.get("win_rate", 0)

    print(f"  {Colors.BOLD}Performance:{Colors.RESET}")
    print(f"    Total Return: {total_return_pct:+.2f}%")
    print(f"    Sharpe Ratio: {m.get('sharpe_ratio', 0):.3f}")
    print(f"    Sortino Ratio: {m.get('sortino_ratio', 0):.3f}")
    print(f"    Max Drawdown: {max_drawdown_pct:.2f}%")
    print(f"    Win Rate: {win_rate_pct:.1f}%")
    print(f"    Profit Factor: {m.get('profit_factor', 0):.2f}")


# =============================================================================
# COMMANDE: SWEEP
# =============================================================================

def cmd_sweep(args) -> int:
    """Exécute une optimisation paramétrique."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from pathlib import Path

    from backtest.engine import BacktestEngine
    from strategies import get_strategy
    from utils.parameters import ParameterSpec, compute_search_space_stats, generate_param_grid

    # Validation stratégie
    strategy_name = args.strategy.lower()
    strat_class = get_strategy(strategy_name)

    if not strat_class:
        print_error(f"Stratégie '{strategy_name}' non trouvée")
        return 1

    # Résolution du chemin des données
    data_path = Path(args.data)
    if not data_path.exists():
        env_data_dir = os.environ.get("BACKTEST_DATA_DIR")
        if env_data_dir:
            data_path = Path(env_data_dir) / args.data
        else:
            data_path = Path(__file__).parent.parent / "data" / "sample_data" / args.data

    if not data_path.exists():
        print_error(f"Fichier non trouvé: {args.data}")
        return 1

    strat = strat_class()

    if not args.quiet:
        print_header("Optimisation Paramétrique (Sweep)")
        print(f"  Stratégie: {strategy_name}")
        print(f"  Données: {data_path}")
        print(f"  Granularité: {args.granularity}")
        print(f"  Métrique: {args.metric}")
        print(f"  Workers: {args.parallel}")
        print()

    # Construire les specs de paramètres
    param_specs = {}
    for name, (min_v, max_v) in strat.param_ranges.items():
        default = strat.default_params.get(name, (min_v + max_v) / 2)
        param_type = "int" if isinstance(default, int) else "float"
        param_specs[name] = ParameterSpec(
            name=name,
            min_val=min_v,
            max_val=max_v,
            default=default,
            param_type=param_type,
        )

    # Générer la grille
    try:
        grid = generate_param_grid(
            param_specs,
            granularity=args.granularity,
            max_total_combinations=args.max_combinations,
        )
    except ValueError as e:
        print_error(str(e))
        print_info("Augmentez --granularity ou --max-combinations")
        return 1

    # Afficher les statistiques d'espace de recherche (unifié)
    stats = compute_search_space_stats(param_specs, max_combinations=args.max_combinations, granularity=args.granularity)

    if not args.quiet:
        print_info(f"Espace de recherche: {stats.total_combinations:,} combinaisons")
        for pname, pcount in stats.per_param_counts.items():
            print(f"    {pname}: {pcount} valeurs")

    # Charger données avec les fonctions internes
    from data.loader import _normalize_ohlcv, _read_file
    df = _read_file(data_path)
    df = _normalize_ohlcv(df)
    try:
        df = _apply_date_filter(df, args.start, args.end)
    except ValueError as e:
        print_error(str(e))
        return 1

    if not args.quiet:
        print_success(f"Données chargées: {len(df)} barres")
        print_info("Lancement de l'optimisation...")
        print()

    # Extraire symbol et timeframe du nom de fichier
    stem = data_path.stem
    parts = stem.split("_")
    symbol = args.symbol or (parts[0] if parts else "UNKNOWN")
    timeframe = args.timeframe or (parts[1] if len(parts) > 1 else "1h")

    # Créer la configuration avec les frais
    from utils.config import Config
    config_kwargs = {"fees_bps": args.fees_bps}
    if args.slippage_bps is not None:
        config_kwargs["slippage_bps"] = args.slippage_bps
    config = Config(**config_kwargs)

    # Exécuter le sweep
    results = []

    for i, params in enumerate(grid):
        engine = BacktestEngine(
            initial_capital=args.capital,
            config=config,
        )

        try:
            result = engine.run(
                df=df,
                strategy=strategy_name,
                params=params,
                symbol=symbol,
                timeframe=timeframe
            )
            # Gérer les métriques qui peuvent être un dict ou un objet avec to_dict()
            if hasattr(result.metrics, 'to_dict'):
                metrics = result.metrics.to_dict()
            else:
                metrics = dict(result.metrics)

            # Normaliser le nom de la métrique
            metric_key = normalize_metric_name(args.metric)

            results.append({
                "params": params,
                "metrics": metrics,
                "score": metrics.get(metric_key, 0),
            })
        except Exception as e:
            if args.verbose:
                print_warning(f"Erreur avec {params}: {e}")

        # Progress
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"\r  Progress: {i+1}/{len(grid)} ({100*(i+1)/len(grid):.1f}%)", end="", flush=True)

    if not args.quiet:
        print("\r" + " " * 50 + "\r", end="")

    # Trier par score (métrique déjà normalisée)
    metric_key = normalize_metric_name(args.metric)
    reverse = args.metric != "max_drawdown"  # Plus bas = mieux pour drawdown

    results.sort(key=lambda x: x.get("score", 0), reverse=reverse)

    # Afficher les meilleurs
    if not args.quiet:
        print_header(f"Top {args.top} Résultats (tri par {args.metric})")

        for i, r in enumerate(results[:args.top]):
            print(f"\n  {Colors.BOLD}#{i+1}{Colors.RESET}")
            print(f"    Paramètres: {r['params']}")
            print(f"    Sharpe: {r['metrics'].get('sharpe_ratio', 0):.3f}")
            print(f"    Return: {r['metrics'].get('total_return_pct', 0):+.2f}%")
            print(f"    Drawdown: {r['metrics'].get('max_drawdown', 0):.2f}%")

    # Export
    if args.output:
        output_path = Path(args.output)

        export_data = {
            "strategy": strategy_name,
            "granularity": args.granularity,
            "metric": args.metric,
            "n_combinations": len(grid),
            "results": results,
        }

        with open(output_path, "w") as f:
            json_module.dump(export_data, f, indent=2, default=str)

        if not args.quiet:
            print()
            print_success(f"Résultats exportés: {output_path}")

    return 0


# =============================================================================
# COMMANDE: VALIDATE
# =============================================================================

def cmd_validate(args) -> int:
    """Valide la configuration."""
    if args.no_color:
        Colors.disable()

    print_header("Validation")

    errors = []
    warnings = []

    # Valider stratégies
    if args.all or args.strategy:
        from strategies import get_strategy, list_strategies

        strategies = [args.strategy] if args.strategy else list_strategies()

        print(f"\n{Colors.BOLD}Stratégies:{Colors.RESET}")
        for name in strategies:
            strat = get_strategy(name)
            if strat:
                try:
                    instance = strat()
                    # Vérifier les attributs requis
                    _ = instance.required_indicators
                    _ = instance.default_params
                    _ = instance.param_ranges
                    print_success(f"  {name}")
                except Exception as e:
                    print_error(f"  {name}: {e}")
                    errors.append(f"Stratégie {name}: {e}")
            else:
                print_error(f"  {name}: non trouvée")
                errors.append(f"Stratégie {name} non trouvée")

    # Valider indicateurs
    if args.all:
        from indicators.registry import get_indicator, list_indicators

        print(f"\n{Colors.BOLD}Indicateurs:{Colors.RESET}")
        for name in list_indicators():
            info = get_indicator(name)
            if info and info.function:
                print_success(f"  {name}")
            else:
                print_error(f"  {name}: fonction manquante")
                errors.append(f"Indicateur {name}: fonction manquante")

    # Valider données
    if args.all or args.data:
        from pathlib import Path

        from data.loader import _normalize_ohlcv, _read_file

        print(f"\n{Colors.BOLD}Données:{Colors.RESET}")

        if args.data:
            data_files = [args.data]
        else:
            data_dir = Path("data/sample_data")
            data_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))

        for f in data_files:
            try:
                df = _read_file(Path(f))
                df = _normalize_ohlcv(df)
                required = ["open", "high", "low", "close", "volume"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    print_warning(f"  {f}: colonnes manquantes {missing}")
                    warnings.append(f"{f}: colonnes manquantes {missing}")
                else:
                    print_success(f"  {f} ({len(df)} barres)")
            except Exception as e:
                print_error(f"  {f}: {e}")
                errors.append(f"{f}: {e}")

    # Résumé
    print()
    if errors:
        print_error(f"Validation échouée: {len(errors)} erreur(s)")
        return 1
    elif warnings:
        print_warning(f"Validation OK avec {len(warnings)} avertissement(s)")
        return 0
    else:
        print_success("Validation réussie!")
        return 0


# =============================================================================
# COMMANDE: EXPORT
# =============================================================================

def cmd_export(args) -> int:
    """Exporte les résultats."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from pathlib import Path

    input_path = Path(args.input)

    if not input_path.exists():
        print_error(f"Fichier non trouvé: {input_path}")
        return 1

    # Charger les résultats
    with open(input_path) as f:
        data = json_module.load(f)

    # Déterminer le fichier de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(f".{args.format}")

    if args.format == "html":
        _export_html(data, output_path)
    elif args.format == "csv":
        _export_csv(data, output_path)
    elif args.format == "excel":
        _export_excel(data, output_path)
    else:
        print_error(f"Format non supporté: {args.format}")
        return 1

    print_success(f"Export réussi: {output_path}")
    return 0


def _export_html(data: dict, output_path: Path):
    """Export en HTML."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 1.2em; margin: 10px 0; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>Rapport de Backtest</h1>
    <p>Stratégie: <strong>{data.get('strategy', 'N/A')}</strong></p>

    <h2>Métriques</h2>
    <div class="metrics">
"""

    metrics = data.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, float):
            css_class = "positive" if value > 0 else "negative"
            html += f'        <p class="metric">{key}: <span class="{css_class}">{value:.4f}</span></p>\n'

    html += """    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def _export_csv(data: dict, output_path: Path):
    """Export en CSV."""
    results = data.get("results", [])
    if results:
        rows = []
        for r in results:
            row = {**r.get("params", {}), **r.get("metrics", {})}
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        # Single backtest
        metrics = data.get("metrics", {})
        pd.DataFrame([metrics]).to_csv(output_path, index=False)


def _export_excel(data: dict, output_path: Path):
    """Export en Excel."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl requis pour export Excel: pip install openpyxl")

    results = data.get("results", [])
    if results:
        rows = []
        for r in results:
            row = {**r.get("params", {}), **r.get("metrics", {})}
            rows.append(row)
        pd.DataFrame(rows).to_excel(output_path, index=False)
    else:
        metrics = data.get("metrics", {})
        pd.DataFrame([metrics]).to_excel(output_path, index=False)


__all__ = [
    "cmd_list",
    "cmd_info",
    "cmd_backtest",
    "cmd_sweep",
    "cmd_validate",
    "cmd_export",
    "cmd_optuna",
    "cmd_visualize",
    "cmd_check_gpu",
    "cmd_llm_optimize",
    "cmd_grid_backtest",
    "cmd_analyze",
    "cmd_indicators",
]


# =============================================================================
# COMMANDE: OPTUNA
# =============================================================================

def cmd_optuna(args) -> int:
    """Exécute une optimisation bayésienne via Optuna."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from pathlib import Path

    # Vérifier que Optuna est disponible
    try:
        from backtest.optuna_optimizer import (
            OPTUNA_AVAILABLE,
            OptunaOptimizer,
            suggest_param_space,
        )
    except ImportError:
        print_error("Module optuna_optimizer non trouvé")
        return 1

    if not OPTUNA_AVAILABLE:
        print_error("Optuna n'est pas installé: pip install optuna")
        return 1

    from strategies import get_strategy, list_strategies

    # Validation stratégie
    strategy_name = args.strategy.lower()
    strat_class = get_strategy(strategy_name)

    if not strat_class:
        print_error(f"Stratégie '{strategy_name}' non trouvée")
        print_info(f"Stratégies disponibles: {', '.join(list_strategies())}")
        return 1

    # Résolution du chemin des données
    data_path = Path(args.data)
    if not data_path.exists():
        env_data_dir = os.environ.get("BACKTEST_DATA_DIR")
        if env_data_dir:
            data_path = Path(env_data_dir) / args.data
        else:
            data_path = Path(__file__).parent.parent / "data" / "sample_data" / args.data

    if not data_path.exists():
        print_error(f"Fichier non trouvé: {args.data}")
        return 1

    if not args.quiet:
        print_header("Optimisation Bayésienne (Optuna)")
        print(f"  Stratégie: {strategy_name}")
        print(f"  Données: {data_path}")
        print(f"  Trials: {args.n_trials}")
        print(f"  Métrique: {args.metric}")
        print(f"  Sampler: {args.sampler}")
        if args.pruning:
            print(f"  Pruning: activé ({args.pruner})")
        if args.multi_objective:
            print("  Mode: Multi-objectif (Pareto)")
        print()

    # Charger données
    from data.loader import _normalize_ohlcv, _read_file
    df = _read_file(data_path)
    df = _normalize_ohlcv(df)
    try:
        df = _apply_date_filter(df, args.start, args.end)
    except ValueError as e:
        print_error(str(e))
        return 1

    if not args.quiet:
        print_success(f"Données chargées: {len(df)} barres")

    # Construire le param_space
    strat = strat_class()

    if args.param_space:
        # Param space personnalisé via JSON
        try:
            param_space = json_module.loads(args.param_space)
        except json_module.JSONDecodeError as e:
            print_error(f"Erreur JSON param_space: {e}")
            return 1
    else:
        # Param space automatique depuis la stratégie
        param_space = suggest_param_space(strategy_name)

        # Enrichir avec les param_ranges de la stratégie si non couvert
        for name, (min_v, max_v) in strat.param_ranges.items():
            if name not in param_space:

                default = strat.default_params.get(name, (min_v + max_v) / 2)
                param_type = "int" if isinstance(default, int) else "float"
                param_space[name] = {
                    "type": param_type,
                    "low": min_v,
                    "high": max_v,
                }

    if not param_space:
        print_error(f"Aucun param_space défini pour {strategy_name}")
        return 1

    # Contraintes
    constraints = []
    if args.constraints:
        for c in args.constraints:
            parts = c.split(",")
            if len(parts) == 3:
                constraints.append((parts[0], parts[1], parts[2]))

    # Extraire symbol et timeframe
    stem = data_path.stem
    parts = stem.split("_")
    symbol = args.symbol or (parts[0] if parts else "UNKNOWN")
    timeframe = args.timeframe or (parts[1] if len(parts) > 1 else "1h")

    # Créer l'optimiseur
    from utils.config import Config
    config_kwargs = {"fees_bps": args.fees_bps}
    if args.slippage_bps is not None:
        config_kwargs["slippage_bps"] = args.slippage_bps
    config = Config(**config_kwargs)

    optimizer = OptunaOptimizer(
        strategy_name=strategy_name,
        data=df,
        param_space=param_space,
        constraints=constraints,
        initial_capital=args.capital,
        early_stop_patience=args.early_stop_patience,
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        seed=args.seed,
    )

    if not args.quiet:
        print_info(f"Lancement optimisation ({args.n_trials} trials)...")
        if args.early_stop_patience:
            print_info(f"Early stopping activé: patience={args.early_stop_patience}")
        print()

    # Lancer l'optimisation
    try:
        if args.multi_objective:
            # Multi-objectif (Pareto)
            metrics = [normalize_metric_name(m.strip()) for m in args.metric.split(",")]
            directions = []
            for m in metrics:
                if m in ["max_drawdown"]:
                    directions.append("minimize")
                else:
                    directions.append("maximize")

            result = optimizer.optimize_multi_objective(
                n_trials=args.n_trials,
                metrics=metrics,
                directions=directions,
                timeout=args.timeout,
            )

            if not args.quiet:
                print_header("Résultats Multi-Objectif (Front de Pareto)")
                print(f"  Solutions Pareto: {len(result.pareto_front)}")
                print()

                for i, sol in enumerate(result.pareto_front[:args.top]):
                    print(f"  {Colors.BOLD}Solution #{i+1}{Colors.RESET}")
                    print(f"    Paramètres: {sol['params']}")
                    print(f"    Valeurs: {sol['values']}")
                    print()
        else:
            # Mono-objectif (normaliser la métrique)
            metric_key = normalize_metric_name(args.metric)
            direction = "minimize" if args.metric == "max_drawdown" else "maximize"

            result = optimizer.optimize(
                n_trials=args.n_trials,
                metric=metric_key,
                direction=direction,
                sampler=args.sampler,
                pruner=args.pruner if args.pruning else "none",
                timeout=args.timeout,
                show_progress=not args.quiet,
            )

            if not args.quiet:
                print_header(f"Résultats Optuna (Top {args.top})")
                print(f"  Trials: {result.n_completed}/{args.n_trials} complétés")
                print(f"  Pruned: {result.n_pruned}")
                print(f"  Temps total: {result.total_time:.1f}s")
                print()

                print(f"  {Colors.GREEN}Meilleur résultat:{Colors.RESET}")
                print(f"    Paramètres: {result.best_params}")
                print(f"    {args.metric}: {result.best_value:.4f}")
                if result.best_metrics:
                    print(f"    Sharpe: {result.best_metrics.get('sharpe_ratio', 'N/A'):.3f}")
                    print(f"    Return: {result.best_metrics.get('total_return_pct', 0):+.2f}%")
                    print(f"    Drawdown: {result.best_metrics.get('max_drawdown', 0)*100:.2f}%")
                print()

                # Top N
                top_df = result.get_top_n(args.top)
                if not top_df.empty and len(top_df) > 0:
                    print(f"  {Colors.BOLD}Top {min(args.top, len(top_df))} configurations:{Colors.RESET}")
                    # Récupérer les noms des paramètres (excluant 'trial' et 'value')
                    param_cols = [c for c in top_df.columns if c not in ['trial', 'value']]
                    for idx, (_, row) in enumerate(top_df.head(args.top).iterrows()):
                        params = {col: row[col] for col in param_cols if col in row}
                        val = row.get('value', float('nan'))
                        val_str = f"{val:.4f}" if np.isfinite(val) else "N/A"
                        print(f"    #{idx+1}: {params} → {val_str}")

    except Exception as e:
        print_error(f"Erreur lors de l'optimisation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Export
    if args.output:
        output_path = Path(args.output)

        if args.multi_objective:
            export_data = {
                "strategy": strategy_name,
                "type": "multi_objective",
                "metrics": metrics,
                "n_trials": args.n_trials,
                "pareto_front": result.pareto_front,
            }
        else:
            export_data = {
                "strategy": strategy_name,
                "type": "single_objective",
                "metric": args.metric,
                "n_trials": args.n_trials,
                "n_completed": result.n_completed,
                "n_pruned": result.n_pruned,
                "total_time": result.total_time,
                "best_params": result.best_params,
                "best_value": result.best_value,
                "best_metrics": result.best_metrics,
                "history": result.history,
            }

        with open(output_path, "w") as f:
            json_module.dump(export_data, f, indent=2, default=str)

        if not args.quiet:
            print_success(f"Résultats exportés: {output_path}")

    return 0


# =============================================================================
# COMMANDE: VISUALIZE
# =============================================================================

def cmd_visualize(args) -> int:
    """Visualise les résultats d'un backtest avec graphiques interactifs."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from pathlib import Path

    try:
        from utils.visualization import (
            PLOTLY_AVAILABLE,
            visualize_backtest,
        )
    except ImportError as e:
        print_error(f"Module visualization non disponible: {e}")
        return 1

    if not PLOTLY_AVAILABLE:
        print_error("Plotly requis pour la visualisation: pip install plotly")
        return 1

    # Charger les résultats
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Fichier non trouvé: {input_path}")
        return 1

    if not args.quiet:
        print_header("Visualisation des Résultats")
        print_info(f"Fichier: {input_path}")

    # Charger les données OHLCV si fournies
    df = None
    data_path = None

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            # Chercher dans BACKTEST_DATA_DIR
            import os
            data_dir = os.environ.get("BACKTEST_DATA_DIR", "data/sample_data")
            alt_path = Path(data_dir) / args.data
            if alt_path.exists():
                data_path = alt_path
            else:
                print_warning(f"Fichier de données non trouvé: {args.data}")
                data_path = None

    # Output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif args.html:
        output_path = input_path.with_suffix('.html')

    try:
        # Charger le fichier de résultats
        with open(input_path) as f:
            results_data = json_module.load(f)

        # Déterminer le type de résultat
        result_type = results_data.get('type', 'backtest')

        if result_type == 'sweep':
            # Résultats de sweep - prendre le meilleur
            all_results = results_data.get('results', [])
            if not all_results:
                print_error("Aucun résultat dans le sweep")
                return 1

            # Trier par métrique (sharpe par défaut)
            metric_key = args.metric or 'sharpe_ratio'
            sorted_results = sorted(
                all_results,
                key=lambda x: x.get('metrics', {}).get(metric_key, float('-inf')),
                reverse=True
            )
            best = sorted_results[0]

            trades = best.get('trades', [])
            metrics = best.get('metrics', {})
            params = best.get('params', {})
            equity_curve = best.get('equity_curve')
            strategy = results_data.get('strategy', 'Unknown')

            if not args.quiet:
                print_info(f"Meilleur résultat (sur {len(all_results)}): {params}")
                print_info(f"{metric_key}: {metrics.get(metric_key, 'N/A'):.4f}")

        elif result_type in ('single_objective', 'multi_objective'):
            # Résultats Optuna
            trades = results_data.get('trades', [])
            metrics = results_data.get('best_metrics', {})
            params = results_data.get('best_params', {})
            equity_curve = results_data.get('equity_curve')
            strategy = results_data.get('strategy', 'Unknown')

            if not trades:
                print_warning("Pas de trades dans les résultats Optuna")
                print_info(f"Meilleurs params: {params}")
                print_info(f"Meilleure valeur: {results_data.get('best_value', 'N/A')}")

                # Relancer un backtest avec les meilleurs params pour avoir les trades
                if args.data and data_path:
                    if not args.quiet:
                        print_info("Exécution du backtest avec les meilleurs paramètres...")

                    from backtest import BacktestEngine
                    from data.loader import _normalize_ohlcv, _read_file
                    from utils.config import Config

                    df = _read_file(data_path)
                    df = _normalize_ohlcv(df)

                    config = Config(fees_bps=args.fees_bps)
                    engine = BacktestEngine(
                        initial_capital=args.capital or 10000,
                        config=config,
                    )

                    stem = data_path.stem
                    parts = stem.split("_")
                    symbol = parts[0] if parts else "UNKNOWN"
                    timeframe = parts[1] if len(parts) > 1 else "1m"

                    result = engine.run(
                        df=df,
                        strategy=strategy,
                        params=params,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    if isinstance(result.trades, pd.DataFrame):
                        trades = result.trades.to_dict("records")
                    else:
                        trades = (
                            [t.__dict__ for t in result.trades]
                            if hasattr(result, "trades")
                            else []
                        )
                    if hasattr(result.metrics, "to_dict"):
                        metrics = result.metrics.to_dict()
                    elif isinstance(result.metrics, dict):
                        metrics = result.metrics
                    else:
                        metrics = vars(result.metrics)
                    equity_curve = result.equity.tolist() if hasattr(result, "equity") else None

        else:
            # Backtest simple
            trades = results_data.get('trades', [])
            metrics = results_data.get('metrics', {})
            params = results_data.get('params', {})
            equity_curve = results_data.get('equity_curve')
            strategy = results_data.get('strategy', 'Unknown')

        if not trades and not equity_curve:
            print_error("Aucun trade ou equity curve à visualiser")
            print_info("Assurez-vous que le fichier contient des données de trades")
            return 1

        # Charger les données OHLCV
        if data_path and data_path.exists():
            # Charger directement depuis le fichier
            if data_path.suffix == '.parquet':
                df = pd.read_parquet(data_path)
            elif data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix == '.json':
                df = pd.read_json(data_path)
            else:
                print_warning(f"Format non supporté: {data_path.suffix}")
                df = None

            if df is not None:
                # Normaliser les colonnes
                df.columns = df.columns.str.lower()
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Supprimer timezone pour éviter les problèmes de comparaison
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                if not args.quiet:
                    print_info(f"Données OHLCV chargées: {len(df)} barres")

        # Préparer le titre
        title = f"Backtest - {strategy}"
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
            title += f"\n({params_str})"

        # Visualisation
        if df is not None:
            visualize_backtest(
                df=df,
                trades=trades,
                metrics=metrics,
                equity_curve=equity_curve,
                title=title,
                output_path=output_path,
                show=not args.no_show,
            )
        else:
            # Pas de données OHLCV - afficher seulement equity curve
            if equity_curve:
                from utils.visualization import plot_equity_curve
                fig = plot_equity_curve(
                    equity_curve,
                    initial_capital=metrics.get('initial_capital', 10000),
                    title=title,
                )
                if not args.no_show:
                    fig.show()

                if output_path:
                    fig.write_html(str(output_path))
                    print_success(f"Graphique sauvegardé: {output_path}")
            else:
                print_error("Pas de données suffisantes pour visualiser")
                return 1

        # Stats finales
        if not args.quiet:
            print()
            print_header("Métriques Clés", "-")
            pnl = metrics.get('pnl', metrics.get('total_pnl', 0))
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0)
            win_rate = metrics.get('win_rate', 0)

            pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
            print(f"  PnL:          {pnl_color}{pnl:+,.2f}{Colors.RESET}")
            print(f"  Sharpe:       {sharpe:.2f}")
            print(f"  Max DD:       {Colors.RED}{max_dd:.1f}%{Colors.RESET}")
            print(f"  Win Rate:     {win_rate*100:.1f}%")
            print(f"  Trades:       {len(trades)}")

        if output_path:
            print_success(f"Rapport HTML: {output_path}")

        return 0

    except Exception as e:
        print_error(f"Erreur lors de la visualisation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# COMMANDE: CHECK-GPU
# =============================================================================

def cmd_check_gpu(args) -> int:
    """Diagnostic GPU : CuPy, CUDA, GPUs disponibles et benchmark."""
    if args.no_color:
        Colors.disable()

    import time

    print_header("Diagnostic GPU", "=")

    # 1. CuPy detection
    try:
        import cupy as cp
        cupy_version = cp.__version__
        print_success(f"CuPy installé: version {cupy_version}")
    except ImportError:
        print_error("CuPy non installé")
        print_info("  Installation: pip install cupy-cuda12x")
        return 1
    except Exception as e:
        print_error(f"Erreur import CuPy: {e}")
        return 1

    # 2. CUDA version
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        print_success(f"CUDA Runtime: {cuda_major}.{cuda_minor}")
    except Exception as e:
        print_warning(f"Impossible de récupérer version CUDA: {e}")

    # 3. GPUs détectés
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print_success(f"GPU(s) détecté(s): {device_count}")
        print()

        if device_count == 0:
            print_warning("Aucun GPU détecté")
            return 1

        # Détails de chaque GPU
        print_header("Détails des GPUs", "-")
        for device_id in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(device_id)

            # Nom du GPU
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()

            # Mémoire
            with cp.cuda.Device(device_id):
                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem = mem_info[0]
                total_mem = mem_info[1]
                used_mem = total_mem - free_mem

            # Compute capability
            compute_cap = f"{props['major']}.{props['minor']}"

            # Affichage
            print(f"\n  {Colors.BOLD}GPU {device_id}: {name}{Colors.RESET}")
            print(f"    Compute Capability:  {compute_cap}")
            print(f"    VRAM Totale:         {format_bytes(total_mem)}")
            print(f"    VRAM Libre:          {format_bytes(free_mem)} ({100*free_mem/total_mem:.1f}%)")
            print(f"    VRAM Utilisée:       {format_bytes(used_mem)} ({100*used_mem/total_mem:.1f}%)")

            if not args.quiet:
                print(f"    Multiprocesseurs:    {props.get('multiProcessorCount', 'N/A')}")
                print(f"    Max Threads/Block:   {props.get('maxThreadsPerBlock', 'N/A')}")
                print(f"    Warp Size:           {props.get('warpSize', 'N/A')}")

    except Exception as e:
        print_error(f"Erreur détection GPU: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # 4. Benchmark CPU vs GPU (si --benchmark)
    if args.benchmark:
        print()
        print_header("Benchmark CPU vs GPU (EMA 10k points)", "-")

        try:
            # Données de test
            n_samples = 10000
            np.random.seed(42)
            prices_np = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

            # Fonction EMA simple (pour test)
            def ema_cpu(prices, period=20):
                """EMA sur CPU (NumPy)."""
                alpha = 2.0 / (period + 1)
                ema = np.zeros(len(prices))
                ema[0] = prices[0]
                for i in range(1, len(prices)):
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                return ema

            def ema_gpu(prices, period=20):
                """EMA sur GPU (CuPy)."""
                alpha = 2.0 / (period + 1)
                prices_gpu = cp.asarray(prices)
                ema = cp.zeros(len(prices_gpu))
                ema[0] = prices_gpu[0]
                for i in range(1, len(prices_gpu)):
                    ema[i] = alpha * prices_gpu[i] + (1 - alpha) * ema[i-1]
                cp.cuda.Device().synchronize()
                return cp.asnumpy(ema)

            # Benchmark CPU
            n_runs = 5
            cpu_times = []
            for _ in range(n_runs):
                start = time.time()
                _ = ema_cpu(prices_np, period=20)
                cpu_times.append(time.time() - start)

            cpu_avg = np.mean(cpu_times) * 1000  # en ms

            # Benchmark GPU (avec warmup)
            _ = ema_gpu(prices_np[:100], period=20)  # warmup

            gpu_times = []
            for _ in range(n_runs):
                start = time.time()
                _ = ema_gpu(prices_np, period=20)
                gpu_times.append(time.time() - start)

            gpu_avg = np.mean(gpu_times) * 1000  # en ms

            # Speedup
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

            # Affichage
            print(f"\n  {Colors.BOLD}Résultats:{Colors.RESET}")
            print(f"    Dataset:        {n_samples:,} points")
            print(f"    Runs:           {n_runs}")
            print(f"    CPU (NumPy):    {cpu_avg:.2f} ms")
            print(f"    GPU (CuPy):     {gpu_avg:.2f} ms")

            if speedup > 1:
                print(f"    Speedup:        {Colors.GREEN}{speedup:.2f}x{Colors.RESET}")
            elif speedup < 1 and speedup > 0:
                print(f"    Speedup:        {Colors.YELLOW}{speedup:.2f}x{Colors.RESET} (GPU plus lent)")
            else:
                print(f"    Speedup:        {Colors.RED}N/A{Colors.RESET}")

            print()
            if speedup > 1:
                print_success("GPU est plus rapide que CPU !")
            elif speedup < 1 and speedup > 0.5:
                print_warning("GPU légèrement plus lent (overhead transfert)")
            elif speedup < 0.5:
                print_warning("GPU significativement plus lent (dataset trop petit ?)")

        except Exception as e:
            print_error(f"Erreur benchmark: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # 5. Recommandations
    if not args.quiet:
        print()
        print_header("Recommandations", "-")
        print("  • Utiliser GPU pour datasets > 5000 points")
        print("  • Activer GPU dans indicateurs: voir RAPPORT_ANALYSE_GPU_CPU.md")
        print("  • Variable d'environnement: BACKTEST_GPU_ID=0 (forcer GPU 0)")
        print("  • Variable d'environnement: CUDA_VISIBLE_DEVICES=0 (limiter à GPU 0)")

    print()
    print_success("Diagnostic GPU terminé")
    return 0


# =============================================================================
# COMMANDE: LLM-OPTIMIZE
# =============================================================================

def cmd_llm_optimize(args) -> int:
    """Lance une optimisation LLM avec orchestrateur multi-agents."""
    if args.no_color:
        Colors.disable()

    from pathlib import Path

    from agents.integration import create_orchestrator_with_backtest
    from agents.llm_client import LLMConfig, LLMProvider
    from data.loader import load_ohlcv
    from strategies import get_strategy

    if not args.quiet:
        print_header("Optimisation LLM Multi-Agents")
        print(f"  Stratégie: {args.strategy}")
        print(f"  Symbole: {args.symbol}")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Période: {args.start} → {args.end}")
        print(f"  Capital: {args.capital:,.0f}")
        print(f"  Modèle LLM: {args.model}")
        print(f"  Max itérations: {args.max_iterations}")
        print()

    # Charger les données
    if not args.quiet:
        print_info("Chargement des données...")

    try:
        df = load_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end
        )
    except Exception as e:
        print_error(f"Erreur chargement données: {e}")
        return 1

    if not args.quiet:
        print_success(f"Données chargées: {len(df)} barres")
        print(f"   Période réelle: {df.index[0]} → {df.index[-1]}")
        print()

    # Configuration LLM
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )

    # Récupérer les paramètres par défaut de la stratégie
    if not args.quiet:
        print_info("Récupération des paramètres par défaut...")

    strategy_class = get_strategy(args.strategy)
    if not strategy_class:
        print_error(f"Stratégie '{args.strategy}' non trouvée")
        return 1

    strategy_instance = strategy_class()
    initial_params = strategy_instance.default_params

    if not args.quiet:
        print_success(f"Paramètres initiaux: {list(initial_params.keys())}")
        print()

    # Créer l'orchestrateur
    if not args.quiet:
        print_info("Création de l'orchestrateur multi-agents...")

    try:
        orchestrator = create_orchestrator_with_backtest(
            strategy_name=args.strategy,
            data=df,
            data_symbol=args.symbol,
            data_timeframe=args.timeframe,
            initial_params=initial_params,
            llm_config=llm_config,
            initial_capital=args.capital,
            max_iterations=args.max_iterations,
            min_sharpe=args.min_sharpe,
            max_drawdown_limit=args.max_drawdown,
        )
    except Exception as e:
        print_error(f"Erreur création orchestrateur: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if not args.quiet:
        print_success("Orchestrateur créé")
        print()
        print_header("Lancement de l'optimisation", "-")

    # Lancer l'optimisation
    try:
        result = orchestrator.run()
    except Exception as e:
        print_error(f"Erreur durant l'optimisation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Afficher les résultats
    if not args.quiet:
        print()
        print_header("Résultats Finaux")

        if result.decision == "APPROVED":
            print_success(f"Décision: {result.decision}")
        elif result.decision == "REJECTED":
            print_warning(f"Décision: {result.decision}")
        else:
            print_error(f"Décision: {result.decision}")

        if result.final_params:
            print(f"\n{Colors.BOLD}Paramètres finaux:{Colors.RESET}")
            for k, v in result.final_params.items():
                print(f"    {k}: {v}")

        if result.final_metrics:
            print(f"\n{Colors.BOLD}Métriques finales:{Colors.RESET}")
            _print_metrics(result.final_metrics)

        print(f"\n  Itérations: {result.iterations}")

        if result.reason:
            print(f"\n  Raison: {result.reason}")

    # Export si demandé
    if args.output:
        output_path = Path(args.output)

        import json as json_module

        export_data = {
            "strategy": args.strategy,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "period": {"start": args.start, "end": args.end},
            "model": args.model,
            "max_iterations": args.max_iterations,
            "decision": result.decision,
            "iterations": result.iterations,
            "final_params": result.final_params,
            "final_metrics": result.final_metrics,
            "reason": result.reason,
            "history": result.history if hasattr(result, 'history') else None,
        }

        with open(output_path, "w") as f:
            json_module.dump(export_data, f, indent=2, default=str)

        if not args.quiet:
            print()
            print_success(f"Résultats exportés: {output_path}")

    return 0


# =============================================================================
# COMMANDE: GRID-BACKTEST
# =============================================================================

def cmd_grid_backtest(args) -> int:
    """Exécute un backtest en mode grille de paramètres."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from itertools import product
    from pathlib import Path

    from backtest.engine import BacktestEngine
    from data.loader import load_ohlcv
    from strategies import get_strategy
    from utils.config import Config

    if not args.quiet:
        print_header("Backtest Mode Grille")
        print(f"  Stratégie: {args.strategy}")
        print(f"  Symbole: {args.symbol}")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Période: {args.start} → {args.end}")
        print(f"  Capital: {args.capital:,.0f}")
        print()

    # Charger les données
    if not args.quiet:
        print_info("Chargement des données...")

    try:
        df = load_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end
        )
    except Exception as e:
        print_error(f"Erreur chargement données: {e}")
        return 1

    if not args.quiet:
        print_success(f"Données chargées: {len(df)} barres")
        print(f"   Période réelle: {df.index[0]} → {df.index[-1]}")
        print()

    # Parser la grille de paramètres depuis JSON
    if args.param_grid:
        try:
            param_grid = json_module.loads(args.param_grid)
        except json_module.JSONDecodeError as e:
            print_error(f"Erreur parsing param_grid JSON: {e}")
            return 1
    else:
        # Utiliser une grille par défaut basée sur la stratégie
        strategy_class = get_strategy(args.strategy)
        if not strategy_class:
            print_error(f"Stratégie '{args.strategy}' non trouvée")
            return 1

        strategy_instance = strategy_class()

        if getattr(args, "include_optional_params", False):
            strategy_instance._include_optional_params = True

        optional_skipped: List[str] = []
        if hasattr(strategy_instance, "parameter_specs") and strategy_instance.parameter_specs:
            optional_skipped = [
                name
                for name, spec in strategy_instance.parameter_specs.items()
                if not getattr(spec, "optimize", True)
            ]

        if optional_skipped and not getattr(strategy_instance, "_include_optional_params", False):
            if not args.quiet:
                skipped = ", ".join(optional_skipped)
                print_info(
                    f"Paramètres optionnels ignorés: {skipped} (ajoutez --include-optional-params ou BACKTEST_INCLUDE_OPTIONAL_PARAMS=1)"
                )
        param_grid = {}

        # Générer une grille simple depuis param_ranges
        for param_name, (min_val, max_val) in strategy_instance.param_ranges.items():
            default = strategy_instance.default_params.get(param_name, (min_val + max_val) / 2)

            if isinstance(default, int):
                # Grille de 3 valeurs pour les entiers
                step = max(1, (max_val - min_val) // 2)
                param_grid[param_name] = [min_val, min_val + step, max_val]
            else:
                # Grille de 3 valeurs pour les floats
                param_grid[param_name] = [
                    min_val,
                    (min_val + max_val) / 2,
                    max_val
                ]

    # Générer toutes les combinaisons
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    # Limiter le nombre de combinaisons
    if len(all_combinations) > args.max_combinations:
        print_warning(f"Nombre de combinaisons ({len(all_combinations)}) > max ({args.max_combinations})")
        print_info(f"Limitation à {args.max_combinations} combinaisons")
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, args.max_combinations)

    if not args.quiet:
        print_header("Grille de paramètres", "-")
        for param_name, values in param_grid.items():
            print(f"  {param_name}: {values}")
        print(f"\n  Total combinaisons: {len(all_combinations)}")
        print()

    # Créer la configuration avec les frais
    config_kwargs = {"fees_bps": args.fees_bps}
    if args.slippage_bps is not None:
        config_kwargs["slippage_bps"] = args.slippage_bps
    config = Config(**config_kwargs)

    # Exécuter les backtests
    if not args.quiet:
        print_info("Lancement des backtests...")
        print()

    results = []

    for i, param_combination in enumerate(all_combinations):
        params = dict(zip(param_names, param_combination))

        engine = BacktestEngine(
            initial_capital=args.capital,
            config=config,
        )

        try:
            result = engine.run(
                df=df,
                strategy=args.strategy,
                params=params,
                symbol=args.symbol,
                timeframe=args.timeframe
            )

            # Gérer les métriques
            if hasattr(result.metrics, 'to_dict'):
                metrics = result.metrics.to_dict()
            else:
                metrics = dict(result.metrics)

            results.append({
                "params": params,
                "metrics": metrics,
            })
        except Exception as e:
            if args.verbose:
                print_warning(f"Erreur avec {params}: {e}")

        # Progress
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"\r  Progress: {i+1}/{len(all_combinations)} ({100*(i+1)/len(all_combinations):.1f}%)", end="", flush=True)

    if not args.quiet:
        print("\r" + " " * 50 + "\r", end="")
        print_success(f"Backtests terminés: {len(results)} résultats")
        print()

    # Trier par métrique
    metric_key = args.metric
    reverse = args.metric != "max_drawdown"  # Plus bas = mieux pour drawdown

    results.sort(key=lambda x: x["metrics"].get(metric_key, 0), reverse=reverse)

    # Afficher les meilleurs
    if not args.quiet:
        print_header(f"Top {args.top} Résultats (tri par {args.metric})")

        for i, r in enumerate(results[:args.top]):
            print(f"\n  {Colors.BOLD}#{i+1}{Colors.RESET}")
            print(f"    Paramètres: {r['params']}")
            print(f"    Sharpe: {r['metrics'].get('sharpe_ratio', 0):.3f}")
            print(f"    Return: {r['metrics'].get('total_return_pct', 0):+.2f}%")
            print(f"    Drawdown: {r['metrics'].get('max_drawdown', 0):.2f}%")
            print(f"    {args.metric}: {r['metrics'].get(metric_key, 0):.4f}")

    # Export
    if args.output:
        output_path = Path(args.output)

        export_data = {
            "strategy": args.strategy,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "period": {"start": args.start, "end": args.end},
            "param_grid": param_grid,
            "n_combinations": len(all_combinations),
            "metric": args.metric,
            "results": results,
        }

        with open(output_path, "w") as f:
            json_module.dump(export_data, f, indent=2, default=str)

        if not args.quiet:
            print()
            print_success(f"Résultats exportés: {output_path}")

    return 0


# =============================================================================
# COMMANDE: ANALYZE
# =============================================================================

def cmd_analyze(args) -> int:
    """Analyse les résultats de backtests stockés."""
    if args.no_color:
        Colors.disable()

    import json as json_module
    from pathlib import Path

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print_error(f"Répertoire non trouvé: {results_dir}")
        return 1

    print_header("Analyse des Résultats de Backtests")
    print(f"  Répertoire: {results_dir}")
    print()

    # Charger l'index
    index_path = results_dir / "index.json"

    if not index_path.exists():
        print_error(f"Fichier index.json non trouvé dans {results_dir}")
        return 1

    with open(index_path) as f:
        index = json_module.load(f)

    print_info(f"Nombre total de runs: {len(index)}")
    print()

    # Filtrer les runs profitables
    if args.profitable_only:
        filtered = {
            run_id: data for run_id, data in index.items()
            if data['metrics'].get('total_pnl', 0) > 0
        }
    else:
        filtered = index

    print_header(f"Résultats {'profitables' if args.profitable_only else 'tous'} ({len(filtered)})", "-")

    # Trier par métrique
    metric_key = args.sort_by
    reverse = args.sort_by != "max_drawdown_pct"  # Plus bas = mieux pour drawdown

    sorted_runs = sorted(
        filtered.items(),
        key=lambda x: x[1]['metrics'].get(metric_key, float('-inf') if reverse else float('inf')),
        reverse=reverse
    )

    # Afficher top N
    for i, (run_id, data) in enumerate(sorted_runs[:args.top], 1):
        print(f"\n{Colors.BOLD}Run #{i} - {run_id}{Colors.RESET}")
        print(f"  Stratégie: {data['strategy']}")
        print(f"  Période: {data.get('period_start', 'N/A')} → {data.get('period_end', 'N/A')}")
        print(f"  Symbole: {data.get('symbol', 'N/A')} | Timeframe: {data.get('timeframe', 'N/A')}")

        print(f"\n  {Colors.BOLD}Paramètres:{Colors.RESET}")
        for param, value in data['params'].items():
            print(f"    {param}: {value}")

        m = data['metrics']
        print(f"\n  {Colors.BOLD}Métriques:{Colors.RESET}")
        print(f"    PnL: ${m.get('total_pnl', 0):.2f} | Return: {m.get('total_return_pct', 0):.2f}%")
        print(f"    Sharpe: {m.get('sharpe_ratio', 0):.2f} | Sortino: {m.get('sortino_ratio', 0):.2f}")
        print(f"    Win Rate: {m.get('win_rate_pct', 0):.2f}% | Profit Factor: {m.get('profit_factor', 0):.2f}")
        print(f"    Max DD: {m.get('max_drawdown_pct', 0):.2f}% | Trades: {m.get('total_trades', 0)}")

    # Statistiques globales
    if args.stats and len(filtered) > 0:
        print()
        print_header("Statistiques Globales", "-")

        import numpy as np

        sharpe_values = [d['metrics'].get('sharpe_ratio', 0) for d in filtered.values()]
        return_values = [d['metrics'].get('total_return_pct', 0) for d in filtered.values()]
        dd_values = [d['metrics'].get('max_drawdown_pct', 0) for d in filtered.values()]

        print(f"  {Colors.BOLD}Sharpe Ratio:{Colors.RESET}")
        print(f"    Moyenne: {np.mean(sharpe_values):.2f}")
        print(f"    Médiane: {np.median(sharpe_values):.2f}")
        print(f"    Min: {np.min(sharpe_values):.2f} | Max: {np.max(sharpe_values):.2f}")

        print(f"\n  {Colors.BOLD}Return %:{Colors.RESET}")
        print(f"    Moyenne: {np.mean(return_values):.2f}%")
        print(f"    Médiane: {np.median(return_values):.2f}%")
        print(f"    Min: {np.min(return_values):.2f}% | Max: {np.max(return_values):.2f}%")

        print(f"\n  {Colors.BOLD}Max Drawdown %:{Colors.RESET}")
        print(f"    Moyenne: {np.mean(dd_values):.2f}%")
        print(f"    Médiane: {np.median(dd_values):.2f}%")
        print(f"    Min: {np.min(dd_values):.2f}% | Max: {np.max(dd_values):.2f}%")

    # Export si demandé
    if args.output:
        output_path = Path(args.output)

        export_data = {
            "total_runs": len(index),
            "filtered_runs": len(filtered),
            "filter": "profitable_only" if args.profitable_only else "all",
            "sort_by": args.sort_by,
            "top_runs": [
                {"run_id": run_id, **data}
                for run_id, data in sorted_runs[:args.top]
            ],
        }

        with open(output_path, "w") as f:
            json_module.dump(export_data, f, indent=2, default=str)

        print()
        print_success(f"Analyse exportée: {output_path}")

    return 0
