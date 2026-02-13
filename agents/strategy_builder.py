"""
Module-ID: agents.strategy_builder

Purpose: Agent LLM capable de créer et itérer sur des stratégies de trading complètes
         en utilisant exclusivement les indicateurs du registry existant.

Role in pipeline: orchestration / génération de code

Key components: StrategyBuilder, BuilderSession, BuilderIteration

Inputs: Objectif textuel, DataFrame OHLCV, LLMClient/LLMConfig

Outputs: Stratégie générée dans sandbox_strategies/<session_id>/strategy.py,
         résultats de backtest par itération

Dependencies: agents.llm_client, agents.backtest_executor, agents.analyst,
              indicators.registry, strategies.base, backtest.engine, utils.template

Conventions: Code généré validé syntaxiquement avant exécution ; chargement dynamique
             via importlib ; nom de classe standardisé BuilderGeneratedStrategy ;
             isolation complète dans sandbox_strategies/.

Read-if: Ajout fonctionnalité au builder, modification boucle itérative, templates.

Skip-if: Vous utilisez uniquement les stratégies existantes ou l'AutonomousStrategist.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pprint
import random
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import pandas as pd
from agents.llm_client import LLMClient, LLMConfig, LLMMessage, create_llm_client
from backtest.engine import BacktestEngine
from indicators.registry import calculate_indicator, list_indicators
from metrics_types import normalize_metrics
from utils.observability import generate_run_id, get_obs_logger
from utils.template import render_prompt

from agents.thought_stream import ThoughtStream

logger = get_obs_logger(__name__)

# Dossier racine des sandbox
SANDBOX_ROOT = Path(__file__).resolve().parent.parent / "sandbox_strategies"

# Nom de classe standardisé attendu dans le code généré
GENERATED_CLASS_NAME = "BuilderGeneratedStrategy"

# Nombre max d'échecs consécutifs avant arrêt (circuit breaker)
MAX_CONSECUTIVE_FAILURES = 3
# Nombre minimum de lignes pour considérer du code comme non-vide
MIN_CODE_LINES = 10
# Nombre max de tentatives de réalignement quand le LLM répond hors phase
MAX_PHASE_REALIGN_ATTEMPTS = 2
# Nombre mini d'itérations backtestées avant d'autoriser un arrêt LLM "stop"
MIN_SUCCESSFUL_ITERATIONS_BEFORE_STOP = 3
# Nombre mini de trades pour accepter une stratégie en cours d'optimisation
MIN_TRADES_FOR_ACCEPT = 10
MAX_DRAWDOWN_PCT_FOR_ACCEPT = 60.0
MIN_RETURN_PCT_FOR_ACCEPT = 0.0
# Nombre max de fallbacks déterministes avant arrêt de la session
MAX_DETERMINISTIC_FALLBACKS = 4

_DICT_INDICATOR_NAMES = {
    "bollinger",
    "macd",
    "stochastic",
    "adx",
    "supertrend",
    "ichimoku",
    "psar",
    "vortex",
    "stoch_rsi",
    "aroon",
    "donchian",
    "keltner",
    "pivot_points",
    "fibonacci",
    "fibonacci_levels",
}

_DICT_INDICATOR_ALLOWED_KEYS: Dict[str, set[str]] = {
    "bollinger": {"upper", "middle", "lower"},
    "macd": {"macd", "signal", "histogram"},
    "stochastic": {"stoch_k", "stoch_d"},
    "adx": {"adx", "plus_di", "minus_di"},
    "supertrend": {"supertrend", "direction"},
    "ichimoku": {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou", "cloud_position"},
    "psar": {"sar", "trend", "signal"},
    "vortex": {"vi_plus", "vi_minus", "signal", "oscillator"},
    "stoch_rsi": {"k", "d", "signal"},
    "aroon": {"aroon_up", "aroon_down"},
    "donchian": {"upper", "middle", "lower"},
    "keltner": {"middle", "upper", "lower"},
    "pivot_points": {"pivot", "r1", "s1", "r2", "s2", "r3", "s3"},
    # fibonacci_levels expose aussi des clés dynamiques de type level_XXX.
    "fibonacci_levels": {"high", "low"},
}

_INDICATOR_ALIAS_HINTS = {
    "bollinger_upper": "indicators['bollinger']['upper']",
    "bollinger_middle": "indicators['bollinger']['middle']",
    "bollinger_lower": "indicators['bollinger']['lower']",
    "macd_line": "indicators['macd']['macd']",
    "macd_signal": "indicators['macd']['signal']",
    "macd_histogram": "indicators['macd']['histogram']",
}

_PROPOSAL_PLACEHOLDER_VALUES = {
    "",
    "-",
    "—",
    "n/a",
    "na",
    "none",
    "null",
    "brief description",
    "what you expect this change to achieve and why",
    "when to buy",
    "when to sell",
    "when to close",
}

_LOG_PREFIX_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|", re.IGNORECASE)
_PIPE_LOG_PREFIX_RE = re.compile(
    r"^\s*\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|",
    re.IGNORECASE,
)
_TRACEBACK_LINE_RE = re.compile(r'^\s*File\s+"[^"]+",\s*line\s+\d+', re.IGNORECASE)
_WINDOWS_PATH_LINE_RE = re.compile(r"^\s*[A-Za-z]:\\")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuilderIteration:
    """Résultat d'une itération du builder."""

    iteration: int
    hypothesis: str = ""
    code: str = ""
    backtest_result: Optional[Any] = None
    error: Optional[str] = None
    analysis: str = ""
    decision: str = ""  # "continue", "accept", "stop"
    change_type: str = ""  # "logic", "params", "both"
    diagnostic_category: str = ""  # computed by compute_diagnostic()
    diagnostic_detail: Dict[str, Any] = field(default_factory=dict)
    phase_feedback: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BuilderSession:
    """Session complète de construction de stratégie."""

    session_id: str
    objective: str
    session_dir: Path
    available_indicators: List[str] = field(default_factory=list)

    # État
    iterations: List[BuilderIteration] = field(default_factory=list)
    best_iteration: Optional[BuilderIteration] = None
    best_sharpe: float = float("-inf")
    status: str = "running"  # "running", "success", "failed", "max_iterations"

    # Configuration
    max_iterations: int = 10
    target_sharpe: float = 1.0
    start_time: datetime = field(default_factory=datetime.now)

    # Contexte de marché (transmis au LLM)
    symbol: str = "UNKNOWN"
    timeframe: str = "1h"
    n_bars: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    fees_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 10000.0


# ---------------------------------------------------------------------------
# Validation du code généré
# ---------------------------------------------------------------------------

def validate_generated_code(code: str) -> tuple[bool, str]:
    """
    Valide le code Python généré avant écriture/exécution.

    Vérifie :
    1. Syntaxe Python valide (ast.parse)
    2. Présence de la classe BuilderGeneratedStrategy
    3. Présence de generate_signals
    4. Absence d'imports dangereux (os.system, subprocess, eval, exec)

    Returns:
        (is_valid, error_message)
    """
    # 1. Syntaxe
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Erreur de syntaxe ligne {e.lineno}: {e.msg}"

    # 2. Vérifier la classe attendue
    class_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]
    if GENERATED_CLASS_NAME not in class_names:
        return False, (
            f"Classe '{GENERATED_CLASS_NAME}' absente. "
            f"Classes trouvées: {class_names}"
        )

    # 3. Vérifier generate_signals (dans la classe attendue)
    generate_fns = _iter_generate_signals_functions(tree)
    if not generate_fns:
        return False, "Méthode 'generate_signals' absente."

    # 3b. Signature minimale (évite TypeError runtime)
    fn = generate_fns[0]
    if len(fn.args.args) < 4 and fn.args.vararg is None:
        return (
            False,
            "Signature invalide: generate_signals doit accepter "
            "(self, df, indicators, params).",
        )

    # 3c. NameError probable: variables coeur utilisées sans définition
    #     (fréquent quand le LLM renomme l'argument `df` mais garde `df[...]` dans le corps)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != GENERATED_CLASS_NAME:
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arg_names = {a.arg for a in item.args.args}
            arg_names.update(a.arg for a in item.args.kwonlyargs)
            load_names, store_names = _collect_name_load_store_sets(item)
            for core in ("df", "indicators", "params"):
                if core in load_names and core not in arg_names and core not in store_names:
                    return (
                        False,
                        f"NameError probable: `{core}` utilisé dans `{item.name}` "
                        "mais non défini (paramètre manquant ou variable non assignée).",
                    )
        break

    # 4. Imports dangereux
    dangerous_patterns = [
        "os.system", "subprocess", "eval(", "exec(",
        "__import__", "shutil.rmtree", "open(",
    ]
    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in code_lower:
            return False, f"Import/appel dangereux détecté: '{pattern}'"

    # 5. Accès invalide aux indicateurs via df[...] au lieu de indicators[...]
    try:
        known_indicators = {ind.lower() for ind in list_indicators()}
    except Exception:
        known_indicators = set()

    # 5b. Indicateurs inconnus via indicators[...] / indicators.get(...)
    used_indicators = _collect_indicator_names(tree)
    if known_indicators and used_indicators:
        unknown = sorted(
            {
                name for name in used_indicators
                if name.lower() not in known_indicators
            }
        )
        if unknown:
            hints = [
                f"{name} -> {_INDICATOR_ALIAS_HINTS[name.lower()]}"
                for name in unknown
                if name.lower() in _INDICATOR_ALIAS_HINTS
            ]
            hint_suffix = (
                f" Corrections possibles: {', '.join(hints)}."
                if hints
                else ""
            )
            return (
                False,
                "Indicateur(s) inconnu(s) via indicators détecté(s): "
                f"{unknown}. Utiliser uniquement les noms du registre."
                f"{hint_suffix}",
            )

    df_indexed = re.findall(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
    bad_df_cols = sorted(
        {col for col in df_indexed if col.lower() in known_indicators}
    )
    if bad_df_cols:
        return (
            False,
            "Accès indicateur invalide via df[...] détecté: "
            f"{bad_df_cols}. Utiliser indicators['name'].",
        )

    # 6. Mauvais usage de np.nan_to_num sur indicateurs dict (bollinger, macd, ...)
    for ind in _DICT_INDICATOR_NAMES:
        bad_pattern = (
            r"np\.nan_to_num\(\s*indicators\s*\[\s*['\"]"
            + re.escape(ind)
            + r"['\"]\s*\]\s*\)"
        )
        if re.search(bad_pattern, code):
            return (
                False,
                f"Usage invalide: np.nan_to_num(indicators['{ind}']) (dict). "
                "Appliquer np.nan_to_num sur ses sous-clés.",
            )

    # 7. Validation sémantique AST (usage indicateurs/arrays)
    semantics_ok, semantics_err = _validate_indicator_usage_semantics(code)
    if not semantics_ok:
        return False, semantics_err

    return True, ""


def sanitize_objective_text(objective: Any) -> str:
    """Nettoie un objectif utilisateur et retire les contaminations de logs.

    Cas traités:
    - Collage accidentel de logs complets (INFO/WARNING/Traceback)
    - Objectif imbriqué dans une ligne de log `... objective='...' indicators=...`
    - Bruit visuel (lignes de séparation terminal)
    """
    text = str(objective or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    # Nettoyage résidus modèles de raisonnement
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # Si un objectif est imbriqué dans des logs, récupérer la dernière occurrence
    lower = text.lower()
    marker = "objective='"
    last_idx = lower.rfind(marker)
    if last_idx >= 0:
        start = last_idx + len(marker)
        end = lower.find("' indicators=", start)
        if end == -1:
            end = lower.find("'\n", start)
        if end == -1:
            end = lower.find("'", start)
        if end > start:
            embedded = text[start:end].strip()
            if len(embedded) >= 20:
                text = embedded

    cleaned_lines: List[str] = []
    in_traceback_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        lower_line = line.lower()
        if "traceback (most recent call last)" in lower_line:
            in_traceback_block = True
            continue
        if in_traceback_block:
            continue

        if _LOG_PREFIX_RE.match(line):
            continue
        if _PIPE_LOG_PREFIX_RE.match(line):
            continue
        if lower_line.startswith("traceback"):
            continue
        if lower_line.startswith("during handling of the above exception"):
            continue
        if _TRACEBACK_LINE_RE.match(line):
            continue
        if _WINDOWS_PATH_LINE_RE.match(line):
            continue
        if line.startswith("PS "):
            continue
        if line.startswith("❱"):
            continue
        if re.match(r"^\d+\s*$", line):
            continue
        if "streamlitapiexception" in lower_line:
            continue
        if "site-packages\\streamlit" in lower_line:
            continue
        if lower_line.startswith("files\\python"):
            continue
        if re.match(r"^[═━\-]{10,}$", line):
            continue
        if re.match(r"^\^+$", line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip("`'\" \n\t")
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000].rstrip()
    return cleaned


def _looks_like_log_pollution(text: str) -> bool:
    """Heuristique simple pour détecter un collage de logs/traceback."""
    if not text:
        return False
    lower = text.lower()
    if "traceback (most recent call last)" in lower:
        return True
    if "streamlitapiexception" in lower:
        return True
    if re.search(r"^\s*\d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|", text, re.MULTILINE):
        return True
    if re.search(
        r"^\s*\|\s*(debug|info|warning|error|critical)\s*\|",
        text,
        re.MULTILINE | re.IGNORECASE,
    ):
        return True
    return False


def _metric_float(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Lecture float robuste d'une métrique sans écraser les zéros valides."""
    value = metrics.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _is_ruined_metrics(metrics: Dict[str, Any]) -> bool:
    """Détecte une configuration ruinée à partir des métriques de backtest."""
    ret = _metric_float(metrics, "total_return_pct", 0.0)
    max_dd = abs(_metric_float(metrics, "max_drawdown_pct", 0.0))
    account_ruined = bool(metrics.get("account_ruined", False))
    return account_ruined or ret <= -90.0 or max_dd >= 90.0


def _ranking_sharpe(metrics: Dict[str, Any]) -> float:
    """Sharpe de ranking, pénalisé pour éviter de promouvoir des runs invalides."""
    sharpe = _metric_float(metrics, "sharpe_ratio", float("-inf"))
    trades = int(metrics.get("total_trades", 0) or 0)
    if _is_ruined_metrics(metrics):
        return -20.0
    if trades <= 0:
        return min(sharpe, -5.0)
    return sharpe


def _metrics_fingerprint(metrics: Dict[str, Any]) -> str:
    """Retourne un fingerprint stable des métriques clés pour détecter la stagnation."""
    keys = ("total_return_pct", "max_drawdown_pct", "total_trades", "win_rate_pct", "profit_factor")
    parts = []
    for k in keys:
        v = metrics.get(k, 0) or 0
        parts.append(f"{k}={float(v):.4f}")
    return "|".join(parts)


def _is_accept_candidate(
    metrics: Dict[str, Any],
    *,
    target_sharpe: float,
) -> tuple[bool, str]:
    """Vérifie si une itération est suffisamment robuste pour terminer en succès."""
    sharpe = _metric_float(metrics, "sharpe_ratio", 0.0)
    trades = int(metrics.get("total_trades", 0) or 0)
    ret = _metric_float(metrics, "total_return_pct", 0.0)
    max_dd = abs(_metric_float(metrics, "max_drawdown_pct", 0.0))

    if _is_ruined_metrics(metrics):
        return False, "ruined_metrics"
    if trades < MIN_TRADES_FOR_ACCEPT:
        return False, "insufficient_trades"
    if sharpe < target_sharpe:
        return False, "target_sharpe_not_reached"
    if ret <= MIN_RETURN_PCT_FOR_ACCEPT:
        return False, "non_positive_return"
    if max_dd > MAX_DRAWDOWN_PCT_FOR_ACCEPT:
        return False, "drawdown_too_high"
    return True, "ok"


def _const_value(node: ast.AST) -> Any:
    """Extrait une valeur constante AST (str/int/float) si possible."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):  # pragma: no cover - compat py<3.8
        return node.s
    return None


def _indicator_name_from_subscript(node: ast.AST) -> Optional[str]:
    """Retourne le nom d'indicateur pour indicators['name']."""
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "indicators":
        return None
    key = _const_value(node.slice)
    if isinstance(key, str):
        return key
    return None


def _indicator_name_from_get_call(node: ast.AST) -> Optional[str]:
    """Retourne le nom d'indicateur pour indicators.get('name', ...)."""
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "indicators":
        return None
    if not node.args:
        return None
    key = _const_value(node.args[0])
    if isinstance(key, str):
        return key
    return None


def _is_np_nan_to_num_call(node: ast.AST) -> bool:
    """Vérifie si le noeud est un appel np.nan_to_num(...)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "nan_to_num"
    )


def _is_params_get_call(node: ast.AST) -> bool:
    """Vérifie si le noeud est un appel params.get(...)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "params"
        and node.func.attr == "get"
    )


def _is_params_subscript(node: ast.AST) -> bool:
    """Vérifie si le noeud est params['x']."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
    )


def _is_scalar_cast_call(node: ast.AST) -> bool:
    """Vérifie si le noeud est un cast scalaire (float/int/bool)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "int", "bool"}
    )


def _is_numeric_nonbool_constant(node: ast.AST) -> bool:
    """True si le noeud est une constante numérique non-bool."""
    if not isinstance(node, ast.Constant):
        return False
    return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _iter_generate_signals_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    """Extrait les méthodes generate_signals de BuilderGeneratedStrategy."""
    out: List[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == GENERATED_CLASS_NAME:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "generate_signals":
                    out.append(item)
    return out


def _iter_child_nodes_excluding_nested_scopes(node: ast.AST) -> Any:
    """Itère récursivement sur les noeuds en excluant les scopes imbriqués.

    Objectif: analyser les Name Load/Store d'une méthode sans descendre dans
    des `def`/`class` internes (closures), qui ont leurs propres variables.
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(cur))


def _collect_name_load_store_sets(fn: ast.AST) -> tuple[set[str], set[str]]:
    """Collecte les noms utilisés (Load) et assignés (Store/Del) dans un noeud.

    Ne descend pas dans les scopes imbriqués (closures) pour éviter les faux
    positifs sur les variables capturées.
    """
    load: set[str] = set()
    store: set[str] = set()
    for node in _iter_child_nodes_excluding_nested_scopes(fn):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                load.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                store.add(node.id)
    return load, store


def _collect_indicator_names(tree: ast.AST) -> set[str]:
    """Collecte les noms d'indicateurs référencés dans generate_signals."""
    names: set[str] = set()
    for fn in _iter_generate_signals_functions(tree):
        for node in ast.walk(fn):
            sub = _indicator_name_from_subscript(node)
            if sub:
                names.add(sub)
            got = _indicator_name_from_get_call(node)
            if got:
                names.add(got)
    return names


def _dict_indicator_key_is_valid(indicator_name: str, key: Any) -> bool:
    """Valide une sous-clé pour un indicateur dict connu."""
    if not isinstance(key, str):
        return True
    name = indicator_name.lower()
    allowed = _DICT_INDICATOR_ALLOWED_KEYS.get(name)
    if not allowed:
        return True
    if key in allowed:
        return True
    if name in {"fibonacci", "fibonacci_levels"} and key.startswith("level_"):
        return True
    return False


def _dict_indicator_allowed_keys_hint(indicator_name: str) -> str:
    """Construit un hint compact des sous-clés valides."""
    name = indicator_name.lower()
    allowed = sorted(_DICT_INDICATOR_ALLOWED_KEYS.get(name, set()))
    if name in {"fibonacci", "fibonacci_levels"}:
        allowed = [*allowed, "level_XXX"]
    if not allowed:
        return "sous-clés string attendues"
    return ", ".join(allowed)


def _validate_indicator_usage_semantics(code: str) -> tuple[bool, str]:
    """Validation AST des usages indicateurs pour éviter erreurs runtime récurrentes."""
    try:
        tree = ast.parse(code)
    except Exception:
        return True, ""

    # var_name -> {"kind": "array|dict|values", "indicator": Optional[str]}
    bindings: Dict[str, Dict[str, Any]] = {}

    for fn in _iter_generate_signals_functions(tree):
        # Pass 1: collect bindings
        for node in ast.walk(fn):
            targets: List[ast.Name] = []
            value: Optional[ast.AST] = None

            if isinstance(node, ast.Assign):
                value = node.value
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = node.value
                targets.append(node.target)
            else:
                continue

            if value is None or not targets:
                continue

            ind_name = _indicator_name_from_subscript(value)
            kind: Optional[str] = None
            if ind_name is not None:
                kind = "dict" if ind_name.lower() in _DICT_INDICATOR_NAMES else "array"
            elif _is_np_nan_to_num_call(value) and getattr(value, "args", None):
                arg0 = value.args[0]
                ind_name = _indicator_name_from_subscript(arg0)
                if ind_name is not None:
                    if ind_name.lower() in _DICT_INDICATOR_NAMES:
                        return (
                            False,
                            f"Usage invalide: np.nan_to_num(indicators['{ind_name}']) "
                            "(indicator dict).",
                        )
                    kind = "array"
                elif isinstance(arg0, ast.Name) and arg0.id in bindings:
                    if bindings[arg0.id]["kind"] == "dict":
                        return (
                            False,
                            f"Usage invalide: np.nan_to_num({arg0.id}) alors que "
                            f"{arg0.id} est un indicator dict.",
                        )
                    kind = "array"
            elif isinstance(value, ast.Attribute) and value.attr == "values":
                kind = "values"
            elif _is_params_get_call(value) or _is_params_subscript(value):
                kind = "scalar"
            elif _is_scalar_cast_call(value) and getattr(value, "args", None):
                arg0 = value.args[0]
                if _is_params_get_call(arg0) or _is_params_subscript(arg0):
                    kind = "scalar"
                elif isinstance(arg0, ast.Name):
                    b = bindings.get(arg0.id)
                    if b and b["kind"] == "scalar":
                        kind = "scalar"

            if kind is not None:
                for t in targets:
                    bindings[t.id] = {"kind": kind, "indicator": ind_name}

        # Pass 2: detect invalid usage
        for node in ast.walk(fn):
            # ndarray.shift(...) / ndarray.rolling(...)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if (
                    attr in {"shift", "rolling", "ewm"}
                    and isinstance(node.func.value, ast.Name)
                ):
                    var = node.func.value.id
                    b = bindings.get(var)
                    if b and b["kind"] in {"array", "values"}:
                        return (
                            False,
                            f"Usage invalide: {var}.{attr}(...) sur ndarray. "
                            "Utiliser pandas Series ou logique vectorisée numpy.",
                        )
                if attr in {"shift", "rolling", "ewm"}:
                    ind_name = _indicator_name_from_subscript(node.func.value)
                    if ind_name:
                        return (
                            False,
                            f"Usage invalide: indicators['{ind_name}'].{attr}(...) "
                            "sur ndarray. Utiliser une logique numpy.",
                        )

                # np.nan_to_num(var_dict)
                if _is_np_nan_to_num_call(node) and getattr(node, "args", None):
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Name):
                        b = bindings.get(arg0.id)
                        if b and b["kind"] == "dict":
                            return (
                                False,
                                f"Usage invalide: np.nan_to_num({arg0.id}) alors que "
                                f"{arg0.id} est un indicator dict.",
                            )

            # .iloc/.loc/.iat/.at sur indicateurs ndarray/dict
            if isinstance(node, ast.Attribute) and node.attr in {"iloc", "loc", "iat", "at"}:
                if isinstance(node.value, ast.Name):
                    var = node.value.id
                    b = bindings.get(var)
                    if b and b["kind"] in {"array", "values", "dict"}:
                        return (
                            False,
                            f"Usage invalide: {var}.{node.attr} sur indicateur "
                            "numpy/dict. Utiliser indexation numpy (`arr[i]`).",
                        )
                ind_name = _indicator_name_from_subscript(node.value)
                if ind_name:
                    return (
                        False,
                        f"Usage invalide: indicators['{ind_name}'].{node.attr} "
                        "n'est pas supporté. Utiliser indexation numpy (`arr[i]`).",
                    )

            # Subscript checks: multi-dim on 1D arrays, numeric key on dict indicators
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name):
                    var = node.value.id
                    b = bindings.get(var)
                    if b:
                        key = _const_value(node.slice)
                        if b["kind"] in {"array", "values"} and isinstance(node.slice, ast.Tuple):
                            return (
                                False,
                                f"Usage invalide: indexation multi-dim `{var}[..., ...]` "
                                "sur indicateur 1D.",
                            )
                        if b["kind"] in {"array", "values"} and isinstance(key, str):
                            return (
                                False,
                                f"Usage invalide: clé string `{var}['{key}']` sur "
                                "indicateur ndarray. Utiliser directement l'array.",
                            )
                        if b["kind"] == "dict" and isinstance(key, (int, float)):
                            return (
                                False,
                                f"Usage invalide: clé numérique `{var}[{key}]` sur "
                                "indicator dict; utiliser des sous-clés string.",
                            )
                        if b["kind"] == "dict" and isinstance(key, str):
                            ind = str(b.get("indicator") or "")
                            if ind and not _dict_indicator_key_is_valid(ind, key):
                                hint = _dict_indicator_allowed_keys_hint(ind)
                                return (
                                    False,
                                    f"Usage invalide: `{var}['{key}']` pour "
                                    f"indicateur dict '{ind}'. Sous-clés valides: {hint}.",
                                )

                # indicators['bollinger'][50] / indicators['ema']['ema_21']
                ind_name = _indicator_name_from_subscript(node.value)
                if ind_name:
                    key = _const_value(node.slice)
                    if ind_name.lower() in _DICT_INDICATOR_NAMES:
                        if isinstance(key, (int, float)):
                            return (
                                False,
                                f"Usage invalide: indicators['{ind_name}'][{key}] — "
                                "utiliser des sous-clés string.",
                            )
                        if isinstance(key, str) and not _dict_indicator_key_is_valid(ind_name, key):
                            hint = _dict_indicator_allowed_keys_hint(ind_name)
                            return (
                                False,
                                f"Usage invalide: indicators['{ind_name}']['{key}'] — "
                                f"sous-clé inconnue. Sous-clés valides: {hint}.",
                            )
                    elif isinstance(key, str):
                        return (
                            False,
                            f"Usage invalide: indicators['{ind_name}']['{key}'] — "
                            f"'{ind_name}' retourne un ndarray, pas un dict.",
                        )
                get_name = _indicator_name_from_get_call(node.value)
                if get_name:
                    key = _const_value(node.slice)
                    if get_name.lower() in _DICT_INDICATOR_NAMES:
                        if isinstance(key, (int, float)):
                            return (
                                False,
                                f"Usage invalide: indicators.get('{get_name}')[{key}] — "
                                "utiliser des sous-clés string.",
                            )
                        if isinstance(key, str) and not _dict_indicator_key_is_valid(get_name, key):
                            hint = _dict_indicator_allowed_keys_hint(get_name)
                            return (
                                False,
                                f"Usage invalide: indicators.get('{get_name}')['{key}'] — "
                                f"sous-clé inconnue. Sous-clés valides: {hint}.",
                            )
                    elif isinstance(key, str):
                        return (
                            False,
                            f"Usage invalide: indicators.get('{get_name}')['{key}'] — "
                            f"'{get_name}' retourne un ndarray, pas un dict.",
                        )

            # Comparaisons/arithmétiques directes sur dict indicators
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                for operand in operands:
                    if isinstance(operand, ast.Name):
                        var = operand.id
                        b = bindings.get(var)
                        if b and b["kind"] == "dict":
                            hint_key = _dict_indicator_allowed_keys_hint(
                                str(b.get("indicator") or var)
                            ).split(",")[0].strip()
                            return (
                                False,
                                f"Usage invalide: comparaison `{var} ...` alors que "
                                f"`{var}` est un indicator dict. Utiliser une sous-clé "
                                f"(ex: {var}['{hint_key}']).",
                            )

            if isinstance(node, ast.BinOp):
                for operand in (node.left, node.right):
                    if isinstance(operand, ast.Name):
                        var = operand.id
                        b = bindings.get(var)
                        if b and b["kind"] == "dict":
                            hint_key = _dict_indicator_allowed_keys_hint(
                                str(b.get("indicator") or var)
                            ).split(",")[0].strip()
                            return (
                                False,
                                f"Usage invalide: opération arithmétique sur `{var}` "
                                "qui est un indicator dict. Utiliser une sous-clé "
                                f"(ex: {var}['{hint_key}']).",
                            )

                if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                    for operand in (node.left, node.right):
                        if isinstance(operand, ast.Name):
                            b = bindings.get(operand.id)
                            if b and b["kind"] == "scalar":
                                return (
                                    False,
                                    f"Usage invalide: opérateur logique bitwise avec "
                                    f"scalaire `{operand.id}`. Comparer d'abord la valeur "
                                    "scalaire (ex: `arr > threshold`) puis combiner les masques.",
                                )
                        if _is_numeric_nonbool_constant(operand):
                            return (
                                False,
                                "Usage invalide: opérateur logique bitwise avec constante "
                                "numérique. Utiliser des comparaisons booléennes de part et d'autre.",
                            )

            if isinstance(node, ast.BoolOp):
                for operand in node.values:
                    if isinstance(operand, ast.Name):
                        var = operand.id
                        b = bindings.get(var)
                        if b and b["kind"] == "dict":
                            hint_key = _dict_indicator_allowed_keys_hint(
                                str(b.get("indicator") or var)
                            ).split(",")[0].strip()
                            return (
                                False,
                                f"Usage invalide: test booléen direct sur `{var}` "
                                "qui est un indicator dict. Utiliser une sous-clé "
                                f"(ex: {var}['{hint_key}']).",
                            )

            if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Name):
                var = node.test.id
                b = bindings.get(var)
                if b and b["kind"] == "dict":
                    hint_key = _dict_indicator_allowed_keys_hint(
                        str(b.get("indicator") or var)
                    ).split(",")[0].strip()
                    return (
                        False,
                        f"Usage invalide: condition `{var}` alors que `{var}` est un "
                        f"indicator dict. Utiliser une sous-clé (ex: {var}['{hint_key}']).",
                    )

    return True, ""


def _extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extrait un bloc JSON depuis une réponse LLM (gère ```json ... ```, <think>, etc.)."""
    def _parse_json_dict(payload: str) -> Dict[str, Any]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    # Nettoyer les tags <think> des modèles de raisonnement (qwen3, deepseek-r1, alia, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    if not text:
        logger.warning("extract_json: réponse vide après nettoyage des tags <think>")
        return {}

    # Chercher bloc ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        parsed = _parse_json_dict(match.group(1).strip())
        if parsed:
            return parsed

    # Essayer le texte brut
    parsed = _parse_json_dict(text.strip())
    if parsed:
        return parsed

    # Chercher premier { ... } englobant
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        parsed = _parse_json_dict(brace_match.group(0))
        if parsed:
            return parsed

    logger.warning(
        "extract_json: aucun JSON valide trouvé. Début réponse: %.200s",
        text[:200],
    )
    return {}


def _extract_python_from_response(text: str) -> str:
    """Extrait un bloc Python depuis une réponse LLM."""
    # Nettoyer les tags <think> des modèles de raisonnement
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback : le texte entier
    return text.strip()


def _fix_class_name(code: str) -> str:
    """Renomme la première sous-classe StrategyBase en GENERATED_CLASS_NAME."""
    if re.search(rf"\bclass\s+{GENERATED_CLASS_NAME}\s*\(", code):
        return code
    # Chercher une sous-classe de StrategyBase
    match = re.search(r"class\s+(\w+)\s*\([^)]*StrategyBase[^)]*\)", code)
    if match:
        old_name = match.group(1)
        if old_name != GENERATED_CLASS_NAME:
            code = re.sub(rf"\b{re.escape(old_name)}\b", GENERATED_CLASS_NAME, code)
        return code
    # Pas de StrategyBase — renommer la première classe
    match = re.search(r"class\s+(\w+)\s*\(", code)
    if match:
        old_name = match.group(1)
        code = re.sub(
            rf"class\s+{re.escape(old_name)}\s*\(",
            f"class {GENERATED_CLASS_NAME}(",
            code, count=1,
        )
    return code


def _inject_generate_signals_core_param_aliases(code: str) -> str:
    """Injecte des alias pour éviter des NameError de variables coeur.

    Cas fréquent: `def generate_signals(self, data, indicators, params):` mais le
    corps fait `signals = ... index=df.index` / `close = df["close"]...`.
    On insère alors `df = data` en tête de méthode (idem pour `indicators/params`).
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    fns = _iter_generate_signals_functions(tree)
    if not fns:
        return code

    lines = code.split("\n")
    insertions: List[Tuple[int, List[str]]] = []

    for fn in fns:
        args = [a.arg for a in fn.args.args]
        if len(args) < 4:
            continue

        df_arg, ind_arg, params_arg = args[1], args[2], args[3]
        load_names, store_names = _collect_name_load_store_sets(fn)

        alias_raw: List[str] = []
        if "df" in load_names and "df" not in args and "df" not in store_names and df_arg != "df":
            alias_raw.append(f"df = {df_arg}")
        if (
            "indicators" in load_names
            and "indicators" not in args
            and "indicators" not in store_names
            and ind_arg != "indicators"
        ):
            alias_raw.append(f"indicators = {ind_arg}")
        if "params" in load_names and "params" not in args and "params" not in store_names and params_arg != "params":
            alias_raw.append(f"params = {params_arg}")

        if not alias_raw:
            continue

        insert_lineno: int
        if fn.body:
            first_stmt = fn.body[0]
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(getattr(first_stmt, "value", None), ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                end = getattr(first_stmt, "end_lineno", None) or first_stmt.lineno
                insert_lineno = int(end) + 1
            else:
                insert_lineno = int(first_stmt.lineno)
        else:
            insert_lineno = int((getattr(fn, "end_lineno", None) or fn.lineno) + 1)

        insert_idx = max(0, min(len(lines), insert_lineno - 1))

        # Indentation = indentation de la première ligne de body (ou fallback def+4)
        indent = ""
        if 0 <= insert_idx < len(lines) and lines:
            indent = re.match(r"^(\s*)", lines[insert_idx]).group(1)
        else:
            def_line_idx = max(0, min(len(lines) - 1, int(fn.lineno) - 1)) if lines else 0
            def_indent = re.match(r"^(\s*)", lines[def_line_idx]).group(1) if lines else ""
            indent = def_indent + "    "

        insertions.append((insert_idx, [indent + line for line in alias_raw]))

    if not insertions:
        return code

    # Appliquer en reverse pour préserver les index
    for idx, new_lines in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines[idx:idx] = new_lines

    return "\n".join(lines)


def _repair_code(code: str) -> str:
    """Auto-repair des erreurs courantes du code genere par LLM.

    Corrige:
    - Tags <think> des modeles de raisonnement (qwen3, deepseek-r1)
    - Docstrings triple-quoted non terminées (cause #1 de crash)
    - Nom de classe incorrect (cause #2 de crash)
    - np.nan_to_num() appliqué directement sur indicateurs dict
    - .shift() / .rolling() / .ewm() sur ndarray → remplacement numpy
    - .iloc / .loc sur indicateurs ndarray → indexation numpy
    - indicators['ema']['ema_XX'] → indicators['ema'] (array, pas dict)
    """
    # 1. Retirer les tags <think> des modeles de raisonnement
    code = re.sub(r"<think>.*?</think>\s*", "", code, flags=re.DOTALL)
    code = re.sub(r"<think>.*", "", code, flags=re.DOTALL)

    # 2. Supprimer le preamble non-Python (markdown, texte explicatif)
    #    avant la première ligne de code réelle (from/import/class/def)
    lines = code.split("\n")
    first_code_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (
            stripped.startswith(("from ", "import ", "class ", "def ", "@"))
            or stripped.startswith("#!")
        ):
            first_code_idx = idx
            break
    if first_code_idx > 0:
        code = "\n".join(lines[first_code_idx:])

    # 3. Supprimer docstrings si syntax error + tenter un dedent sur erreurs d'indentation
    try:
        ast.parse(code)
    except SyntaxError as e:
        msg = str(getattr(e, "msg", "") or "").lower()
        if "unexpected indent" in msg or "unindent" in msg or "indentation" in msg:
            code = textwrap.dedent(code)
        code = _strip_docstrings(code)

    # 3. Fixer le nom de classe
    code = _fix_class_name(code)

    # 4. np.nan_to_num(indicators["bollinger"]) → indicateur dict accédé directement
    #    Remplacer par extraction des sous-clés
    for dict_ind in _DICT_INDICATOR_NAMES:
        # np.nan_to_num(indicators["bollinger"]) → indicators["bollinger"]
        code = re.sub(
            r"np\.nan_to_num\(\s*indicators\s*\[\s*['\"]"
            + re.escape(dict_ind)
            + r"['\"]\s*\]\s*\)",
            f'indicators["{dict_ind}"]',
            code,
        )
        # np.nan_to_num(indicators.get("bollinger")) → indicators.get("bollinger")
        code = re.sub(
            r"np\.nan_to_num\(\s*indicators\.get\(\s*['\"]"
            + re.escape(dict_ind)
            + r"['\"]\s*\)\s*\)",
            f'indicators.get("{dict_ind}")',
            code,
        )

    # 4b. Normaliser les sous-clés erronées courantes (stochastic)
    #     Certains modèles utilisent `indicators['stochastic']['signal|stochastic']`.
    code = re.sub(
        r"(indicators\s*\[\s*['\"]stochastic['\"]\s*\]\s*\[\s*['\"])(stochastic|k)(['\"]\s*\])",
        r"\1stoch_k\3",
        code,
        flags=re.IGNORECASE,
    )
    code = re.sub(
        r"(indicators\s*\[\s*['\"]stochastic['\"]\s*\]\s*\[\s*['\"])(signal|d)(['\"]\s*\])",
        r"\1stoch_d\3",
        code,
        flags=re.IGNORECASE,
    )

    # 5. .shift(N) sur ndarray → np.roll(..., N)
    #    pattern: var_name.shift(N)  → np.roll(var_name, N)
    code = re.sub(
        r"(\b\w+)\.shift\(\s*(\d+)\s*\)",
        r"np.roll(\1, \2)",
        code,
    )
    # .shift() sans arg → np.roll(..., 1)
    code = re.sub(
        r"(\b\w+)\.shift\(\s*\)",
        r"np.roll(\1, 1)",
        code,
    )

    # 6. indicators['ema']['ema_XX'] → indicators['ema']
    #    (ema/rsi/atr sont des plain arrays, pas des dicts)
    for arr_ind in ("ema", "rsi", "atr", "sma", "cci", "mfi",
                    "williams_r", "momentum", "obv", "roc"):
        code = re.sub(
            r"indicators\s*\[\s*['\"]"
            + re.escape(arr_ind)
            + r"['\"]\s*\]\s*\[\s*['\"][^'\"]*['\"]\s*\]",
            f'indicators["{arr_ind}"]',
            code,
        )

    # 7. Supprimer les imports incorrects "from indicators import ..."
    #    Le LLM local tente parfois d'importer directement les indicateurs
    #    alors qu'ils sont fournis via le dict `indicators`.
    code = re.sub(
        r"^from\s+indicators\s+import\s+[^\n]+\n?",
        "",
        code,
        flags=re.MULTILINE,
    )
    code = re.sub(
        r"^import\s+indicators\b[^\n]*\n?",
        "",
        code,
        flags=re.MULTILINE,
    )

    # 8. Garantir les imports obligatoires
    _REQUIRED_IMPORTS = [
        ("from typing import", "from typing import Any, Dict, List\n"),
        ("from strategies.base import StrategyBase", "from strategies.base import StrategyBase\n"),
        ("from utils.parameters import ParameterSpec", "from utils.parameters import ParameterSpec\n"),
        ("import numpy", "import numpy as np\n"),
        ("import pandas", "import pandas as pd\n"),
    ]
    for check_str, import_line in _REQUIRED_IMPORTS:
        if check_str not in code:
            code = import_line + code

    # 9. Garantir l'héritage StrategyBase
    #    Si la classe est définie sans parent ou avec un parent incorrect,
    #    forcer l'héritage de StrategyBase.
    code = re.sub(
        rf"class\s+{GENERATED_CLASS_NAME}\s*:\s*\n",
        f"class {GENERATED_CLASS_NAME}(StrategyBase):\n",
        code,
    )
    code = re.sub(
        rf"class\s+{GENERATED_CLASS_NAME}\s*\(\s*\)\s*:",
        f"class {GENERATED_CLASS_NAME}(StrategyBase):",
        code,
    )

    # 10. Alias variables coeur dans generate_signals (évite NameError df/indicators/params)
    code = _inject_generate_signals_core_param_aliases(code)

    return code


def _build_deterministic_fallback_code(
    proposal: Dict[str, Any],
    variant: int = 0,
) -> str:
    """Construit un code de stratégie conservateur, robuste et syntaxiquement valide.

    Utilisé en dernier recours si le LLM renvoie un code invalide même après retry.
    Le paramètre ``variant`` permet de varier la logique quand le fallback est
    appelé plusieurs fois dans la même session (évite la stagnation).

    Variantes:
        0 — mean-reversion RSI/Bollinger + SL/TP ATR (impulsions, overtrading-safe)
        1 — trend-following Supertrend/ADX + SL/TP ATR (impulsions)
        2 — momentum RSI/EMA + SL/TP ATR (impulsions)
    """
    strategy_name = str(proposal.get("strategy_name", "BuilderFallback")).strip()
    if not strategy_name:
        strategy_name = "BuilderFallback"
    strategy_name = strategy_name.replace('"', "").replace("'", "")

    used = proposal.get("used_indicators", [])
    if not isinstance(used, list):
        used = []
    safe_used: List[str] = []
    for x in used:
        if isinstance(x, str):
            sx = x.strip().lower()
            if sx:
                if sx not in safe_used:
                    safe_used.append(sx)
    if len(safe_used) > 20:
        safe_used = safe_used[:20]

    default_params = proposal.get("default_params", {})
    if not isinstance(default_params, dict):
        default_params = {}
    default_params.setdefault("warmup", 50)
    default_params.setdefault("atr_period", 14)
    default_params.setdefault("stop_atr_mult", 1.5)
    default_params.setdefault("tp_atr_mult", 3.0)
    default_params["leverage"] = 1  # Force leverage=1 (not setdefault)

    effective_variant = variant % 3

    if effective_variant == 1:
        # ── Variante 1: trend-following Supertrend/ADX ──
        for needed in ("supertrend", "adx", "atr"):
            if needed not in safe_used:
                safe_used.append(needed)
        default_params.setdefault("supertrend_atr_period", 10)
        default_params.setdefault("supertrend_multiplier", 3.0)
        default_params.setdefault("adx_period", 14)
        default_params.setdefault("adx_threshold", 20.0)
        signals_body = (
            "        stop_atr_mult = float(params.get('stop_atr_mult', 1.5))\n"
            "        tp_atr_mult = float(params.get('tp_atr_mult', 3.0))\n"
            "        adx_threshold = float(params.get('adx_threshold', 20.0))\n"
            "        close = np.nan_to_num(df['close'].values.astype(np.float64))\n"
            "        if len(close) < warmup + 2:\n"
            "            return signals\n"
            "        atr_raw = indicators.get('atr')\n"
            "        if isinstance(atr_raw, np.ndarray):\n"
            "            atr = np.nan_to_num(atr_raw.astype(np.float64))\n"
            "        else:\n"
            "            atr = np.full(n, 0.0)\n"
            "        st_raw = indicators.get('supertrend')\n"
            "        if isinstance(st_raw, dict):\n"
            "            direction = np.nan_to_num(st_raw.get('direction', np.zeros(n))).astype(np.float64)\n"
            "        else:\n"
            "            direction = np.full(n, 0.0)\n"
            "        adx_raw = indicators.get('adx')\n"
            "        if isinstance(adx_raw, dict):\n"
            "            adx = np.nan_to_num(adx_raw.get('adx', np.zeros(n))).astype(np.float64)\n"
            "        else:\n"
            "            adx = np.full(n, 0.0)\n"
            "        df.loc[:, 'bb_stop_long'] = np.nan\n"
            "        df.loc[:, 'bb_tp_long'] = np.nan\n"
            "        df.loc[:, 'bb_stop_short'] = np.nan\n"
            "        df.loc[:, 'bb_tp_short'] = np.nan\n"
            "        bull = direction > 0\n"
            "        bear = direction < 0\n"
            "        bull_prev = np.roll(bull, 1)\n"
            "        bear_prev = np.roll(bear, 1)\n"
            "        bull_prev[:1] = False\n"
            "        bear_prev[:1] = False\n"
            "        long_entry = bull & (~bull_prev) & (adx >= adx_threshold)\n"
            "        short_entry = bear & (~bear_prev) & (adx >= adx_threshold)\n"
            "        long_entry[:warmup] = False\n"
            "        short_entry[:warmup] = False\n"
            "        signals[long_entry] = 1.0\n"
            "        signals[short_entry] = -1.0\n"
            "        df.loc[long_entry, 'bb_stop_long'] = close[long_entry] - stop_atr_mult * atr[long_entry]\n"
            "        df.loc[long_entry, 'bb_tp_long'] = close[long_entry] + tp_atr_mult * atr[long_entry]\n"
            "        df.loc[short_entry, 'bb_stop_short'] = close[short_entry] + stop_atr_mult * atr[short_entry]\n"
            "        df.loc[short_entry, 'bb_tp_short'] = close[short_entry] - tp_atr_mult * atr[short_entry]\n"
        )
    elif effective_variant == 2:
        # ── Variante 2: momentum RSI/EMA ──
        for needed in ("rsi", "ema", "atr"):
            if needed not in safe_used:
                safe_used.append(needed)
        default_params.setdefault("rsi_mid", 50.0)
        default_params.setdefault("ema_period", 50)
        signals_body = (
            "        rsi_mid = float(params.get('rsi_mid', 50.0))\n"
            "        stop_atr_mult = float(params.get('stop_atr_mult', 1.5))\n"
            "        tp_atr_mult = float(params.get('tp_atr_mult', 3.0))\n"
            "        close = np.nan_to_num(df['close'].values.astype(np.float64))\n"
            "        if len(close) < warmup + 2:\n"
            "            return signals\n"
            "        atr_raw = indicators.get('atr')\n"
            "        if isinstance(atr_raw, np.ndarray):\n"
            "            atr = np.nan_to_num(atr_raw.astype(np.float64))\n"
            "        else:\n"
            "            atr = np.full(n, 0.0)\n"
            "        rsi_raw = indicators.get('rsi')\n"
            "        if isinstance(rsi_raw, np.ndarray):\n"
            "            rsi = np.nan_to_num(rsi_raw.astype(np.float64))\n"
            "        else:\n"
            "            rsi = np.full(n, 50.0)\n"
            "        ema_raw = indicators.get('ema')\n"
            "        if isinstance(ema_raw, np.ndarray):\n"
            "            ema = np.nan_to_num(ema_raw.astype(np.float64))\n"
            "        else:\n"
            "            ema = close.copy()\n"
            "        df.loc[:, 'bb_stop_long'] = np.nan\n"
            "        df.loc[:, 'bb_tp_long'] = np.nan\n"
            "        df.loc[:, 'bb_stop_short'] = np.nan\n"
            "        df.loc[:, 'bb_tp_short'] = np.nan\n"
            "        long_cond = (rsi > rsi_mid) & (close > ema)\n"
            "        short_cond = (rsi < rsi_mid) & (close < ema)\n"
            "        long_prev = np.roll(long_cond, 1)\n"
            "        short_prev = np.roll(short_cond, 1)\n"
            "        long_prev[:1] = False\n"
            "        short_prev[:1] = False\n"
            "        long_entry = long_cond & (~long_prev)\n"
            "        short_entry = short_cond & (~short_prev)\n"
            "        long_entry[:warmup] = False\n"
            "        short_entry[:warmup] = False\n"
            "        signals[long_entry] = 1.0\n"
            "        signals[short_entry] = -1.0\n"
            "        df.loc[long_entry, 'bb_stop_long'] = close[long_entry] - stop_atr_mult * atr[long_entry]\n"
            "        df.loc[long_entry, 'bb_tp_long'] = close[long_entry] + tp_atr_mult * atr[long_entry]\n"
            "        df.loc[short_entry, 'bb_stop_short'] = close[short_entry] + stop_atr_mult * atr[short_entry]\n"
            "        df.loc[short_entry, 'bb_tp_short'] = close[short_entry] - tp_atr_mult * atr[short_entry]\n"
        )
    else:
        # ── Variante 0: mean-reversion RSI/Bollinger ──
        for needed in ("rsi", "bollinger", "atr"):
            if needed not in safe_used:
                safe_used.append(needed)
        default_params.setdefault("rsi_oversold", 30)
        default_params.setdefault("rsi_overbought", 70)
        signals_body = (
            "        rsi_oversold = float(params.get('rsi_oversold', 30))\n"
            "        rsi_overbought = float(params.get('rsi_overbought', 70))\n"
            "        stop_atr_mult = float(params.get('stop_atr_mult', 1.5))\n"
            "        tp_atr_mult = float(params.get('tp_atr_mult', 3.0))\n"
            "        close = np.nan_to_num(df['close'].values.astype(np.float64))\n"
            "        if len(close) < warmup + 2:\n"
            "            return signals\n"
            "        atr_raw = indicators.get('atr')\n"
            "        if isinstance(atr_raw, np.ndarray):\n"
            "            atr = np.nan_to_num(atr_raw.astype(np.float64))\n"
            "        else:\n"
            "            atr = np.full(n, 0.0)\n"
            "        rsi_raw = indicators.get('rsi')\n"
            "        bb_raw = indicators.get('bollinger')\n"
            "        has_rsi = isinstance(rsi_raw, np.ndarray)\n"
            "        has_bb = isinstance(bb_raw, dict)\n"
            "        if has_rsi:\n"
            "            rsi = np.nan_to_num(rsi_raw.astype(np.float64))\n"
            "        else:\n"
            "            rsi = np.full(n, 50.0)\n"
            "        if has_bb:\n"
            "            bb_lower = np.nan_to_num(bb_raw.get('lower', np.zeros(n)).astype(np.float64))\n"
            "            bb_upper = np.nan_to_num(bb_raw.get('upper', np.zeros(n)).astype(np.float64))\n"
            "        else:\n"
            "            bb_lower = np.full(n, 0.0)\n"
            "            bb_upper = np.full(n, np.inf)\n"
            "        df.loc[:, 'bb_stop_long'] = np.nan\n"
            "        df.loc[:, 'bb_tp_long'] = np.nan\n"
            "        df.loc[:, 'bb_stop_short'] = np.nan\n"
            "        df.loc[:, 'bb_tp_short'] = np.nan\n"
            "        long_cond = (rsi < rsi_oversold) & (close <= bb_lower)\n"
            "        short_cond = (rsi > rsi_overbought) & (close >= bb_upper)\n"
            "        long_prev = np.roll(long_cond, 1)\n"
            "        short_prev = np.roll(short_cond, 1)\n"
            "        long_prev[:1] = False\n"
            "        short_prev[:1] = False\n"
            "        long_entry = long_cond & (~long_prev)\n"
            "        short_entry = short_cond & (~short_prev)\n"
            "        long_entry[:warmup] = False\n"
            "        short_entry[:warmup] = False\n"
            "        signals[long_entry] = 1.0\n"
            "        signals[short_entry] = -1.0\n"
            "        df.loc[long_entry, 'bb_stop_long'] = close[long_entry] - stop_atr_mult * atr[long_entry]\n"
            "        df.loc[long_entry, 'bb_tp_long'] = close[long_entry] + tp_atr_mult * atr[long_entry]\n"
            "        df.loc[short_entry, 'bb_stop_short'] = close[short_entry] + stop_atr_mult * atr[short_entry]\n"
            "        df.loc[short_entry, 'bb_tp_short'] = close[short_entry] - tp_atr_mult * atr[short_entry]\n"
        )

    # ── Partie commune: assemblage du code final ──
    default_params_literal = _format_python_dict_literal(default_params)
    default_params_lines = default_params_literal.splitlines() or ["{}"]
    if len(default_params_lines) == 1:
        default_params_block = f"        return {default_params_lines[0]}\n\n"
    else:
        default_params_block = "        return " + default_params_lines[0] + "\n"
        default_params_block += "".join(
            f"        {line}\n" for line in default_params_lines[1:]
        )
        default_params_block += "\n"

    return (
        "from typing import Any, Dict, List\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from utils.parameters import ParameterSpec\n"
        "from strategies.base import StrategyBase\n\n\n"
        f"class {GENERATED_CLASS_NAME}(StrategyBase):\n"
        "    def __init__(self):\n"
        f"        super().__init__(name=\"{strategy_name}\")\n\n"
        "    @property\n"
        "    def required_indicators(self) -> List[str]:\n"
        f"        return {safe_used!r}\n\n"
        "    @property\n"
        "    def default_params(self) -> Dict[str, Any]:\n"
        f"{default_params_block}"
        "    @property\n"
        "    def parameter_specs(self) -> Dict[str, ParameterSpec]:\n"
        "        return {}\n\n"
        "    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:\n"
        "        n = len(df)\n"
        "        signals = pd.Series(0.0, index=df.index, dtype=np.float64)\n"
        "        warmup = int(params.get('warmup', 50))\n"
        f"{signals_body}"
        "        signals.iloc[:warmup] = 0.0\n"
        "        return signals\n"
    )


# Prefixes de ligne indiquant du vrai code Python (pas du texte de docstring)
_CODE_LINE_STARTS = (
    "def ", "class ", "@", "return ", "import ", "from ",
    "self.", "super(", "if ", "for ", "while ", "try:", "with ",
    "raise ", "yield ", "assert ", "pass", "break", "continue",
    "signals", "result", "n =", "n=",
)


def _strip_docstrings(code: str) -> str:
    """Supprime tous les blocs triple-quoted, y compris non terminés.

    Utilise une heuristique pour détecter la fin d'un docstring non terminé:
    si une ligne ressemble à du code Python (def, class, @, return, ...),
    on considère que le docstring est terminé et on préserve la ligne.
    """
    lines = code.split("\n")
    result = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        if in_docstring:
            # Fermeture explicite du docstring
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
                continue
            # Heuristique: si la ligne ressemble à du code, le docstring
            # non terminé est considéré comme fini → préserver la ligne
            if stripped.startswith(_CODE_LINE_STARTS):
                in_docstring = False
                result.append(line)
                continue
            # Toujours dans le docstring → ignorer
            continue

        # Détecter l'ouverture d'un triple-quote
        for tq in ['"""', "'''"]:
            if tq in stripped:
                cnt = stripped.count(tq)
                if cnt >= 2:
                    # Docstring fermé sur la même ligne → ignorer la ligne
                    break
                # Docstring multi-ligne ouvert → commencer à ignorer
                in_docstring = True
                break
        else:
            # Pas de triple-quote → conserver la ligne
            result.append(line)

    return "\n".join(result)


def _normalize_proposal_keys(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise les clés JSON d'une proposition LLM (case-insensitive).

    Certains modèles locaux retournent des clés en casse mixte
    (ex: ``used_indiCATORS``, ``default_PARAMS``).  Cette fonction
    mappe chaque clé vers sa version canonique attendue.
    """
    if not proposal:
        return proposal

    _CANONICAL = {
        "strategy_name": "strategy_name",
        "hypothesis": "hypothesis",
        "change_type": "change_type",
        "used_indicators": "used_indicators",
        "indicator_params": "indicator_params",
        "entry_long_logic": "entry_long_logic",
        "entry_short_logic": "entry_short_logic",
        "exit_logic": "exit_logic",
        "risk_management": "risk_management",
        "default_params": "default_params",
        "parameter_specs": "parameter_specs",
    }
    # Build lowercase → canonical mapping
    lower_map = {k.lower(): v for k, v in _CANONICAL.items()}

    normalized: Dict[str, Any] = {}
    for key, value in proposal.items():
        canonical = lower_map.get(key.lower().replace(" ", "_"), key)
        normalized[canonical] = value

    # Normaliser change_type (certains LLM retournent "logic|params|both")
    if "change_type" in normalized:
        normalized["change_type"] = _normalize_change_type(
            normalized.get("change_type", "")
        )

    return normalized


def _is_empty_proposal(proposal: Dict[str, Any]) -> bool:
    """Vérifie si une proposition LLM est vide ou inutilisable."""
    if not proposal:
        return True
    hyp = str(proposal.get("hypothesis", "")).strip()
    inds = proposal.get("used_indicators", [])
    if not hyp or hyp in ("—", "-", "N/A", ""):
        return True
    if not inds:
        return True
    return False


def _normalize_change_type(change_type: Any) -> str:
    """Normalise le type de changement dans {logic, params, both, accept}."""
    raw = str(change_type or "").strip().lower()
    if raw in {"logic", "params", "both", "accept"}:
        return raw
    if "param" in raw:
        return "params"
    if "logic" in raw:
        return "logic"
    if "accept" in raw:
        return "accept"
    return "logic"


def _policy_change_type_override(
    *,
    session: "BuilderSession",
    last_iteration: Optional["BuilderIteration"],
) -> Optional[str]:
    """Force un type de modification cohérent avec le diagnostic récent.

    Objectif: éviter les oscillations `both` quand le problème est clairement
    structurel (ruined/no_trades/etc.).
    """
    if last_iteration is None:
        return None

    cat = str(getattr(last_iteration, "diagnostic_category", "") or "").strip().lower()
    sev = str(
        (getattr(last_iteration, "diagnostic_detail", {}) or {}).get("severity", "")
    ).strip().lower()

    # Pattern oscillant fréquent: ruined <-> no_trades
    recent = [
        str(getattr(it, "diagnostic_category", "") or "").strip().lower()
        for it in (session.iterations[-3:] if session.iterations else [])
        if str(getattr(it, "diagnostic_category", "") or "").strip()
    ]
    if len(recent) >= 2 and set(recent[-2:]).issubset({"ruined", "no_trades"}):
        return "logic"

    logic_cats = {
        "ruined",
        "no_trades",
        "overtrading",
        "wrong_direction",
        "high_drawdown",
        "needs_work",
    }
    param_cats = {"approaching_target", "marginal", "target_reached"}

    if cat in logic_cats:
        return "logic"
    if cat in param_cats and sev in {"info", "success"}:
        return "params"
    return None


def _is_placeholder_text(value: Any) -> bool:
    """Détecte un champ placeholder/générique au lieu d'une vraie consigne."""
    text = str(value or "").strip().lower()
    if text in _PROPOSAL_PLACEHOLDER_VALUES:
        return True
    return (
        "placeholder" in text
        or text.startswith("example")
        or text.startswith("exemple")
        or "to achieve and why" in text
    )


def _proposal_issues(proposal: Dict[str, Any]) -> List[str]:
    """Retourne la liste des raisons rendant une proposition invalide."""
    issues: List[str] = []
    if not proposal:
        issues.append("empty_payload")
        return issues

    hyp = str(proposal.get("hypothesis", "")).strip()
    inds = proposal.get("used_indicators", [])
    if not hyp or hyp in ("—", "-", "N/A", ""):
        issues.append("missing_hypothesis")
    if not isinstance(inds, list) or not inds:
        issues.append("missing_used_indicators")

    critical_fields = (
        "hypothesis",
        "entry_long_logic",
        "entry_short_logic",
        "exit_logic",
        "risk_management",
    )
    for key in critical_fields:
        if _is_placeholder_text(proposal.get(key, "")):
            issues.append(f"placeholder_{key}")

    default_params = proposal.get("default_params")
    if default_params is not None and not isinstance(default_params, dict):
        issues.append("default_params_not_dict")

    ct = _normalize_change_type(proposal.get("change_type", "logic"))
    if ct not in ("logic", "params", "both", "accept"):
        issues.append("invalid_change_type")

    return issues


def _proposal_has_placeholder_fields(proposal: Dict[str, Any]) -> bool:
    """Détecte les placeholders sur les champs critiques d'une proposition."""
    critical_fields = (
        "hypothesis",
        "entry_long_logic",
        "entry_short_logic",
        "exit_logic",
        "risk_management",
    )
    for key in critical_fields:
        if _is_placeholder_text(proposal.get(key, "")):
            return True
    return False


def _is_invalid_proposal(proposal: Dict[str, Any]) -> bool:
    """Validation minimale d'une proposition avant phase code."""
    return bool(_proposal_issues(proposal))


def _is_empty_code(code: str) -> bool:
    """Vérifie si le code généré est vide ou trivial."""
    stripped = code.strip()
    if not stripped:
        return True
    return len(stripped.splitlines()) < MIN_CODE_LINES


def _looks_like_python_code(text: str) -> bool:
    """Heuristique: détecte un contenu ressemblant à du code Python."""
    if not text:
        return False
    lowered = text.lower()
    markers = (
        "```python",
        "class ",
        "def ",
        "import ",
        "from ",
        "return ",
        "np.",
        "pd.",
    )
    return any(m in lowered for m in markers)


def _looks_like_json_object(text: str) -> bool:
    """Heuristique: détecte un contenu ressemblant à un objet JSON."""
    if not text:
        return False
    stripped = text.strip().lower()
    if stripped.startswith("```json"):
        return True
    return stripped.startswith("{") and stripped.endswith("}")


def _looks_like_strategy_code(raw_text: str, code: str) -> bool:
    """Validation heuristique du contenu attendu en phase code."""
    if _is_empty_code(code):
        return False
    if _looks_like_json_object(raw_text) and not _looks_like_python_code(raw_text):
        return False

    lowered = code.lower()
    return "class " in lowered and "generate_signals" in lowered


def _classify_raw_response(text: str) -> str:
    """Retourne la nature d'une réponse brute LLM pour debug de phase."""
    if not text or not text.strip():
        return "empty"
    if _looks_like_json_object(text):
        return "json"
    if _looks_like_python_code(text):
        return "python"
    return "text"


def _extract_required_indicators_signature(code: str) -> tuple[str, ...]:
    """Retourne une signature stable des required_indicators depuis le code."""
    try:
        tree = ast.parse(code)
    except Exception:
        return tuple()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == GENERATED_CLASS_NAME:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "required_indicators":
                    for stmt in item.body:
                        if isinstance(stmt, ast.Return):
                            try:
                                value = ast.literal_eval(stmt.value)
                            except Exception:
                                return tuple()
                            if isinstance(value, (list, tuple)):
                                normalized = [str(v) for v in value]
                                return tuple(sorted(normalized))
    return tuple()


def _extract_generate_signals_signature(code: str) -> str:
    """Retourne une signature AST du corps de generate_signals."""
    try:
        tree = ast.parse(code)
    except Exception:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == GENERATED_CLASS_NAME:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "generate_signals":
                    return ast.dump(
                        ast.Module(body=item.body, type_ignores=[]),
                        include_attributes=False,
                    )
    return ""


def _params_only_contract_respected(previous_code: str, new_code: str) -> tuple[bool, str]:
    """Vérifie qu'une itération params-only n'a pas modifié la logique."""
    prev_inds = _extract_required_indicators_signature(previous_code)
    new_inds = _extract_required_indicators_signature(new_code)
    if prev_inds and new_inds and prev_inds != new_inds:
        return (
            False,
            f"required_indicators modifiés: avant={prev_inds} après={new_inds}",
        )

    prev_sig = _extract_generate_signals_signature(previous_code)
    new_sig = _extract_generate_signals_signature(new_code)
    if prev_sig and new_sig and prev_sig != new_sig:
        return (
            False,
            "generate_signals modifié alors que change_type=params",
        )

    return True, ""


def _format_python_dict_literal(data: Dict[str, Any]) -> str:
    """Formate un dict Python de manière stable pour insertion dans le code."""
    return pprint.pformat(data, width=88, sort_dicts=True, compact=False)


def _rewrite_default_params_from_proposal(
    previous_code: str,
    proposal: Dict[str, Any],
) -> Optional[str]:
    """Réécrit uniquement default_params dans un code existant (mode params-only)."""
    default_params = proposal.get("default_params")
    if not isinstance(default_params, dict) or not default_params:
        return None

    pattern = re.compile(
        r"(?ms)^(\s*)(def\s+default_params\s*\(\s*self\s*\)\s*(?:->\s*[^:\n]+)?\s*:)\n"
        r".*?(?=^\1(?:def\s+|@property)|^\s*class\s+|\Z)"
    )
    match = pattern.search(previous_code)
    if not match:
        return None

    indent = match.group(1)
    def_header = match.group(2)
    body_indent = indent + "    "
    literal = _format_python_dict_literal(default_params)
    literal_lines = literal.splitlines() or ["{}"]

    if len(literal_lines) == 1:
        return_stmt = f"{body_indent}return {literal_lines[0]}\n"
    else:
        return_stmt = f"{body_indent}return {literal_lines[0]}\n"
        return_stmt += "".join(f"{body_indent}{line}\n" for line in literal_lines[1:])

    replacement = f"{indent}{def_header}\n{return_stmt}"

    patched = previous_code[:match.start()] + replacement + previous_code[match.end():]
    return patched


# ---------------------------------------------------------------------------
# Diagnostic déterministe
# ---------------------------------------------------------------------------

def compute_diagnostic(
    metrics: Dict[str, Any],
    iteration_history: List[Dict[str, Any]],
    target_sharpe: float = 1.0,
) -> Dict[str, Any]:
    """
    Diagnostic déterministe basé sur les métriques de backtest et l'historique.

    Classifie le problème principal, grade chaque dimension (profitabilité,
    risque, efficacité, qualité signaux), recommande le type de modification
    et fournit des actions concrètes.

    Le LLM reçoit ce diagnostic pré-calculé et se concentre sur la SOLUTION
    créative plutôt que sur l'identification du problème.
    """
    # --- Extraction sécurisée ---
    n = metrics.get("total_trades", 0) or 0
    sharpe = metrics.get("sharpe_ratio", 0) or 0
    sortino = metrics.get("sortino_ratio", 0) or 0
    calmar = metrics.get("calmar_ratio", 0) or 0
    ret = metrics.get("total_return_pct", 0) or 0
    dd = abs(metrics.get("max_drawdown_pct", 0) or 0)
    wr = metrics.get("win_rate_pct", 0) or 0
    pf = metrics.get("profit_factor", 0) or 0
    exp = metrics.get("expectancy", 0) or 0
    avg_w = metrics.get("avg_win", 0) or 0
    avg_l = abs(metrics.get("avg_loss", 0) or 0)
    vol = metrics.get("volatility_annual", 0) or 0
    _rr = metrics.get("risk_reward_ratio", 0) or 0  # noqa: F841

    # --- Score card A/B/C/D/F ---
    def _g(v, thresholds):
        for grade, thresh in thresholds:
            if v >= thresh:
                return grade
        return "F"

    sc = {
        "profitability": {
            "grade": _g(ret, [("A", 20), ("B", 5), ("C", 0), ("D", -20)]),
            "detail": f"Return {ret:+.1f}%, PF {pf:.2f}, Expectancy {exp:.2f}",
        },
        "risk": {
            "grade": _g(-dd, [("A", -10), ("B", -25), ("C", -40), ("D", -60)]),
            "detail": f"MaxDD {dd:.1f}%, Vol {vol:.1f}%",
        },
        "efficiency": {
            "grade": _g(sharpe, [("A", 1.5), ("B", 1.0), ("C", 0.5), ("D", 0)]),
            "detail": f"Sharpe {sharpe:.3f}, Sortino {sortino:.3f}, Calmar {calmar:.3f}",
        },
        "signal_quality": {
            "grade": _g(wr, [("A", 50), ("B", 40), ("C", 35), ("D", 25)]),
            "detail": f"WR {wr:.1f}%, Trades {n}, AvgW/L {avg_w:.2f}/{avg_l:.2f}",
        },
    }

    # --- Catégorie principale (par gravité décroissante) ---
    if n == 0:
        cat, sev, ct = "no_trades", "critical", "logic"
        summary = "Aucun trade — conditions d'entrée trop restrictives"
        actions = [
            "Relâcher les seuils (RSI 70→65, Bollinger 2.0σ→1.5σ)",
            "Réduire le nombre de conditions AND combinées",
            "Vérifier NaN handling: np.nan_to_num() avant comparaison",
            "S'assurer que les signaux retournent 1.0/-1.0 (pas True/False)",
        ]
        donts = [
            "Ne PAS ajuster les paramètres numériques — problème structurel",
            "Ne PAS ajouter plus de conditions",
        ]
    elif n < 5:
        cat, sev, ct = "insufficient_trades", "warning", "logic"
        summary = f"Seulement {n} trade(s) — statistiquement insignifiant"
        actions = [
            "Relâcher la condition d'entrée la plus restrictive",
            "Vérifier que exit_logic ne ferme pas immédiatement",
            "Utiliser des seuils moins extrêmes (RSI 80→70, ADX 30→20)",
            "Simplifier: 1 indicateur puis ajouter filtres progressivement",
        ]
        donts = ["Ne PAS interpréter Sharpe/PF avec < 5 trades"]
    elif ret < -90 or dd > 90:
        cat, sev, ct = "ruined", "critical", "logic"
        summary = f"Compte ruiné (Return {ret:.0f}%, DD {dd:.0f}%)"
        actions = [
            "URGENT: Réduire leverage à 1-2× max",
            "URGENT: Ajouter stop-loss ATR (1.5-2× ATR)",
            "Vérifier si signaux LONG/SHORT sont inversés",
            "Repartir d'une logique minimale avec SL/TP obligatoires",
        ]
        donts = [
            "Ne PAS garder la même structure+paramètres ajustés",
            "Ne PAS augmenter le leverage",
        ]
    elif n > 300 and wr < 35:
        cat, sev, ct = "overtrading", "warning", "logic"
        summary = f"Suractivité ({n} trades, WR {wr:.0f}%)"
        actions = [
            "Ajouter filtre tendance (ADX > 25 OU direction EMA longue)",
            "Augmenter seuils pour garder les signaux les plus forts",
            "Dédupliquer: pas de signal identique consécutif",
            "Ajouter cooldown minimum entre trades (N barres)",
        ]
        donts = ["Ne PAS juste ajuster numériquement sans filtrer"]
    elif dd > 50:
        cat, sev, ct = "high_drawdown", "warning", "logic"
        summary = f"Drawdown excessif ({dd:.0f}%)"
        actions = [
            "Ajouter/resserrer stop-loss (ATR 1.5× ou % du prix)",
            "Ajouter take-profit (ATR 2-3×)",
            "Réduire leverage si > 2×",
            "Filtre volatilité: ne pas trader si ATR > percentile_80",
        ]
        donts = ["Ne PAS ignorer le drawdown pour maximiser le rendement"]
    elif ret < -20 and n > 20:
        cat, sev, ct = "wrong_direction", "warning", "logic"
        summary = f"Direction probablement inversée (Return {ret:.0f}%, {n} trades)"
        actions = [
            "DIAGNOSTIC: signaux peut-être inversés (1.0=SHORT?)",
            "Tester: inverser tous les signaux (*= -1)",
            "Vérifier conditions LONG = attente de hausse",
            "Revoir exit_logic: positions fermées au mauvais moment?",
        ]
        donts = ["Ne PAS augmenter les params — la direction est le problème"]
    elif pf < 0.8 and n > 20:
        cat, sev, ct = "losing_per_trade", "warning", "both"
        rr_str = f"AvgWin={avg_w:.2f} vs AvgLoss={avg_l:.2f}" if avg_w > 0 else ""
        summary = f"PF faible ({pf:.2f}) — perd par trade. {rr_str}"
        actions = [
            "Améliorer ratio R/R: TP plus loin OU SL plus serré",
            "Ajouter confirmation: 2ème indicateur avant entrée",
            "Filtrer marchés en range (ADX < 20 = ne pas trader)",
            "Optimiser timing: attendre pullback après signal",
        ]
        donts = ["Ne PAS augmenter le volume de trades pour compenser"]
    elif wr < 30 and n > 20 and pf >= 0.8:
        cat, sev, ct = "low_win_rate", "info", "both"
        summary = f"WR bas ({wr:.0f}%) mais PF acceptable ({pf:.2f})"
        actions = [
            "Si PF > 1: stratégie OK malgré WR — affiner paramètres",
            "Sinon: améliorer timing entrée avec confirmation",
            "Filtre tendance pour trader dans la direction dominante",
            "Sorties plus agressives (trailing stop, break-even)",
        ]
        donts = []
    elif 0 < ret < 5 and sharpe < 0.5 and n > 20:
        cat, sev, ct = "marginal", "info", "params"
        summary = f"Rentable mais marginal (Return {ret:.1f}%, Sharpe {sharpe:.3f})"
        actions = [
            "Focus paramètres: ajuster ±20% les périodes indicateurs",
            "Optimiser ratio SL/TP (levier le plus efficace)",
            "La logique produit des résultats positifs — NE PAS la casser",
            "Tester de légers changements de seuils d'entrée",
        ]
        donts = ["Ne PAS restructurer la logique — elle fonctionne"]
    elif sharpe >= target_sharpe:
        cat, sev, ct = "target_reached", "success", "accept"
        robust = n > 20 and dd < 40
        summary = f"Cible atteinte (Sharpe {sharpe:.3f} >= {target_sharpe})"
        if not robust:
            summary += f" — robustifier ({'peu de trades' if n <= 20 else 'DD élevé'})"
        actions = ["Accepter" if robust else "Continuer pour robustifier"]
        donts = []
    elif target_sharpe > 0 and sharpe >= target_sharpe * 0.5:
        cat, sev, ct = "approaching_target", "info", "params"
        pct = sharpe / target_sharpe * 100
        summary = f"En progression ({pct:.0f}% de la cible Sharpe {target_sharpe})"
        actions = [
            "Fine-tuning UNIQUEMENT: ajuster seuils ±10-20%",
            "Optimiser SL ATR mult (tester 1.0 / 1.5 / 2.0 / 2.5)",
            "Optimiser TP ATR mult (tester 2.0 / 3.0 / 4.0)",
            "Ajuster périodes indicateurs (RSI 14→12 ou 14→16)",
        ]
        donts = [
            "Ne PAS changer la logique — elle fonctionne",
            "Ne PAS ajouter d'indicateurs (risque overfitting)",
        ]
    else:
        cat, sev, ct = "needs_work", "info", "both"
        summary = f"Résultats médiocres (Sharpe {sharpe:.3f}, Return {ret:.1f}%)"
        actions = [
            "Essayer une combinaison d'indicateurs différente",
            "Revoir logique d'entrée/sortie",
            "Simplifier: 1-2 indicateurs max avec logique claire",
        ]
        donts = []

    # --- Détection tendance historique ---
    trend, trend_detail = "first", ""

    if iteration_history:
        prev_sharpes = [float(h.get("sharpe", 0) or 0) for h in iteration_history]
        prev_cats = [h.get("diagnostic_category", "") for h in iteration_history]

        if prev_sharpes:
            delta = sharpe - prev_sharpes[-1]
            if delta > 0.05:
                trend, trend_detail = "improving", f"+{delta:.3f} vs précédent"
            elif delta < -0.05:
                trend, trend_detail = "declining", f"{delta:.3f} vs précédent"
            else:
                trend, trend_detail = "stable", f"Δ={delta:+.3f} (stagnant)"

        # Stagnation: même catégorie 3× consécutives
        recent = (prev_cats[-2:] + [cat]) if len(prev_cats) >= 2 else []
        if len(recent) == 3 and len(set(recent)) == 1 and recent[0]:
            trend = "stagnated"
            trend_detail = (
                f"Même problème '{cat}' 3× de suite — changer d'approche"
            )

        # Oscillation: sharpe en zigzag
        if len(prev_sharpes) >= 2:
            ds = [
                prev_sharpes[j + 1] - prev_sharpes[j]
                for j in range(len(prev_sharpes) - 1)
            ]
            ds.append(sharpe - prev_sharpes[-1])
            if len(ds) >= 2 and all(
                (ds[k] > 0) != (ds[k + 1] > 0) for k in range(len(ds) - 1)
            ):
                trend = "oscillating"
                trend_detail = "Zigzag — stabiliser les modifications"

    return {
        "category": cat,
        "severity": sev,
        "change_type": ct,
        "summary": summary,
        "actions": actions,
        "donts": donts,
        "trend": trend,
        "trend_detail": trend_detail,
        "score_card": sc,
    }


# ---------------------------------------------------------------------------
# Strategy Builder
# ---------------------------------------------------------------------------

class StrategyBuilder:
    """
    Agent capable de générer itérativement des stratégies de trading.

    Workflow :
    1. Recevoir un objectif (ex: "Trend-following BTC 30m avec Bollinger + ATR")
    2. Demander au LLM une proposition (indicateurs, logique, paramètres)
    3. Demander au LLM le code Python complet de la stratégie
    4. Valider le code (syntaxe + sécurité)
    5. Charger dynamiquement la stratégie
    6. Lancer un backtest via BacktestExecutor
    7. Analyser les résultats (LLM)
    8. Décider : itérer (modifier la logique) ou accepter

    Les stratégies générées sont isolées dans sandbox_strategies/<session_id>/.

    Example:
        >>> builder = StrategyBuilder(llm_config=LLMConfig.from_env())
        >>> session = builder.run(
        ...     objective="Trend-following BTC 30m avec Bollinger + ATR",
        ...     data=ohlcv_df,
        ...     max_iterations=5,
        ... )
        >>> print(session.best_sharpe)
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ):
        if llm_client is not None:
            self.llm = llm_client
        elif llm_config is not None:
            self.llm = create_llm_client(llm_config)
        else:
            self.llm = create_llm_client(LLMConfig.from_env())

        self.available_indicators = list_indicators()
        self.stream_callback = stream_callback

    # ------------------------------------------------------------------
    # LLM call helper (streaming si callback défini)
    # ------------------------------------------------------------------

    def _chat_llm(
        self,
        messages: List[LLMMessage],
        *,
        phase: str = "",
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Appel LLM avec streaming optionnel.

        Si ``self.stream_callback`` est défini et que le client supporte
        ``chat_stream``, chaque token généré est relayé via
        ``stream_callback(phase, chunk)`` à l'UI.
        """
        if self.stream_callback and hasattr(self.llm, "chat_stream"):
            return self.llm.chat_stream(
                messages,
                on_chunk=lambda c: self.stream_callback(phase, c),
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self.llm.chat(
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_session_id(objective: str) -> str:
        """Génère un identifiant de session unique."""
        normalized = sanitize_objective_text(objective).lower()
        slug = re.sub(r"[^a-z0-9]+", "_", normalized)[:40].strip("_")
        if not slug:
            slug = "builder_session"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{slug}"

    @staticmethod
    def get_session_dir(session_id: str) -> Path:
        """Retourne le chemin du dossier sandbox pour une session."""
        return SANDBOX_ROOT / session_id

    # ------------------------------------------------------------------
    # LLM interactions
    # ------------------------------------------------------------------

    def _ask_proposal(
        self,
        session: BuilderSession,
        last_iteration: Optional[BuilderIteration] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Demande au LLM une proposition de stratégie.

        Returns:
            (proposal, feedback)
        """
        context = {
            "objective": session.objective,
            "available_indicators": self.available_indicators,
            "iteration": len(session.iterations) + 1,
            "max_iterations": session.max_iterations,
            # Contexte de marché
            "symbol": session.symbol,
            "timeframe": session.timeframe,
            "n_bars": session.n_bars,
            "date_range_start": session.date_range_start,
            "date_range_end": session.date_range_end,
            "fees_bps": session.fees_bps,
            "slippage_bps": session.slippage_bps,
            "initial_capital": session.initial_capital,
        }

        if last_iteration and last_iteration.backtest_result:
            metrics = last_iteration.backtest_result.metrics
            context["last_metrics"] = {
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "volatility_annual": metrics.get("volatility_annual", 0),
                "win_rate_pct": metrics.get("win_rate_pct", 0),
                "total_trades": metrics.get("total_trades", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "expectancy": metrics.get("expectancy", 0),
                "avg_win": metrics.get("avg_win", 0),
                "avg_loss": metrics.get("avg_loss", 0),
                "risk_reward_ratio": metrics.get("risk_reward_ratio", 0),
            }
            context["last_code"] = last_iteration.code
            context["last_analysis"] = last_iteration.analysis
            context["best_sharpe"] = session.best_sharpe
            # Diagnostic pré-calculé de la dernière itération
            if last_iteration.diagnostic_detail:
                context["diagnostic"] = last_iteration.diagnostic_detail
            # Stagnation détectée : forcer le LLM à changer radicalement
            stag = (last_iteration.phase_feedback or {}).get("stagnation", {})
            if stag.get("identical_metrics"):
                context["stagnation_warning"] = (
                    "CRITICAL: Previous iteration produced IDENTICAL metrics. "
                    "Your changes had NO effect. You MUST change the fundamental "
                    "approach: use DIFFERENT indicators, DIFFERENT entry logic, "
                    "or DIFFERENT strategy type (e.g. trend-following instead of "
                    "mean-reversion). Do NOT repeat the same logic with minor tweaks."
                )

        if session.iterations:
            context["iteration_history"] = [
                {
                    "iteration": it.iteration,
                    "hypothesis": it.hypothesis,
                    "change_type": it.change_type,
                    "diagnostic_category": it.diagnostic_category,
                    "sharpe": (
                        it.backtest_result.metrics.get("sharpe_ratio", 0)
                        if it.backtest_result else None
                    ),
                    "return_pct": (
                        it.backtest_result.metrics.get("total_return_pct", 0)
                        if it.backtest_result else None
                    ),
                    "trades": (
                        it.backtest_result.metrics.get("total_trades", 0)
                        if it.backtest_result else None
                    ),
                    "error": it.error,
                }
                for it in session.iterations[-5:]
            ]

        prompt = render_prompt("strategy_builder_proposal.jinja2", context)
        base_messages = [
            LLMMessage(role="system", content=self._system_prompt_proposal()),
            LLMMessage(role="user", content=prompt),
        ]

        response = self._chat_llm(
            messages=base_messages,
            phase="proposal",
            json_mode=True,
            max_tokens=4096,
        )
        raw = response.content or ""
        feedback: Dict[str, Any] = {
            "phase": "proposal",
            "initial_kind": _classify_raw_response(raw),
            "realign_attempts": 0,
            "realign_success": False,
            "issues": [],
        }
        proposal = _normalize_proposal_keys(_extract_json_from_response(raw))
        issues = _proposal_issues(proposal)
        feedback["issues"] = issues
        if not issues:
            proposal["change_type"] = _normalize_change_type(
                proposal.get("change_type", "logic")
            )
            feedback["final_kind"] = feedback["initial_kind"]
            feedback["final_valid"] = True
            return proposal, feedback

        # Phase guard: certains modèles répondent du code / texte libre.
        for attempt in range(1, MAX_PHASE_REALIGN_ATTEMPTS + 1):
            if _looks_like_python_code(raw):
                mismatch = "You answered with Python code, but this is PROPOSAL phase."
            elif _looks_like_json_object(raw):
                mismatch = "You answered JSON but with missing/placeholder fields."
            else:
                mismatch = "You answered with text/explanations, not strict strategy JSON."

            correction = (
                "PHASE LOCK: PROPOSAL ONLY.\n"
                f"{mismatch}\n\n"
                "Return EXACTLY one valid JSON object and nothing else.\n"
                "Forbidden in this phase: Python code, markdown, commentary, objective rewrite.\n"
                "All fields must be concrete (no placeholders like 'brief description').\n"
                "Required keys: strategy_name, hypothesis, change_type, "
                "used_indicators, entry_long_logic, entry_short_logic, exit_logic, "
                "risk_management, default_params, parameter_specs.\n"
                "change_type must be one of: logic, params, both, accept."
            )
            response = self._chat_llm(
                messages=[
                    *base_messages,
                    LLMMessage(role="assistant", content=raw[:4000]),
                    LLMMessage(role="user", content=correction),
                ],
                phase=f"proposal_realign_{attempt}",
                json_mode=(attempt == 1),
                max_tokens=4096,
            )
            raw = response.content or ""
            feedback["realign_attempts"] = attempt
            proposal = _normalize_proposal_keys(_extract_json_from_response(raw))
            issues = _proposal_issues(proposal)
            feedback["issues"] = issues
            if not issues:
                proposal["change_type"] = _normalize_change_type(
                    proposal.get("change_type", "logic")
                )
                feedback["realign_success"] = True
                feedback["final_kind"] = _classify_raw_response(raw)
                feedback["final_valid"] = True
                return proposal, feedback

        feedback["final_kind"] = _classify_raw_response(raw)
        feedback["final_valid"] = False
        return proposal, feedback

    def _ask_code(
        self,
        session: BuilderSession,
        proposal: Dict[str, Any],
        last_iteration: Optional[BuilderIteration] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Demande au LLM de générer le code Python complet.

        Returns:
            (code, feedback)
        """
        # Extraire les actions diagnostiques de la dernière itération
        diag_actions: List[str] = []
        diag_donts: List[str] = []
        if last_iteration is not None:
            diag_detail = getattr(last_iteration, "diagnostic_detail", {}) or {}
            diag_actions = diag_detail.get("actions", [])
            diag_donts = diag_detail.get("donts", [])

        context = {
            "objective": session.objective,
            "proposal": proposal,
            "available_indicators": self.available_indicators,
            "class_name": GENERATED_CLASS_NAME,
            # Contexte de marché
            "symbol": session.symbol,
            "timeframe": session.timeframe,
            "n_bars": session.n_bars,
            "fees_bps": session.fees_bps,
            "slippage_bps": session.slippage_bps,
            "initial_capital": session.initial_capital,
            "previous_code": (
                last_iteration.code
                if last_iteration is not None and getattr(last_iteration, "code", "")
                else ""
            ),
            # Diagnostic de l'itération précédente (injecté dans le template)
            "diagnostic_actions": diag_actions,
            "diagnostic_donts": diag_donts,
        }

        prompt = render_prompt("strategy_builder_code.jinja2", context)
        base_messages = [
            LLMMessage(role="system", content=self._system_prompt_code()),
            LLMMessage(role="user", content=prompt),
        ]

        response = self._chat_llm(
            messages=base_messages,
            phase="code",
            max_tokens=4096,
        )
        raw = response.content or ""
        feedback: Dict[str, Any] = {
            "phase": "code",
            "initial_kind": _classify_raw_response(raw),
            "realign_attempts": 0,
            "realign_success": False,
        }
        code = _extract_python_from_response(raw)
        if _looks_like_strategy_code(raw, code):
            feedback["final_kind"] = feedback["initial_kind"]
            feedback["final_valid"] = True
            return code, feedback

        # Phase guard: certains modèles reviennent en mode JSON/proposition.
        for attempt in range(1, MAX_PHASE_REALIGN_ATTEMPTS + 1):
            if _looks_like_json_object(raw):
                mismatch = "You answered JSON/proposal content, but this is CODE phase."
            elif _looks_like_python_code(raw):
                mismatch = (
                    "You answered Python but not a full strategy class. "
                    "Return a complete class."
                )
            else:
                mismatch = "You answered non-code text, not executable Python."

            correction = (
                "PHASE LOCK: CODE ONLY.\n"
                f"{mismatch}\n\n"
                f"Return ONLY executable Python code for class {GENERATED_CLASS_NAME}.\n"
                "Do not output JSON, strategy objective rewrite, or commentary.\n"
                "Must include: class definition, required_indicators, default_params, "
                "generate_signals.\n"
                "No placeholders."
            )
            response = self._chat_llm(
                messages=[
                    *base_messages,
                    LLMMessage(role="assistant", content=raw[:6000]),
                    LLMMessage(role="user", content=correction),
                ],
                phase=f"code_realign_{attempt}",
                max_tokens=4096,
            )
            raw = response.content or ""
            feedback["realign_attempts"] = attempt
            code = _extract_python_from_response(raw)
            if _looks_like_strategy_code(raw, code):
                feedback["realign_success"] = True
                feedback["final_kind"] = _classify_raw_response(raw)
                feedback["final_valid"] = True
                return code, feedback

        feedback["final_kind"] = _classify_raw_response(raw)
        feedback["final_valid"] = False
        return code, feedback

    def _retry_proposal_simple(self, objective: str) -> Dict[str, Any]:
        """Prompt simplifié quand le LLM ne répond pas au template riche.

        Tente d'abord avec json_mode, puis sans (certains modèles locaux
        gèrent mal le format JSON forcé).
        """
        indicators_str = ", ".join(self.available_indicators[:15])
        prompt = (
            f"Design a simple trading strategy for: {objective}\n\n"
            f"Available indicators: {indicators_str}\n\n"
            "Reply ONLY with this JSON:\n"
            "{\n"
            '  "strategy_name": "my_strategy",\n'
            '  "hypothesis": "one concrete sentence explaining why this setup should work",\n'
            '  "change_type": "logic",\n'
            '  "used_indicators": ["rsi", "bollinger"],\n'
            '  "entry_long_logic": "explicit rule with thresholds, e.g. RSI<30 and close<lower band",\n'
            '  "entry_short_logic": "explicit rule with thresholds, e.g. RSI>70 and close>upper band",\n'
            '  "exit_logic": "explicit close rule, e.g. mean reversion to middle band or RSI cross 50",\n'
            '  "risk_management": "ATR stop and ATR take-profit with concrete multipliers",\n'
            '  "default_params": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0},\n'
            '  "parameter_specs": {"rsi_period": {"min": 5, "max": 50, "default": 14, "type": "int"}}\n'
            "}"
        )
        sys_msg = LLMMessage(
            role="system",
            content=(
                "You are a quant trader. "
                "Reply ONLY with valid JSON. No commentary. No thinking. "
                "No placeholders."
            ),
        )
        user_msg = LLMMessage(role="user", content=prompt)

        # Tentative 1 : avec json_mode
        response = self._chat_llm(
            messages=[sys_msg, user_msg],
            phase="retry_proposal",
            json_mode=True,
            max_tokens=4096,
        )
        result = _normalize_proposal_keys(
            _extract_json_from_response(response.content)
        )
        if result:
            return result

        # Tentative 2 : sans json_mode (certains modèles locaux échouent avec format=json)
        logger.warning(
            "retry_proposal: json_mode a échoué, tentative sans json_mode. "
            "Réponse brute (200 premiers chars): %.200s",
            response.content[:200] if response.content else "(vide)",
        )
        response = self._chat_llm(
            messages=[sys_msg, user_msg],
            phase="retry_proposal_nojson",
            json_mode=False,
            max_tokens=4096,
        )
        return _normalize_proposal_keys(
            _extract_json_from_response(response.content)
        )

    def _retry_code_simple(self, proposal: Dict[str, Any]) -> str:
        """Prompt simplifié quand le LLM ne génère pas de code valide."""
        inds = proposal.get("used_indicators", ["rsi", "bollinger"])
        entry_l = proposal.get("entry_long_logic", "RSI < 30")
        entry_s = proposal.get("entry_short_logic", "RSI > 70")
        prompt = (
            f"Generate a COMPLETE Python class called BuilderGeneratedStrategy.\n"
            f"It must inherit from StrategyBase.\n"
            f"Indicators: {inds}\n"
            f"LONG when: {entry_l}\n"
            f"SHORT when: {entry_s}\n\n"
            "IMPORTANT:\n"
            "- indicator values are numpy arrays (or dict of numpy arrays)\n"
            "- never use .iloc/.loc on indicators; use arr[i] or vectorized masks\n"
            "- bollinger must be indicators['bollinger']['upper|middle|lower']\n\n"
            "- adx must be indicators['adx']['adx|plus_di|minus_di'] (not indicators['adx'] directly)\n"
            "- supertrend must be indicators['supertrend']['supertrend|direction'] (no upper/lower)\n"
            "- stochastic must be indicators['stochastic']['stoch_k|stoch_d'] (no 'signal' key)\n"
            "- do not compare dict indicators directly (e.g. NEVER `adx > 25`)\n"
            "- for `&` / `|`, ensure both sides are boolean masks (no float/int scalar in bitwise op)\n\n"
            "- ema/rsi/atr are plain arrays: NEVER use indicators['ema']['ema_21'] style\n\n"
            "- ALWAYS include leverage=1 in default_params\n"
            "- If using ATR-based SL/TP: write df['bb_stop_long/bb_tp_long/bb_stop_short/bb_tp_short'] on entry bars\n\n"
            "EXACT structure:\n\n"
            "```python\n"
            "from typing import Any, Dict, List\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from strategies.base import StrategyBase\n\n"
            "class BuilderGeneratedStrategy(StrategyBase):\n"
            "    def __init__(self):\n"
            '        super().__init__(name="generated")\n\n'
            "    @property\n"
            "    def required_indicators(self):\n"
            f"        return {inds}\n\n"
            "    @property\n"
            "    def default_params(self):\n"
            "        return {\"leverage\": 1, \"warmup\": 50, \"stop_atr_mult\": 1.5, \"tp_atr_mult\": 3.0}\n\n"
            "    def generate_signals(self, df, indicators, params):\n"
            "        signals = pd.Series(0.0, index=df.index)\n"
            "        return signals\n"
            "```\n\n"
            "Generate the COMPLETE code with ACTUAL logic (no placeholders)."
        )
        response = self._chat_llm(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Generate ONLY Python code inside a ```python block. "
                        "No explanation. No commentary."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            phase="retry_code",
        )
        return _extract_python_from_response(response.content)

    def _retry_code_runtime_fix(
        self,
        proposal: Dict[str, Any],
        failing_code: str,
        runtime_error: str,
    ) -> str:
        """Demande une correction ciblée d'un code qui a échoué au runtime backtest."""
        prompt = (
            "The following strategy code failed at runtime during backtest.\n\n"
            f"Runtime error (may include traceback tail):\n{runtime_error}\n\n"
            "Fix ONLY what is necessary to remove the runtime error while keeping "
            "the strategy intent intact.\n"
            "Rules:\n"
            "- Class name must remain BuilderGeneratedStrategy\n"
            "- Keep required_indicators coherent with indicator usage\n"
            "- Keep generate_signals signature EXACTLY: "
            "def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series\n"
            "- Never reference undefined globals like `df`, `indicators`, `params` inside helper methods; "
            "pass what you need as arguments.\n"
            "- Use indicators dict correctly (dict indicators via sub-keys)\n"
            "- Indicator values are numpy arrays; never call .iloc/.loc on indicators\n"
            "- Use bollinger as indicators['bollinger']['upper|middle|lower'] only\n"
            "- Use adx as indicators['adx']['adx|plus_di|minus_di'] only\n"
            "- Use supertrend as indicators['supertrend']['supertrend|direction'] only\n"
            "- Use stochastic as indicators['stochastic']['stoch_k|stoch_d'] only\n"
            "- Do NOT compare dict indicators directly (e.g. avoid `adx > threshold`)\n"
            "- For bitwise `&` / `|`, each side must be a boolean mask expression\n"
            "- ema/rsi/atr are arrays: do not use sub-keys like indicators['ema']['ema_21']\n"
            "- Do not use df['rsi']/df['ema']/df['bollinger']\n"
            "- Return only Python code in one ```python block\n\n"
            f"Current proposal context: {proposal}\n\n"
            "Failing code:\n"
            "```python\n"
            f"{failing_code}\n"
            "```"
        )
        response = self._chat_llm(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a senior Python quant developer. "
                        "Fix runtime errors in trading strategy code. "
                        "Output code only."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            phase="retry_code_runtime",
            max_tokens=4096,
        )
        return _extract_python_from_response(response.content)

    def _ask_analysis(
        self,
        session: BuilderSession,
        iteration: BuilderIteration,
        diagnostic: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Analyse le résultat et décide de continuer ou accepter.

        Args:
            diagnostic: résultat de compute_diagnostic() — enrichit le prompt.

        Returns:
            (analysis_text, decision) où decision ∈ {"continue", "accept", "stop"}
        """
        if not iteration.backtest_result:
            return "Pas de résultat de backtest disponible.", "continue"

        metrics = iteration.backtest_result.metrics
        n_trades = metrics.get("total_trades", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        sortino = metrics.get("sortino_ratio", 0)
        ret = metrics.get("total_return_pct", 0)
        dd = metrics.get("max_drawdown_pct", 0)
        wr = metrics.get("win_rate_pct", 0)
        pf = metrics.get("profit_factor", 0)
        exp = metrics.get("expectancy", 0)

        # --- Construire le prompt enrichi ---
        lines = [
            f"## Analyse — itération {iteration.iteration}/{session.max_iterations}",
            f"Objectif: {session.objective}",
            f"Hypothèse testée: {iteration.hypothesis}",
            f"Marché: {session.symbol} {session.timeframe} ({session.n_bars} barres, {session.date_range_start} → {session.date_range_end})",
            f"Configuration: capital={session.initial_capital}$, fees={session.fees_bps}bps, slippage={session.slippage_bps}bps",
            "",
            "### Résultats",
            f"- Sharpe: {sharpe:.3f}  |  Sortino: {sortino:.3f}",
            f"- Return: {ret:+.2f}%  |  MaxDD: {dd:.2f}%",
            f"- Trades: {n_trades}  |  WinRate: {wr:.1f}%  |  PF: {pf:.2f}",
            f"- Expectancy: {exp:.3f}",
            f"- Meilleur Sharpe session: {session.best_sharpe:.3f}",
        ]

        # Diagnostic pré-calculé
        if diagnostic:
            lines.append("")
            lines.append("### Diagnostic automatique")
            lines.append(f"Catégorie: {diagnostic.get('category', '?')} ({diagnostic.get('severity', '?')})")
            lines.append(f"Résumé: {diagnostic.get('summary', '')}")
            lines.append(f"Modification: {diagnostic.get('change_type', '?')}")
            sc = diagnostic.get("score_card", {})
            if sc:
                grades = ", ".join(f"{k}: {v['grade']}" for k, v in sc.items())
                lines.append(f"Score card: {grades}")
            trend = diagnostic.get("trend", "")
            if trend:
                lines.append(f"Tendance: {trend} {diagnostic.get('trend_detail', '')}")
            for a in diagnostic.get("actions", []):
                lines.append(f"  → {a}")
            for d in diagnostic.get("donts", []):
                lines.append(f"  ⚠️ {d}")

            # Auto-accept si target atteint + robuste
            cat = diagnostic.get("category", "")
            if cat == "target_reached" and n_trades > 20 and abs(dd) < 40:
                return (
                    f"Cible atteinte (Sharpe {sharpe:.3f}), stratégie robuste "
                    f"({n_trades} trades, DD {dd:.1f}%). Acceptation automatique.",
                    "accept",
                )

            # Alerte stagnation
            if diagnostic.get("trend") == "stagnated":
                lines.append("")
                lines.append("⚠️ STAGNATION DÉTECTÉE — même catégorie 3× de suite.")
                lines.append("Tu DOIS changer d'approche radicalement.")

        remaining = session.max_iterations - iteration.iteration
        lines.append("")
        lines.append(f"Itérations restantes: {remaining}")
        lines.append("")
        lines.append('Réponds en JSON: {{"analysis": "...", "decision": "accept|continue|stop", "suggestions": [...]}}')

        prompt = "\n".join(lines)

        response = self._chat_llm(
            messages=[
                LLMMessage(role="system", content=(
                    "Tu es un analyste quantitatif expert. "
                    "Analyse les résultats de backtest et le diagnostic fourni. "
                    "Décide: accept (cible atteinte + robuste), "
                    "continue (amélioration possible), stop (impasse). "
                    "Privilégie 'continue' tant que des tests supplémentaires "
                    "peuvent améliorer la stratégie. "
                    "Sois concis. Réponds en JSON."
                )),
                LLMMessage(role="user", content=prompt),
            ],
            phase="analysis",
            json_mode=True,
        )

        parsed = _extract_json_from_response(response.content)
        analysis = parsed.get("analysis", response.content[:500])
        decision = parsed.get("decision", "continue")

        if decision not in ("continue", "accept", "stop"):
            decision = "continue"

        return analysis, decision

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt_proposal() -> str:
        return """You are an expert quantitative trading strategy designer.
You design strategies using ONLY the available indicators listed in the user prompt.
You NEVER invent new indicators — only combine existing ones with clever logic.

RULES:
- Respond with ONLY valid JSON (no markdown, no commentary, no thinking)
- Every indicator in used_indicators must exist in the available list
- Always include ATR-based stop-loss/take-profit in risk_management
- Use 2-3 indicators max to avoid overfitting
- Include realistic default_params with sensible ranges in parameter_specs
- hypothesis must explain WHY this combination should work, not just WHAT it does
- Never output placeholder values (e.g. "brief description", "when to BUY")
- This phase is proposal-only: NEVER output Python code

Focus on signal quality, risk management, and robustness."""

    @staticmethod
    def _system_prompt_code() -> str:
        return f"""You are an expert Python developer specializing in trading systems.
Generate a COMPLETE, WORKING Python strategy class.

CRITICAL RULES:
1. Class name MUST be EXACTLY '{GENERATED_CLASS_NAME}' — any other name = FATAL ERROR
2. Inherit from StrategyBase
3. generate_signals returns pd.Series: 1.0=LONG, -1.0=SHORT, 0.0=FLAT
4. Use indicators from the 'indicators' dict (pre-computed by engine)
5. ALWAYS wrap indicators with np.nan_to_num() before any comparison
6. NEVER use os, subprocess, eval, exec, open, or __import__
7. ONLY import: numpy, pandas, strategies.base, utils.parameters
8. Do NOT use triple-quoted docstrings — use single-line # comments ONLY
9. Output ONLY Python code in a ```python block. No text before or after.
10. Skip warmup: set signals.iloc[:50] = 0.0 to avoid NaN-driven false signals
11. STRICT CHANGE CONTRACT:
   - if proposal.change_type == "params": keep same required_indicators and same generate_signals logic.
     Only edit default_params (and optionally parameter_specs).
   - if proposal.change_type == "logic": modify logic/indicators/filters.
   - if proposal.change_type == "both": modify both logic and params.
12. This phase is code-only: NEVER output JSON/proposal/objective rewrite.
13. Never access indicators via df['rsi']/df['ema']/df['bollinger']; always use indicators['name'].
14. For dict indicators (bollinger/macd/adx/stochastic/etc), access sub-keys before np.nan_to_num.
15. Indicator values are numpy arrays (or dict of numpy arrays): NEVER use .iloc/.loc/.shift/.rolling on indicators.
16. Bollinger must be used as indicators['bollinger']['upper|middle|lower'] (never indicators['bollinger_upper']).
17. EMA/RSI/ATR are plain arrays: NEVER use sub-keys like indicators['ema']['ema_21'].
18. ADX must be used as indicators['adx']['adx|plus_di|minus_di'] (never compare indicators['adx'] directly).
19. Supertrend must be used as indicators['supertrend']['supertrend|direction'] (no upper/lower keys).
20. Stochastic must be used as indicators['stochastic']['stoch_k|stoch_d'] (no 'signal' key).
21. For bitwise '&' and '|', both sides must be boolean mask expressions (never float/int scalars).
22. ALWAYS set "leverage": 1 in default_params. The backtest engine defaults to leverage=3 which ruins accounts.
23. To implement ATR-based SL/TP, write price levels into the DataFrame columns:
    - df.loc[:, "bb_stop_long"] = entry_price - stop_atr_mult * atr  (NaN where no entry)
    - df.loc[:, "bb_tp_long"]   = entry_price + tp_atr_mult * atr    (NaN where no entry)
    - df.loc[:, "bb_stop_short"] / df.loc[:, "bb_tp_short"] for short positions.
    The simulator reads these columns automatically. Only write values on entry signal bars (NaN elsewhere).

The code must be ready to execute with ZERO modifications."""

    # ------------------------------------------------------------------
    # Core: load strategy dynamically
    # ------------------------------------------------------------------

    def _save_and_load(
        self,
        session: BuilderSession,
        code: str,
        iteration_num: int,
    ) -> type:
        """Sauvegarde le code et charge dynamiquement la classe.

        Raises:
            ImportError: Si le module ne peut pas être chargé
            AttributeError: Si la classe attendue n'existe pas
        """
        strategy_path = session.session_dir / "strategy.py"
        strategy_path.write_text(code, encoding="utf-8")

        # Sauvegarder aussi une copie versionnée
        versioned = session.session_dir / f"strategy_v{iteration_num}.py"
        versioned.write_text(code, encoding="utf-8")

        # Charger dynamiquement
        module_name = f"sandbox_{session.session_id}_v{iteration_num}"

        # Supprimer ancien module du cache si présent
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Impossible de créer spec pour {strategy_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        cls = getattr(module, GENERATED_CLASS_NAME, None)
        if cls is None or not isinstance(cls, type):
            raise AttributeError(
                f"Classe '{GENERATED_CLASS_NAME}' absente du module généré"
            )

        return cast(type, cls)

    # ------------------------------------------------------------------
    # Core: auto-fix required_indicators from code inspection
    # ------------------------------------------------------------------

    def _auto_fix_required_indicators(
        self, strategy_cls: type, code: str
    ) -> type:
        """Détecte les indicateurs utilisés dans le code généré et complète required_indicators.

        Scanne le code pour les patterns indicators["xxx"] et indicators['xxx'],
        cross-référence avec le registre, et monkey-patche la propriété si des
        indicateurs sont manquants.

        Returns:
            La classe (éventuellement patchée)
        """
        # Extraire tous les noms d'indicateurs référencés dans le code
        try:
            tree = ast.parse(code)
            used_in_code = _collect_indicator_names(tree)
        except Exception:
            used_in_code = set(
                re.findall(r'indicators\s*\[\s*["\'](\w+)["\']\s*\]', code)
            )

        if not used_in_code:
            return strategy_cls

        # Comparer avec required_indicators déclarés
        try:
            instance = strategy_cls()
            declared = set(
                name.lower() for name in instance.required_indicators
            )
        except Exception:
            return strategy_cls

        # Vérifier contre le registre réel
        known_indicators = set(name.lower() for name in list_indicators())
        missing = {
            ind for ind in used_in_code
            if ind.lower() in known_indicators and ind.lower() not in declared
        }

        if not missing:
            return strategy_cls

        # Monkey-patch required_indicators pour ajouter les manquants
        new_required = list(instance.required_indicators) + sorted(missing)
        logger.warning(
            "auto_fix_required_indicators added=%s total=%s",
            sorted(missing), new_required
        )

        strategy_cls._patched_required = new_required

        @property
        def _patched_required_indicators(self):
            return type(self)._patched_required

        strategy_cls.required_indicators = _patched_required_indicators

        return strategy_cls

    # ------------------------------------------------------------------
    # Core: run backtest on generated strategy
    # ------------------------------------------------------------------

    def _run_backtest(
        self,
        strategy_cls: type,
        data: pd.DataFrame,
        params: Dict[str, Any],
        initial_capital: float = 10000.0,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        fees_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> Any:
        """Lance un backtest sur la stratégie générée.

        Utilise BacktestEngine directement avec la classe instanciée.
        """
        run_id = generate_run_id()
        engine = BacktestEngine(initial_capital=initial_capital, run_id=run_id)

        # Instancier la stratégie
        strategy_instance = strategy_cls()

        # Injecter fees/slippage dans params pour le moteur
        merged_params = dict(params)
        merged_params.setdefault("fees_bps", fees_bps)
        merged_params.setdefault("slippage_bps", slippage_bps)

        # Exécuter le backtest via l'engine (mode objet)
        result = engine.run(
            df=data,
            strategy=strategy_instance,
            params=merged_params,
            symbol=symbol,
            timeframe=timeframe,
            silent_mode=True,
        )

        # Convertir en résultat léger avec .metrics dict
        metrics_pct = normalize_metrics(result.metrics, "pct")

        return SimpleNamespace(
            success=True,
            metrics=metrics_pct,
            sharpe_ratio=metrics_pct.get("sharpe_ratio", 0.0),
            total_return_pct=metrics_pct.get("total_return_pct", 0.0),
            max_drawdown_pct=metrics_pct.get("max_drawdown_pct", 0.0),
            total_trades=metrics_pct.get("total_trades", 0),
            execution_time_ms=getattr(result, "execution_time_ms", 0),
        )

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(
        self,
        objective: str,
        data: pd.DataFrame,
        *,
        max_iterations: int = 10,
        target_sharpe: float = 1.0,
        initial_capital: float = 10000.0,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        fees_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> BuilderSession:
        """
        Lance la boucle complète de construction de stratégie.

        Args:
            objective: Description textuelle de la stratégie souhaitée
            data: DataFrame OHLCV pour backtest
            max_iterations: Nombre max d'itérations
            target_sharpe: Sharpe cible pour acceptation automatique
            initial_capital: Capital initial pour les backtests
            symbol: Symbole/token (ex: BTCUSDT, DOGEUSDC)
            timeframe: Timeframe des données (ex: 1h, 5m, 4h)
            fees_bps: Frais de trading en basis points
            slippage_bps: Slippage en basis points

        Returns:
            BuilderSession avec l'historique complet et le meilleur résultat
        """
        raw_objective = str(objective or "")
        objective = sanitize_objective_text(raw_objective)
        if not objective and not _looks_like_log_pollution(raw_objective):
            objective = raw_objective.strip()
        if raw_objective.strip() != objective:
            logger.warning(
                "builder_objective_sanitized raw_len=%d clean_len=%d",
                len(raw_objective),
                len(objective),
            )
        if not objective:
            raise ValueError(
                "Objectif Builder vide ou invalide après nettoyage "
                "(probable collage de logs/traceback)."
            )

        session_id = self.create_session_id(objective)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Calculer le contexte de marché à partir des données
        n_bars = len(data)
        date_range_start = ""
        date_range_end = ""
        try:
            idx = data.index
            if hasattr(idx, 'min'):
                date_range_start = str(idx.min())[:19]
                date_range_end = str(idx.max())[:19]
        except Exception:
            pass

        session = BuilderSession(
            session_id=session_id,
            objective=objective,
            session_dir=session_dir,
            available_indicators=self.available_indicators,
            max_iterations=max_iterations,
            target_sharpe=target_sharpe,
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
        )

        logger.info(
            "strategy_builder_start session=%s objective='%s' indicators=%d",
            session_id, objective, len(self.available_indicators),
        )

        # ── Flux de pensée temps réel ──
        model_name = getattr(getattr(self.llm, 'config', None), 'model', '?')
        ts = ThoughtStream(session_id, objective, model_name)

        last_iteration: Optional[BuilderIteration] = None
        consecutive_failures = 0
        fallback_count = 0  # compteur de fallbacks déterministes dans la session

        for i in range(1, max_iterations + 1):
            iteration = BuilderIteration(iteration=i)
            ts.iteration_start(i, max_iterations)

            # ── Circuit breaker ──
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                ts.circuit_breaker(consecutive_failures, MAX_CONSECUTIVE_FAILURES)
                logger.warning(
                    "builder_circuit_breaker consecutive=%d",
                    consecutive_failures,
                )
                session.status = "failed"
                break

            # ── Circuit breaker fallback ──
            if fallback_count >= MAX_DETERMINISTIC_FALLBACKS:
                ts.warning(
                    f"Arrêt: {fallback_count} fallbacks déterministes utilisés. "
                    "Le LLM ne parvient pas à générer du code valide pour cette API."
                )
                logger.warning(
                    "builder_fallback_circuit_breaker count=%d",
                    fallback_count,
                )
                session.status = "failed"
                break

            try:
                # ── Phase 1 : Proposition ──
                logger.info("builder_iter_%d_proposal", i)
                ts.proposal_sent(has_previous=last_iteration is not None)
                t0 = time.perf_counter()
                proposal, proposal_feedback = self._ask_proposal(
                    session, last_iteration
                )
                iteration.phase_feedback["proposal"] = proposal_feedback
                dt_proposal = time.perf_counter() - t0

                # Garde : proposition vide → retry avec prompt simplifié
                if _is_invalid_proposal(proposal):
                    ts.warning(
                        "Proposition invalide/hors phase — retry avec prompt simplifié"
                    )
                    ts.retry("proposition", 2)
                    logger.warning(
                        "builder_iter_%d_invalid_proposal proposal=%s — retrying",
                        i, proposal,
                    )
                    t0 = time.perf_counter()
                    proposal = self._retry_proposal_simple(objective)
                    iteration.phase_feedback.setdefault(
                        "proposal", {}
                    )["fallback_retry_used"] = True
                    iteration.phase_feedback.setdefault(
                        "proposal", {}
                    )["issues_after_retry"] = _proposal_issues(proposal)
                    dt_proposal = time.perf_counter() - t0

                    if _is_invalid_proposal(proposal):
                        ts.warning(
                            "Proposition toujours invalide après retry — skip itération"
                        )
                        iteration.error = (
                            "LLM incapable de produire une proposition. "
                            "Vérifiez le modèle ou l'objectif."
                        )
                        consecutive_failures += 1
                        session.iterations.append(iteration)
                        last_iteration = iteration
                        continue

                iteration.hypothesis = proposal.get(
                    "hypothesis", f"Itération {i}"
                )
                proposal["change_type"] = _normalize_change_type(
                    proposal.get("change_type", "logic")
                )
                policy_ct = _policy_change_type_override(
                    session=session,
                    last_iteration=last_iteration,
                )
                if policy_ct and proposal["change_type"] != policy_ct:
                    iteration.phase_feedback.setdefault("proposal", {})[
                        "change_type_overridden"
                    ] = {
                        "from": proposal["change_type"],
                        "to": policy_ct,
                        "reason": "diagnostic_policy",
                    }
                    logger.info(
                        "builder_iter_%d_change_type_overridden from=%s to=%s",
                        i,
                        proposal["change_type"],
                        policy_ct,
                    )
                    proposal["change_type"] = policy_ct
                iteration.change_type = proposal["change_type"]
                ts.proposal_received(proposal, dt_proposal)

                # Valider que les indicateurs demandés existent
                used = proposal.get("used_indicators", [])
                unknown = [
                    ind for ind in used
                    if ind.lower() not in (x.lower() for x in self.available_indicators)
                ]
                if unknown:
                    logger.warning(
                        "builder_unknown_indicators unknown=%s", unknown
                    )
                    proposal["used_indicators"] = [
                        ind for ind in used if ind.lower() in
                        (x.lower() for x in self.available_indicators)
                    ]

                # ── Phase 2 : Génération de code ──
                logger.info("builder_iter_%d_codegen", i)
                ts.codegen_sent()
                t0 = time.perf_counter()
                change_type = _normalize_change_type(
                    proposal.get("change_type", "logic")
                )
                has_stable_base_code = bool(
                    last_iteration
                    and last_iteration.code
                    and last_iteration.error is None
                    and last_iteration.backtest_result is not None
                )
                if change_type == "params" and not has_stable_base_code:
                    iteration.phase_feedback.setdefault("proposal", {})[
                        "change_type_overridden"
                    ] = {
                        "from": "params",
                        "to": "logic",
                        "reason": "no_stable_base_code",
                    }
                    logger.info(
                        "builder_iter_%d_change_type_overridden from=params to=logic "
                        "reason=no_stable_base_code",
                        i,
                    )
                    change_type = "logic"
                    proposal["change_type"] = "logic"
                    iteration.change_type = "logic"
                code: str
                code_feedback: Dict[str, Any] = {
                    "phase": "code",
                    "initial_kind": "local_patch",
                    "realign_attempts": 0,
                    "realign_success": False,
                    "final_valid": True,
                }
                if change_type == "params" and has_stable_base_code and last_iteration and last_iteration.code:
                    patched = _rewrite_default_params_from_proposal(
                        last_iteration.code, proposal,
                    )
                    if patched:
                        code = patched
                        code_feedback["source"] = "params_patch"
                        code_feedback["final_kind"] = "python"
                        iteration.phase_feedback["code"] = code_feedback
                        logger.info(
                            "builder_iter_%d_params_only_patch applied (no logic rewrite)",
                            i,
                        )
                    else:
                        code, code_feedback = self._ask_code(
                            session, proposal, last_iteration
                        )
                        iteration.phase_feedback["code"] = code_feedback
                else:
                    code, code_feedback = self._ask_code(
                        session, proposal, last_iteration
                    )
                    iteration.phase_feedback["code"] = code_feedback
                dt_code = time.perf_counter() - t0

                # Garde : code vide/trivial → retry avec prompt simplifié
                if not _looks_like_strategy_code(code, code):
                    ts.warning(
                        f"Code invalide/hors phase ({len(code.strip().splitlines())} lignes)"
                        " — retry avec prompt simplifié"
                    )
                    ts.retry("code", 2)
                    logger.warning("builder_iter_%d_empty_code — retrying", i)
                    t0 = time.perf_counter()
                    code = self._retry_code_simple(proposal)
                    iteration.phase_feedback.setdefault("code", {})[
                        "fallback_retry_used"
                    ] = True
                    dt_code = time.perf_counter() - t0

                    if not _looks_like_strategy_code(code, code):
                        ts.warning(
                            "Code toujours invalide après retry — skip itération"
                        )
                        iteration.code = code
                        iteration.error = (
                            "LLM incapable de générer du code valide. "
                            f"Reçu: {len(code.strip().splitlines())} lignes."
                        )
                        consecutive_failures += 1
                        session.iterations.append(iteration)
                        last_iteration = iteration
                        continue

                iteration.code = code
                ts.codegen_received(code, dt_code)

                # Contrat strict params-only: logique identique entre itérations.
                if change_type == "params" and last_iteration and last_iteration.code:
                    contract_ok, contract_reason = _params_only_contract_respected(
                        last_iteration.code,
                        code,
                    )
                    if not contract_ok:
                        ts.warning(f"Violation params-only: {contract_reason}")
                        logger.warning(
                            "builder_iter_%d_params_only_violation: %s",
                            i,
                            contract_reason,
                        )
                        patched = _rewrite_default_params_from_proposal(
                            last_iteration.code, proposal,
                        )
                        if patched:
                            code = patched
                            iteration.code = code
                            ts.warning(
                                "Correctif automatique appliqué: "
                                "logique précédente conservée, params réécrits."
                            )
                        else:
                            # Fallback non bloquant: conserver la version précédente
                            # plutôt que casser la session entière sur une itération params.
                            code = last_iteration.code
                            iteration.code = code
                            iteration.phase_feedback.setdefault("code", {})[
                                "params_contract_fallback"
                            ] = "reused_previous_code"
                            ts.warning(
                                "Fallback params-only: code précédent conservé "
                                "(patch default_params impossible)."
                            )

                # ── Phase 3 : Auto-repair + Validation syntaxe + sécurité ──
                code = _repair_code(code)
                iteration.code = code
                is_valid, error_msg = validate_generated_code(code)

                # Si invalide → retry avec prompt squelette + repair
                if not is_valid:
                    ts.warning(f"Code invalide: {error_msg} — retry simplifié")
                    ts.retry("code_validation", 2)
                    logger.warning(
                        "builder_iter_%d_invalid code=%s — retrying", i, error_msg,
                    )
                    iteration.phase_feedback.setdefault("code", {})[
                        "validation_error"
                    ] = error_msg
                    retry_code = self._retry_code_simple(proposal)
                    retry_code = _repair_code(retry_code)
                    is_valid_r, error_msg_r = validate_generated_code(retry_code)
                    if is_valid_r:
                        code = retry_code
                        iteration.code = code
                        is_valid, error_msg = True, ""
                    else:
                        iteration.phase_feedback.setdefault("code", {})[
                            "validation_error_retry"
                        ] = error_msg_r
                        # Fallback déterministe pour ne pas perdre l'itération
                        fallback_code = _build_deterministic_fallback_code(
                            proposal, variant=fallback_count,
                        )
                        fallback_count += 1
                        fallback_code = _repair_code(fallback_code)
                        is_valid_fb, error_msg_fb = validate_generated_code(fallback_code)
                        if is_valid_fb:
                            code = fallback_code
                            iteration.code = code
                            iteration.phase_feedback.setdefault("code", {})[
                                "fallback_deterministic_used"
                            ] = True
                            iteration.phase_feedback.setdefault("code", {})[
                                "source"
                            ] = "deterministic_fallback"
                            iteration.phase_feedback.setdefault("code", {})[
                                "fallback_variant"
                            ] = fallback_count - 1
                            ts.warning(
                                f"Code LLM invalide après retry: fallback déterministe v{fallback_count - 1} appliqué."
                            )
                            is_valid, error_msg = True, ""
                        else:
                            error_msg = (
                                f"{error_msg} | retry: {error_msg_r} | "
                                f"fallback: {error_msg_fb}"
                            )

                ts.validation(is_valid, error_msg)
                if not is_valid:
                    iteration.error = f"Validation échouée: {error_msg}"
                    logger.warning("builder_iter_%d_invalid_final code=%s", i, error_msg)
                    consecutive_failures += 1
                    session.iterations.append(iteration)
                    last_iteration = iteration
                    continue

                # ── Phase 4 : Chargement dynamique ──
                logger.info("builder_iter_%d_load", i)
                strategy_cls = self._save_and_load(session, code, i)

                # ── Phase 4b : Auto-fix required_indicators ──
                strategy_cls = self._auto_fix_required_indicators(
                    strategy_cls, code
                )

                # ── Phase 5 : Backtest ──
                logger.info("builder_iter_%d_backtest", i)
                ts.backtest_start()
                default_params = proposal.get("default_params", {})
                try:
                    bt_result = self._run_backtest(
                        strategy_cls, data, default_params, initial_capital,
                        symbol=session.symbol,
                        timeframe=session.timeframe,
                        fees_bps=session.fees_bps,
                        slippage_bps=session.slippage_bps,
                    )
                except Exception as bt_exc:
                    bt_error = f"{type(bt_exc).__name__}: {bt_exc}"
                    tb = traceback.format_exc()
                    tb_tail = ""
                    if tb:
                        tb_lines = [line.rstrip() for line in tb.splitlines() if line.rstrip()]
                        tb_tail = "\n".join(tb_lines[-16:]).strip()
                    iteration.phase_feedback.setdefault("backtest", {})[
                        "runtime_error"
                    ] = bt_error
                    if tb_tail:
                        iteration.phase_feedback.setdefault("backtest", {})[
                            "runtime_traceback_tail"
                        ] = tb_tail
                    ts.warning(
                        f"Backtest runtime error: {bt_error} — tentative auto-fix"
                    )
                    logger.warning(
                        "builder_iter_%d_backtest_runtime_error: %s", i, bt_error
                    )

                    runtime_error_for_llm = bt_error
                    if tb_tail:
                        runtime_error_for_llm = (
                            f"{bt_error}\n\nTraceback (tail):\n{tb_tail}"
                        )

                    retry_code = self._retry_code_runtime_fix(
                        proposal=proposal,
                        failing_code=code,
                        runtime_error=runtime_error_for_llm,
                    )
                    retry_code = _repair_code(retry_code)
                    valid_retry, retry_err = validate_generated_code(retry_code)
                    used_runtime_fallback = False
                    if not valid_retry:
                        iteration.phase_feedback.setdefault("backtest", {})[
                            "runtime_fix_validation_error"
                        ] = retry_err
                        fallback_code = _build_deterministic_fallback_code(
                            proposal, variant=fallback_count,
                        )
                        fallback_count += 1
                        fallback_code = _repair_code(fallback_code)
                        valid_fb, fb_err = validate_generated_code(fallback_code)
                        if not valid_fb:
                            raise ValueError(
                                "Runtime-fix invalide et fallback déterministe invalide: "
                                f"{retry_err} | {fb_err}"
                            )
                        retry_code = fallback_code
                        used_runtime_fallback = True
                        iteration.phase_feedback.setdefault("backtest", {})[
                            "runtime_fix_fallback_deterministic_used"
                        ] = True
                        iteration.phase_feedback.setdefault("code", {})[
                            "source"
                        ] = "deterministic_fallback"

                    retry_cls = self._save_and_load(session, retry_code, i)
                    retry_cls = self._auto_fix_required_indicators(
                        retry_cls, retry_code
                    )
                    try:
                        bt_result = self._run_backtest(
                            retry_cls, data, default_params, initial_capital,
                            symbol=session.symbol,
                            timeframe=session.timeframe,
                            fees_bps=session.fees_bps,
                            slippage_bps=session.slippage_bps,
                        )
                    except Exception as retry_bt_exc:
                        if used_runtime_fallback:
                            raise
                        iteration.phase_feedback.setdefault("backtest", {})[
                            "runtime_fix_retry_error"
                        ] = f"{type(retry_bt_exc).__name__}: {retry_bt_exc}"
                        fallback_code = _build_deterministic_fallback_code(
                            proposal, variant=fallback_count,
                        )
                        fallback_count += 1
                        fallback_code = _repair_code(fallback_code)
                        valid_fb2, fb_err2 = validate_generated_code(fallback_code)
                        if not valid_fb2:
                            raise ValueError(
                                "Runtime-fix backtest failed and deterministic fallback "
                                f"is invalid: {fb_err2}"
                            )
                        fallback_cls = self._save_and_load(session, fallback_code, i)
                        fallback_cls = self._auto_fix_required_indicators(
                            fallback_cls, fallback_code
                        )
                        bt_result = self._run_backtest(
                            fallback_cls, data, default_params, initial_capital,
                            symbol=session.symbol,
                            timeframe=session.timeframe,
                            fees_bps=session.fees_bps,
                            slippage_bps=session.slippage_bps,
                        )
                        retry_code = fallback_code
                        used_runtime_fallback = True
                        iteration.phase_feedback.setdefault("backtest", {})[
                            "runtime_fix_fallback_deterministic_used"
                        ] = True
                        iteration.phase_feedback.setdefault("code", {})[
                            "source"
                        ] = "deterministic_fallback"
                    code = retry_code
                    iteration.code = retry_code
                    iteration.phase_feedback.setdefault("backtest", {})[
                        "runtime_fix_applied"
                    ] = True
                iteration.backtest_result = bt_result
                ts.backtest_result(bt_result.metrics)

                # Backtest réussi → reset circuit breaker
                consecutive_failures = 0

                # ── Phase 6 : Mise à jour best ──
                metrics_cur = bt_result.metrics or {}
                sharpe = _metric_float(metrics_cur, "sharpe_ratio", float("-inf"))
                rank_sharpe = _ranking_sharpe(metrics_cur)
                if rank_sharpe > session.best_sharpe:
                    session.best_sharpe = rank_sharpe
                    session.best_iteration = iteration
                    ts.best_update(rank_sharpe, i)

                # ── Phase 6b : Détection de stagnation ──
                cur_fp = _metrics_fingerprint(metrics_cur)
                if last_iteration and last_iteration.backtest_result:
                    prev_fp = _metrics_fingerprint(
                        last_iteration.backtest_result.metrics or {}
                    )
                    if cur_fp == prev_fp:
                        iteration.phase_feedback.setdefault("stagnation", {})[
                            "identical_metrics"
                        ] = True
                        ts.warning(
                            "Stagnation détectée: métriques identiques à "
                            "l'itération précédente — forçage changement radical."
                        )
                        logger.warning(
                            "builder_iter_%d_stagnation fingerprint=%s",
                            i, cur_fp,
                        )
                        # Injecter un signal fort dans la proposition pour
                        # forcer le LLM à changer d'approche à l'itération suivante
                        proposal["_stagnation_detected"] = True

                # ── Phase 7 : Diagnostic déterministe + Analyse LLM ──
                logger.info("builder_iter_%d_diagnostic", i)
                diag_history = [
                    {
                        "sharpe": (
                            it.backtest_result.metrics.get("sharpe_ratio", 0)
                            if it.backtest_result else 0
                        ),
                        "diagnostic_category": it.diagnostic_category,
                    }
                    for it in session.iterations
                ]
                diag = compute_diagnostic(
                    bt_result.metrics, diag_history, session.target_sharpe,
                )
                iteration.diagnostic_category = diag["category"]
                iteration.diagnostic_detail = diag
                if not iteration.change_type:
                    iteration.change_type = _normalize_change_type(
                        diag.get("change_type", "logic")
                    )
                ts.diagnostic(diag)

                logger.info(
                    "builder_iter_%d_analysis diag=%s sev=%s",
                    i, diag["category"], diag["severity"],
                )
                ts.analysis_sent()
                t0 = time.perf_counter()
                analysis, decision = self._ask_analysis(
                    session, iteration, diag,
                )
                dt_analysis = time.perf_counter() - t0

                # Garde anti-arrêt prématuré: forcer la phase d'ajustement
                # tant que la session n'a pas suffisamment itéré.
                successful_iters = (
                    sum(1 for it in session.iterations if it.backtest_result is not None)
                    + 1
                )
                if decision == "stop" and i < max_iterations:
                    if (
                        successful_iters < MIN_SUCCESSFUL_ITERATIONS_BEFORE_STOP
                        or session.best_sharpe < session.target_sharpe
                    ):
                        ts.warning(
                            "Décision 'stop' ignorée: poursuite obligatoire "
                            "de la phase test/ajustement."
                        )
                        logger.info(
                            "builder_iter_%d_stop_overridden successful_iters=%d "
                            "best_sharpe=%.3f target=%.3f",
                            i,
                            successful_iters,
                            session.best_sharpe,
                            session.target_sharpe,
                        )
                        decision = "continue"
                        analysis = (
                            f"{analysis}\n"
                            "[Policy] stop overridden to continue optimization."
                        )
                        iteration.phase_feedback.setdefault("decision", {})[
                            "stop_overridden"
                        ] = True

                trades = int(metrics_cur.get("total_trades", 0) or 0)
                max_dd = abs(float(metrics_cur.get("max_drawdown_pct", 0) or 0))
                accept_allowed, accept_reason = _is_accept_candidate(
                    metrics_cur,
                    target_sharpe=session.target_sharpe,
                )
                if decision == "accept" and i < max_iterations:
                    if not accept_allowed:
                        ts.warning(
                            "Décision 'accept' ignorée: qualité statistique "
                            "insuffisante, poursuite optimisation."
                        )
                        logger.info(
                            "builder_iter_%d_accept_overridden reason=%s trades=%d "
                            "rank_sharpe=%.3f target=%.3f max_dd=%.2f",
                            i,
                            accept_reason,
                            trades,
                            session.best_sharpe,
                            session.target_sharpe,
                            max_dd,
                        )
                        decision = "continue"
                        analysis = (
                            f"{analysis}\n"
                            "[Policy] accept overridden to continue optimization."
                        )
                        iteration.phase_feedback.setdefault("decision", {})[
                            "accept_overridden"
                        ] = True

                iteration.analysis = analysis
                iteration.decision = decision
                ts.analysis_received(
                    analysis, decision, iteration.change_type, dt_analysis,
                )

                session.iterations.append(iteration)
                last_iteration = iteration

                logger.info(
                    "builder_iter_%d_done sharpe=%.3f decision=%s",
                    i, sharpe, decision,
                )

                if decision == "accept":
                    accept_now, accept_now_reason = _is_accept_candidate(
                        metrics_cur,
                        target_sharpe=session.target_sharpe,
                    )
                    if accept_now:
                        session.status = "success"
                    else:
                        session.status = "failed"
                        logger.info(
                            "builder_iter_%d_accept_rejected reason=%s",
                            i,
                            accept_now_reason,
                        )
                    break
                if decision == "stop":
                    best_metrics = (
                        session.best_iteration.backtest_result.metrics
                        if session.best_iteration and session.best_iteration.backtest_result
                        else {}
                    )
                    best_ok, best_reason = _is_accept_candidate(
                        best_metrics,
                        target_sharpe=session.target_sharpe,
                    )
                    session.status = "success" if best_ok else "failed"
                    if not best_ok:
                        logger.info(
                            "builder_iter_%d_stop_rejected_success reason=%s "
                            "best_rank_sharpe=%.3f",
                            i,
                            best_reason,
                            session.best_sharpe,
                        )
                    break

            except Exception as e:
                iteration.error = f"{type(e).__name__}: {e}"
                ts.error(i, str(e))
                consecutive_failures += 1
                logger.error(
                    "builder_iter_%d_error error=%s\n%s",
                    i, e, traceback.format_exc(),
                )
                session.iterations.append(iteration)
                last_iteration = iteration

        else:
            session.status = "max_iterations"

        ts.session_end(
            session.status, session.best_sharpe, len(session.iterations),
        )

        # Sauvegarder le résumé de session
        self._save_session_summary(session)

        logger.info(
            "strategy_builder_end session=%s status=%s best_sharpe=%.3f iters=%d",
            session.session_id, session.status,
            session.best_sharpe, len(session.iterations),
        )

        return session

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_session_summary(self, session: BuilderSession) -> None:
        """Sauvegarde un résumé JSON de la session."""
        summary = {
            "session_id": session.session_id,
            "objective": session.objective,
            "status": session.status,
            "best_sharpe": session.best_sharpe,
            "total_iterations": len(session.iterations),
            "available_indicators": session.available_indicators,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "hypothesis": it.hypothesis,
                    "change_type": it.change_type,
                    "diagnostic_category": it.diagnostic_category,
                    "error": it.error,
                    "decision": it.decision,
                    "sharpe": (
                        it.backtest_result.metrics.get("sharpe_ratio", 0)
                        if it.backtest_result else None
                    ),
                    "return_pct": (
                        it.backtest_result.metrics.get("total_return_pct", 0)
                        if it.backtest_result else None
                    ),
                    "trades": (
                        it.backtest_result.metrics.get("total_trades", 0)
                        if it.backtest_result else None
                    ),
                    "score_card": (
                        it.diagnostic_detail.get("score_card")
                        if it.diagnostic_detail else None
                    ),
                    "phase_feedback": it.phase_feedback or None,
                }
                for it in session.iterations
            ],
        }

        summary_path = session.session_dir / "session_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Générateurs d'objectifs pour le mode autonome
# ---------------------------------------------------------------------------

# Groupes d'indicateurs par famille de stratégie (combinaisons cohérentes)
_INDICATOR_FAMILIES: Dict[str, Dict[str, Any]] = {
    "trend-following": {
        "label": "Trend-following",
        "primary": ["ema", "sma", "macd", "supertrend", "adx", "ichimoku", "vortex", "aroon"],
        "entry_templates": [
            "Entrée long quand {ind1} confirme une tendance haussière et {ind2} valide le momentum.",
            "Entrée sur croisement haussier de {ind1} avec filtre de tendance {ind2}.",
            "Position dans le sens de la tendance détectée par {ind1}, confirmée par {ind2}.",
        ],
        "exit_templates": [
            "Sortie sur retournement de {ind1} ou signal contraire de {ind2}.",
            "Sortie quand la tendance s'essouffle (divergence {ind1}/{ind2}).",
        ],
    },
    "mean-reversion": {
        "label": "Mean-reversion",
        "primary": ["bollinger", "rsi", "stochastic", "cci", "williams_r", "stoch_rsi", "keltner", "donchian"],
        "entry_templates": [
            "Entrée quand le prix touche la bande extrême de {ind1} avec {ind2} en zone de survente/surachat.",
            "Achat en survente ({ind1} < seuil) avec confirmation {ind2}, vente en surachat.",
            "Entrée contrariante quand {ind1} atteint un extrême et {ind2} montre un retournement.",
        ],
        "exit_templates": [
            "Sortie quand le prix revient vers la moyenne ({ind1} neutre).",
            "Take-profit au retour à la bande médiane, stop si {ind2} continue dans la tendance.",
        ],
    },
    "momentum": {
        "label": "Momentum",
        "primary": ["rsi", "macd", "momentum", "roc", "stochastic", "mfi"],
        "entry_templates": [
            "Entrée quand {ind1} dépasse son seuil de momentum avec confirmation {ind2}.",
            "Position quand le momentum ({ind1}) accélère et {ind2} est aligné.",
            "Entrée sur divergence haussière/baissière entre {ind1} et {ind2}.",
        ],
        "exit_templates": [
            "Sortie quand le momentum ({ind1}) s'épuise ou diverge du prix.",
            "Take-profit sur perte de momentum, stop basé sur ATR.",
        ],
    },
    "breakout": {
        "label": "Breakout",
        "primary": ["bollinger", "donchian", "keltner", "atr", "supertrend", "adx"],
        "entry_templates": [
            "Entrée sur cassure de la bande supérieure/inférieure de {ind1} avec volume confirmé.",
            "Position quand le prix sort du range {ind1} avec {ind2} montrant une expansion de volatilité.",
            "Entrée sur breakout validé par {ind1} et force de tendance ({ind2}).",
        ],
        "exit_templates": [
            "Sortie si le prix réintègre le range ou trailing stop basé sur ATR.",
            "Take-profit en multiple d'ATR, stop si faux breakout ({ind1} se contracte).",
        ],
    },
    "scalping": {
        "label": "Scalping",
        "primary": ["ema", "macd", "rsi", "stochastic", "vwap", "bollinger"],
        "entry_templates": [
            "Entrée rapide sur signal {ind1} avec confirmation {ind2} sur timeframe court.",
            "Scalp quand {ind1} croise en zone extrême avec {ind2} aligné.",
            "Entrée quand prix croise {ind1} avec {ind2} en confirmation, objectif serré.",
        ],
        "exit_templates": [
            "Sortie rapide : take-profit serré (1-1.5x ATR), stop-loss serré (0.5-1x ATR).",
            "Sortie sur premier signal de retournement de {ind1}.",
        ],
    },
    "multi-factor": {
        "label": "Multi-factor",
        "primary": ["ema", "rsi", "macd", "bollinger", "adx", "supertrend", "stochastic", "obv"],
        "entry_templates": [
            "Entrée quand au moins 3 facteurs sont alignés : tendance ({ind1}), momentum ({ind2}), volatilité ({ind3}).",
            "Signal composite : {ind1} + {ind2} + {ind3} doivent tous confirmer la direction.",
        ],
        "exit_templates": [
            "Sortie quand plus de la moitié des facteurs se retournent.",
            "Sortie progressive : réduction quand {ind1} diverge, clôture si {ind2} se retourne.",
        ],
    },
}

# Templates de risk management
_RISK_TEMPLATES = [
    "Stop-loss = {sl_mult}x ATR, take-profit = {tp_mult}x ATR.",
    "Stop-loss dynamique basé sur ATR ({sl_mult}x), ratio risk/reward {rr}:1.",
    "Trailing stop à {sl_mult}x ATR, take-profit à {tp_mult}x ATR.",
    "Stop serré {sl_mult}x ATR pour limiter le drawdown, TP à {tp_mult}x ATR.",
]


def generate_random_objective(
    symbol: "str | List[str]" = "BTCUSDC",
    timeframe: "str | List[str]" = "1h",
    available_indicators: Optional[List[str]] = None,
) -> str:
    """Génère un objectif de stratégie aléatoire à partir de templates.

    Accepte des listes de symboles/timeframes : un couple est choisi
    aléatoirement pour diversifier les objectifs en mode autonome.

    Combine une famille de stratégie, des indicateurs du registry,
    des conditions d'entrée/sortie et du risk management.

    Returns:
        Objectif structuré en français prêt à être passé au StrategyBuilder.
    """
    # Normaliser listes → valeur unique (choix aléatoire)
    if isinstance(symbol, list):
        symbol = random.choice(symbol) if symbol else "BTCUSDC"
    if isinstance(timeframe, list):
        timeframe = random.choice(timeframe) if timeframe else "1h"

    if available_indicators is None:
        available_indicators = list_indicators()

    avail_lower = {ind.lower() for ind in available_indicators}

    # Choisir une famille
    family_key = random.choice(list(_INDICATOR_FAMILIES.keys()))
    family = _INDICATOR_FAMILIES[family_key]

    # Filtrer les indicateurs disponibles dans cette famille
    valid_primary = [ind for ind in family["primary"] if ind.lower() in avail_lower]
    if len(valid_primary) < 2:
        valid_primary = [ind for ind in available_indicators if ind.lower() != "atr"]

    # Sélectionner 2-3 indicateurs + ATR pour le risk management
    n_indicators = random.randint(2, min(3, len(valid_primary)))
    selected = random.sample(valid_primary, n_indicators)
    if "atr" not in [s.lower() for s in selected] and "atr" in avail_lower:
        selected.append("atr")

    # Générer l'entrée
    ind1 = selected[0].upper()
    ind2 = selected[1].upper() if len(selected) > 1 else selected[0].upper()
    ind3 = selected[2].upper() if len(selected) > 2 else ind1

    entry = random.choice(family["entry_templates"]).format(
        ind1=ind1, ind2=ind2, ind3=ind3,
    )
    exit_rule = random.choice(family["exit_templates"]).format(
        ind1=ind1, ind2=ind2, ind3=ind3,
    )

    # Risk management
    sl_mult = round(random.uniform(1.0, 2.5), 1)
    tp_mult = round(sl_mult * random.uniform(1.5, 3.0), 1)
    rr = round(tp_mult / sl_mult, 1)
    risk = random.choice(_RISK_TEMPLATES).format(
        sl_mult=sl_mult, tp_mult=tp_mult, rr=rr,
    )

    indicators_str = " + ".join(ind.upper() for ind in selected)

    objective = (
        f"Stratégie de {family['label']} sur {symbol} {timeframe}. "
        f"Indicateurs : {indicators_str}. "
        f"{entry} "
        f"{exit_rule} "
        f"{risk}"
    )

    return objective


def generate_llm_objective(
    llm_client: Any,
    symbol: "str | List[str]" = "BTCUSDC",
    timeframe: "str | List[str]" = "1h",
    available_indicators: Optional[List[str]] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
    recent_markets: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Génère un objectif de stratégie via un appel LLM.

    Accepte des listes de symboles/timeframes : le LLM est invité à
    choisir le couple le plus pertinent pour sa stratégie.

    Returns:
        Objectif en texte libre généré par le LLM.
    """
    if available_indicators is None:
        available_indicators = list_indicators()

    indicators_list = ", ".join(sorted(available_indicators))

    # Normaliser en listes pour construire le prompt multi-marché
    symbols_list = symbol if isinstance(symbol, list) else [symbol]
    timeframes_list = timeframe if isinstance(timeframe, list) else [timeframe]
    symbols_list = [s for s in symbols_list if s] or ["BTCUSDC"]
    timeframes_list = [t for t in timeframes_list if t] or ["1h"]

    # Construire l'instruction marché selon l'univers disponible
    if len(symbols_list) > 1 or len(timeframes_list) > 1:
        # Mélanger pour réduire le biais de position (BTC toujours 1er)
        shuffled_symbols = symbols_list.copy()
        random.shuffle(shuffled_symbols)
        shuffled_timeframes = timeframes_list.copy()
        random.shuffle(shuffled_timeframes)

        market_instruction = (
            f"Symboles disponibles (SEULS autorisés) : {', '.join(shuffled_symbols)}\n"
            f"Timeframes disponibles (SEULS autorisés) : {', '.join(shuffled_timeframes)}\n"
            "CHOISIS le symbole et le timeframe les plus adaptés à ta stratégie. "
            "Tu ne DOIS utiliser QUE des symboles et timeframes de ces listes. "
            "N'invente AUCUN timeframe (pas de 3m, 5m, 2h, etc. s'ils ne sont pas listés). "
            "Ne te limite pas à BTC — explore les altcoins si ta stratégie s'y prête mieux.\n\n"
        )
        # Injecter l'historique récent pour forcer la diversité
        if recent_markets:
            recent_str = ", ".join(f"{s} {tf}" for s, tf in recent_markets[-6:])
            market_instruction += (
                f"IMPORTANT — Les marchés suivants ont DÉJÀ été utilisés récemment : {recent_str}. "
                "Tu DOIS choisir un couple symbol/timeframe DIFFÉRENT de ceux-ci. "
                "Varie les tokens ET les timeframes.\n\n"
            )
    else:
        market_instruction = f"Marché : {symbols_list[0]} en {timeframes_list[0]}.\n\n"

    system_msg = LLMMessage(
        role="system",
        content=(
            "Tu es un quant designer spécialisé en stratégies de trading crypto. "
            "Génère UN objectif de stratégie original et précis. "
            "Réponds UNIQUEMENT avec l'objectif, sans explication ni formatage markdown."
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=(
            f"Génère un objectif de stratégie de trading.\n\n"
            f"{market_instruction}"
            f"Indicateurs disponibles : {indicators_list}\n\n"
            "Format attendu :\n"
            "[Style] sur [marché] [timeframe]. "
            "Indicateurs : [ind1] + [ind2] + [ind3]. "
            "Entrées : [conditions]. "
            "Sorties : [conditions]. "
            "Risk management : [SL/TP].\n\n"
            "Sois créatif : explore des combinaisons inhabituelles, "
            "des filtres originaux, des approches multi-timeframe conceptuelles. "
            "L'objectif doit faire 2-4 phrases."
        ),
    )

    if stream_callback and hasattr(llm_client, "chat_stream"):
        result = llm_client.chat_stream(
            [system_msg, user_msg],
            on_chunk=lambda c: stream_callback("objective_gen", c),
            max_tokens=300,
        )
    else:
        result = llm_client.chat([system_msg, user_msg], max_tokens=300)

    objective = str(result).strip()
    # Nettoyer les tags <think> si présents
    objective = re.sub(r"<think>.*?</think>", "", objective, flags=re.DOTALL).strip()
    objective = re.sub(r"<think>.*", "", objective, flags=re.DOTALL).strip()
    objective = sanitize_objective_text(objective)

    # Fallback si le LLM retourne du vide
    if not objective or len(objective) < 20:
        logger.warning("generate_llm_objective: résultat LLM vide, fallback template")
        return generate_random_objective(symbol, timeframe, available_indicators)

    # ── Post-validation : remplacer les TF/tokens hallucinés ──
    tf_pattern = re.compile(r"\b(\d{1,2}[mhdwM])\b")
    found_tfs = tf_pattern.findall(objective)
    for found_tf in found_tfs:
        if found_tf not in timeframes_list:
            replacement = random.choice(timeframes_list)
            objective = objective.replace(found_tf, replacement, 1)
            logger.info(
                "generate_llm_objective: TF halluciné '%s' → '%s'",
                found_tf, replacement,
            )

    sym_upper_set = {s.upper() for s in symbols_list}
    # Vérifier que le symbole mentionné est valide
    sym_pattern = re.compile(r"\b([A-Z]{2,10}USDC)\b")
    found_syms = sym_pattern.findall(objective.upper())
    for found_sym in found_syms:
        if found_sym not in sym_upper_set:
            replacement = random.choice(symbols_list)
            objective = re.sub(
                re.escape(found_sym), replacement, objective,
                count=1, flags=re.IGNORECASE,
            )
            logger.info(
                "generate_llm_objective: token halluciné '%s' → '%s'",
                found_sym, replacement,
            )

    return objective


# ---------------------------------------------------------------------------
# Catalogue d'objectifs pré-construits pour exploration systématique
# ---------------------------------------------------------------------------

@dataclass
class CatalogObjective:
    """Un objectif pré-construit du catalogue d'exploration."""

    id: str
    family: str
    indicators: List[str]
    direction: str          # "long_only", "short_only", "long_short"
    risk_profile: str       # "tight", "balanced", "wide"
    description: str        # Objectif formaté ({symbol} et {timeframe} en placeholders)
    sl_mult: float
    tp_mult: float
    tags: List[str] = field(default_factory=list)


def _make_catalog_entry(
    family_key: str,
    indicators: List[str],
    direction: str,
    risk_name: str,
    sl_mult: float,
    tp_mult: float,
    tags: Optional[List[str]] = None,
    entry_idx: int = 0,
    exit_idx: int = 0,
) -> CatalogObjective:
    """Construit un CatalogObjective à partir des templates existants."""
    family = _INDICATOR_FAMILIES[family_key]
    entries = family["entry_templates"]
    exits = family["exit_templates"]
    entry_tpl = entries[entry_idx % len(entries)]
    exit_tpl = exits[exit_idx % len(exits)]

    inds_upper = {
        "ind1": indicators[0].upper(),
        "ind2": indicators[1].upper() if len(indicators) > 1 else indicators[0].upper(),
        "ind3": indicators[2].upper() if len(indicators) > 2 else indicators[0].upper(),
    }
    entry = entry_tpl.format(**inds_upper)
    exit_rule = exit_tpl.format(**inds_upper)

    dir_label = {
        "long_only": "Long uniquement.",
        "short_only": "Short uniquement.",
        "long_short": "Long et short.",
    }.get(direction, "Long et short.")

    rr = round(tp_mult / max(sl_mult, 0.1), 1)
    risk = f"Stop-loss = {sl_mult}x ATR, take-profit = {tp_mult}x ATR (RR {rr}:1)."
    indicators_str = " + ".join(ind.upper() for ind in indicators)

    description = (
        f"Stratégie de {family['label']} sur {{symbol}} {{timeframe}}. "
        f"Indicateurs : {indicators_str}. "
        f"{entry} {exit_rule} {dir_label} {risk}"
    )

    obj_id = hashlib.md5(
        f"{family_key}_{'-'.join(indicators)}_{direction}_{risk_name}_{entry_idx}".encode()
    ).hexdigest()[:12]

    return CatalogObjective(
        id=obj_id,
        family=family_key,
        indicators=list(indicators),
        direction=direction,
        risk_profile=risk_name,
        description=description,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        tags=tags or [],
    )


def _ensure_atr(inds: List[str]) -> List[str]:
    if "atr" not in [i.lower() for i in inds]:
        return list(inds) + ["atr"]
    return list(inds)


_RISK_PROFILES = {
    "tight":    (1.0, 2.0),
    "balanced": (1.5, 3.0),
    "wide":     (2.0, 5.0),
}

_SCALP_RISK_PROFILES = {
    "tight":    (0.5, 1.0),
    "balanced": (1.0, 1.5),
}


def _build_objective_catalog() -> List[CatalogObjective]:
    """Construit le catalogue complet d'objectifs pré-définis (~150)."""
    catalog: List[CatalogObjective] = []

    def _add(family: str, pairs: list, directions: list, risks: dict,
             tags_base: Optional[List[str]] = None):
        for pi, (inds, tags) in enumerate(pairs):
            full_inds = _ensure_atr(inds)
            all_tags = (tags_base or []) + tags
            for direction in directions:
                for risk_name, (sl, tp) in risks.items():
                    catalog.append(_make_catalog_entry(
                        family, full_inds, direction, risk_name, sl, tp,
                        tags=all_tags, entry_idx=pi, exit_idx=pi,
                    ))

    # ── Trend-following (~42) ──
    _add("trend-following", [
        (["ema", "macd"],        ["trending", "momentum"]),
        (["sma", "adx"],         ["trending", "strong_trend"]),
        (["supertrend", "adx"],  ["trending", "breakout"]),
        (["ema", "aroon"],       ["trending", "reversal"]),
        (["macd", "supertrend"], ["trending", "momentum"]),
        (["ema", "vortex"],      ["trending", "oscillator"]),
        (["sma", "aroon"],       ["trending", "pullback"]),
    ], ["long_only", "short_only"], _RISK_PROFILES)

    # ── Mean-reversion (~42) ──
    _add("mean-reversion", [
        (["bollinger", "rsi"],        ["ranging", "oversold"]),
        (["bollinger", "stochastic"], ["ranging", "overbought"]),
        (["keltner", "cci"],          ["ranging", "channel"]),
        (["donchian", "williams_r"],  ["ranging", "breakout"]),
        (["bollinger", "stoch_rsi"],  ["ranging", "extreme"]),
        (["keltner", "rsi"],          ["ranging", "mean_revert"]),
        (["donchian", "rsi"],         ["ranging", "pullback"]),
    ], ["long_only", "short_only"], _RISK_PROFILES)

    # ── Momentum (~30) ──
    _add("momentum", [
        (["rsi", "macd"],         ["momentum", "divergence"]),
        (["macd", "roc"],         ["momentum", "acceleration"]),
        (["momentum", "stochastic"], ["momentum", "oscillator"]),
        (["rsi", "mfi"],          ["momentum", "volume"]),
        (["macd", "stochastic"],  ["momentum", "confirmation"]),
    ], ["long_only", "short_only", "long_short"], {"balanced": (1.5, 3.0), "wide": (2.0, 4.5)})

    # ── Breakout (~24) ──
    _add("breakout", [
        (["bollinger", "adx"],      ["volatile", "expansion"]),
        (["donchian", "atr"],       ["breakout", "channel"]),
        (["keltner", "supertrend"], ["breakout", "trending"]),
        (["bollinger", "supertrend"], ["volatile", "trending"]),
    ], ["long_only", "short_only", "long_short"], {"balanced": (1.5, 3.5), "wide": (2.0, 5.0)})

    # ── Scalping (~12) ──
    _add("scalping", [
        (["ema", "stochastic"], ["scalp", "short_term"]),
        (["macd", "rsi"],       ["scalp", "momentum"]),
        (["bollinger", "vwap"], ["scalp", "mean_revert"]),
        (["ema", "rsi"],        ["scalp", "pullback"]),
        (["stochastic", "vwap"], ["scalp", "oscillator"]),
        (["ema", "macd"],       ["scalp", "cross"]),
    ], ["long_short"], _SCALP_RISK_PROFILES)

    # ── Multi-factor (~12) ──
    _add("multi-factor", [
        (["ema", "rsi", "bollinger"],         ["composite", "three_factor"]),
        (["supertrend", "adx", "stochastic"], ["composite", "trend_confirm"]),
        (["macd", "bollinger", "obv"],        ["composite", "volume"]),
        (["ema", "macd", "rsi"],              ["composite", "momentum"]),
    ], ["long_short"], _RISK_PROFILES)

    logger.info("objective_catalog_built count=%d", len(catalog))
    return catalog


# Singleton catalogue (lazy init)
_OBJECTIVE_CATALOG: Optional[List[CatalogObjective]] = None


def get_objective_catalog() -> List[CatalogObjective]:
    """Retourne le catalogue d'objectifs (lazy init)."""
    global _OBJECTIVE_CATALOG
    if _OBJECTIVE_CATALOG is None:
        _OBJECTIVE_CATALOG = _build_objective_catalog()
    return _OBJECTIVE_CATALOG


# ---------------------------------------------------------------------------
# Exploration Tracker — persistance et suivi de couverture
# ---------------------------------------------------------------------------

class ExplorationTracker:
    """Gère l'exploration systématique du catalogue d'objectifs.

    Persiste l'état dans un fichier JSON pour survivre aux redémarrages.
    """

    STATE_FILE = SANDBOX_ROOT / "_exploration_state.json"

    def __init__(self, catalog: List[CatalogObjective]):
        self.catalog = catalog
        self.catalog_by_id: Dict[str, CatalogObjective] = {
            obj.id: obj for obj in catalog
        }
        self._catalog_hash = self._compute_catalog_hash()
        self.state = self._load_or_create_state()

    # -- persistence --

    def _compute_catalog_hash(self) -> str:
        ids = "|".join(obj.id for obj in self.catalog)
        return hashlib.md5(ids.encode()).hexdigest()[:16]

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("catalog_hash") == self._catalog_hash:
                    return state
                logger.info("exploration_tracker_catalog_changed — reset")
            except Exception:
                pass
        return self._fresh_state()

    def _fresh_state(self) -> Dict[str, Any]:
        order = [obj.id for obj in self.catalog]
        random.shuffle(order)
        return {
            "version": "1.0",
            "catalog_hash": self._catalog_hash,
            "exploration_order": order,
            "current_index": 0,
            "explored": {},
        }

    def _save(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
        tmp.replace(self.STATE_FILE)

    # -- API --

    def get_next_objective(self) -> Optional[CatalogObjective]:
        """Retourne le prochain objectif non exploré, ou None si épuisé."""
        order = self.state["exploration_order"]
        explored = self.state["explored"]
        idx = self.state["current_index"]
        while idx < len(order):
            obj_id = order[idx]
            if obj_id not in explored and obj_id in self.catalog_by_id:
                self.state["current_index"] = idx
                return self.catalog_by_id[obj_id]
            idx += 1
        return None

    def mark_explored(
        self,
        obj_id: str,
        *,
        status: str,
        best_sharpe: float,
        session_id: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        self.state["explored"][obj_id] = {
            "tested_at": datetime.now().isoformat(),
            "status": status,
            "best_sharpe": best_sharpe,
            "session_id": session_id,
            "symbol": symbol,
            "timeframe": timeframe,
        }
        self.state["current_index"] = self.state.get("current_index", 0) + 1
        self._save()

    def get_coverage_stats(self) -> Dict[str, Any]:
        total = len(self.catalog)
        explored = self.state["explored"]
        n_explored = len(explored)
        n_success = sum(1 for v in explored.values() if v.get("status") == "success")
        best = max(
            (v.get("best_sharpe", float("-inf")) for v in explored.values()),
            default=0.0,
        )
        return {
            "total_objectives": total,
            "explored_count": n_explored,
            "success_count": n_success,
            "coverage_pct": round(100.0 * n_explored / max(total, 1), 1),
            "best_sharpe_overall": round(best, 3),
        }

    def reset(self) -> None:
        """Reset complet : re-shuffle et repart de zéro."""
        self.state = self._fresh_state()
        self._save()
        logger.info("exploration_tracker_reset")


# Singleton tracker (lazy init)
_EXPLORATION_TRACKER: Optional[ExplorationTracker] = None


def _get_exploration_tracker() -> ExplorationTracker:
    global _EXPLORATION_TRACKER
    if _EXPLORATION_TRACKER is None:
        _EXPLORATION_TRACKER = ExplorationTracker(get_objective_catalog())
    return _EXPLORATION_TRACKER


def get_next_catalog_objective(
    symbol: "str | List[str]" = "BTCUSDC",
    timeframe: "str | List[str]" = "1h",
) -> Optional[tuple[str, str]]:
    """Retourne (description, obj_id) du prochain objectif non exploré.

    Accepte des listes de symboles/timeframes : un couple est choisi
    aléatoirement pour diversifier l'exploration du catalogue.

    Returns:
        ``(objective_text, obj_id)`` ou ``None`` si le catalogue est épuisé.
    """
    # Normaliser listes → valeur unique (choix aléatoire)
    if isinstance(symbol, list):
        symbol = random.choice(symbol) if symbol else "BTCUSDC"
    if isinstance(timeframe, list):
        timeframe = random.choice(timeframe) if timeframe else "1h"

    tracker = _get_exploration_tracker()
    obj = tracker.get_next_objective()
    if obj is None:
        return None
    text = obj.description.replace("{symbol}", symbol).replace("{timeframe}", timeframe)
    return text, obj.id


def mark_catalog_objective_explored(
    obj_id: str,
    *,
    status: str,
    best_sharpe: float,
    session_id: str,
    symbol: str,
    timeframe: str,
) -> None:
    """Enregistre un objectif du catalogue comme exploré."""
    tracker = _get_exploration_tracker()
    tracker.mark_explored(
        obj_id,
        status=status,
        best_sharpe=best_sharpe,
        session_id=session_id,
        symbol=symbol,
        timeframe=timeframe,
    )


def get_catalog_coverage() -> Dict[str, Any]:
    """Retourne les statistiques de couverture du catalogue."""
    tracker = _get_exploration_tracker()
    return tracker.get_coverage_stats()


def reset_catalog_exploration() -> None:
    """Reset l'exploration du catalogue (re-shuffle)."""
    global _EXPLORATION_TRACKER
    tracker = _get_exploration_tracker()
    tracker.reset()
    _EXPLORATION_TRACKER = None  # force lazy re-init


def recommend_market_context(
    llm_client: Any,
    *,
    objective: str,
    candidate_symbols: List[str],
    candidate_timeframes: List[str],
    default_symbol: str = "BTCUSDC",
    default_timeframe: str = "1h",
    stream_callback: Optional[Callable[[str, str], None]] = None,
    recent_markets: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Recommande un couple (symbol, timeframe) adapté à un objectif Builder.

    Le choix est strictement borné à l'univers fourni (`candidate_symbols`,
    `candidate_timeframes`). En cas de réponse invalide du LLM, un fallback
    déterministe est appliqué.
    """

    def _unique_non_empty(values: List[str], *, upper: bool = False) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in values:
            val = str(raw or "").strip()
            if not val:
                continue
            if upper:
                val = val.upper()
            if val in seen:
                continue
            seen.add(val)
            out.append(val)
        return out

    symbol_re = re.compile(r"^[A-Za-z0-9_.-]{2,24}$")
    timeframe_re = re.compile(r"^\d+[mhdwM]$")

    symbols = _unique_non_empty(
        [*candidate_symbols, default_symbol or "BTCUSDC"],
        upper=True,
    )
    symbols = [s for s in symbols if symbol_re.match(s)]

    timeframes = _unique_non_empty(
        [*candidate_timeframes, default_timeframe or "1h"],
        upper=False,
    )
    timeframes = [tf for tf in timeframes if timeframe_re.match(tf)]

    fallback_symbol = symbols[0] if symbols else "BTCUSDC"
    fallback_timeframe = timeframes[0] if timeframes else "1h"

    if not symbols or not timeframes:
        return {
            "symbol": fallback_symbol,
            "timeframe": fallback_timeframe,
            "confidence": 0.0,
            "reason": "Univers marché incomplet, fallback par défaut.",
            "source": "fallback_no_candidates",
        }

    clean_objective = sanitize_objective_text(objective)
    if not clean_objective:
        clean_objective = str(objective or "").strip()

    # Mélanger pour réduire le biais de position
    shuffled_symbols = symbols.copy()
    random.shuffle(shuffled_symbols)
    shuffled_timeframes = timeframes.copy()
    random.shuffle(shuffled_timeframes)

    diversity_instruction = ""
    if recent_markets:
        recent_str = ", ".join(f"{s} {tf}" for s, tf in recent_markets[-6:])
        diversity_instruction = (
            f"\n- DÉJÀ UTILISÉS récemment : {recent_str}. "
            "Tu DOIS choisir un couple DIFFÉRENT. Varie tokens ET timeframes."
        )

    system_msg = LLMMessage(
        role="system",
        content=(
            "Tu es un analyste quant. Choisis UN seul couple symbole/timeframe "
            "le plus pertinent pour l'objectif. Réponds en JSON strict uniquement."
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=(
            "Objectif:\n"
            f"{clean_objective}\n\n"
            "Contraintes:\n"
            f"- symbol MUST be one of: {', '.join(shuffled_symbols)}\n"
            f"- timeframe MUST be one of: {', '.join(shuffled_timeframes)}\n"
            "- Retourne un JSON strict, sans markdown:\n"
            '{"symbol":"...","timeframe":"...","confidence":0.0,"reason":"..."}\n'
            f"- confidence doit être entre 0 et 1.{diversity_instruction}"
        ),
    )

    try:
        if stream_callback and hasattr(llm_client, "chat_stream"):
            raw = llm_client.chat_stream(
                [system_msg, user_msg],
                on_chunk=lambda c: stream_callback("market_pick", c),
                max_tokens=180,
            )
        else:
            raw = llm_client.chat([system_msg, user_msg], max_tokens=180)
        raw_text = str(raw or "").strip()
    except Exception as exc:
        logger.warning("recommend_market_context: fallback exception=%s", exc)
        return {
            "symbol": fallback_symbol,
            "timeframe": fallback_timeframe,
            "confidence": 0.0,
            "reason": f"Échec appel LLM ({exc}). Fallback appliqué.",
            "source": "fallback_exception",
        }

    payload = _extract_json_from_response(raw_text)
    symbol = str(payload.get("symbol", "")).strip().upper()
    timeframe = str(payload.get("timeframe", "")).strip()

    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason", "") or "").strip()
    if len(reason) > 280:
        reason = reason[:280].rstrip()

    source = "llm"
    if symbol not in symbols:
        source = "fallback_out_of_universe"
        symbol = fallback_symbol
    if timeframe not in timeframes:
        source = "fallback_out_of_universe"
        timeframe = fallback_timeframe

    if not payload:
        source = "fallback_invalid_json"
        symbol = fallback_symbol
        timeframe = fallback_timeframe
        confidence = 0.0
        if not reason:
            reason = "Réponse LLM non parseable en JSON. Fallback appliqué."

    if not reason:
        if source == "llm":
            reason = "Choix basé sur style de stratégie, volatilité attendue et fréquence des signaux."
        else:
            reason = "Choix par défaut suite à une réponse LLM non exploitable."

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "confidence": confidence,
        "reason": reason,
        "source": source,
    }
