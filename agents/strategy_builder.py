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
import importlib.util
import json
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, cast

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

    # 3. Vérifier generate_signals
    has_generate_signals = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
            has_generate_signals = True
            break
    if not has_generate_signals:
        return False, "Méthode 'generate_signals' absente."

    # 4. Imports dangereux
    dangerous_patterns = [
        "os.system", "subprocess", "eval(", "exec(",
        "__import__", "shutil.rmtree", "open(",
    ]
    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in code_lower:
            return False, f"Import/appel dangereux détecté: '{pattern}'"

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


def _repair_code(code: str) -> str:
    """Auto-repair des erreurs courantes du code genere par LLM.

    Corrige:
    - Tags <think> des modeles de raisonnement (qwen3, deepseek-r1)
    - Docstrings triple-quoted non terminées (cause #1 de crash)
    - Nom de classe incorrect (cause #2 de crash)
    """
    # 1. Retirer les tags <think> des modeles de raisonnement
    code = re.sub(r"<think>.*?</think>\s*", "", code, flags=re.DOTALL)
    code = re.sub(r"<think>.*", "", code, flags=re.DOTALL)

    # 2. Test rapide — si le code parse déjà, juste fixer le nom de classe
    try:
        ast.parse(code)
        return _fix_class_name(code)
    except SyntaxError:
        pass

    # 3. Supprimer TOUS les blocs triple-quoted (terminés et non terminés)
    code = _strip_docstrings(code)

    # 4. Fixer le nom de classe
    code = _fix_class_name(code)

    return code


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
    ct = str(normalized.get("change_type", "")).strip()
    if ct and ct not in ("logic", "params", "both", "accept"):
        if "logic" in ct:
            normalized["change_type"] = "logic"
        elif "params" in ct:
            normalized["change_type"] = "params"
        else:
            normalized["change_type"] = "both"

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


def _is_empty_code(code: str) -> bool:
    """Vérifie si le code généré est vide ou trivial."""
    stripped = code.strip()
    if not stripped:
        return True
    return len(stripped.splitlines()) < MIN_CODE_LINES


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
        slug = re.sub(r"[^a-z0-9]+", "_", objective.lower())[:40].strip("_")
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
    ) -> Dict[str, Any]:
        """Demande au LLM une proposition de stratégie."""
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

        response = self._chat_llm(
            messages=[
                LLMMessage(role="system", content=self._system_prompt_proposal()),
                LLMMessage(role="user", content=prompt),
            ],
            phase="proposal",
            json_mode=True,
            max_tokens=4096,
        )

        return _normalize_proposal_keys(
            _extract_json_from_response(response.content)
        )

    def _ask_code(
        self,
        session: BuilderSession,
        proposal: Dict[str, Any],
    ) -> str:
        """Demande au LLM de générer le code Python complet."""
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
        }

        prompt = render_prompt("strategy_builder_code.jinja2", context)

        response = self._chat_llm(
            messages=[
                LLMMessage(role="system", content=self._system_prompt_code()),
                LLMMessage(role="user", content=prompt),
            ],
            phase="code",
            max_tokens=4096,
        )

        return _extract_python_from_response(response.content)

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
            '  "hypothesis": "brief description",\n'
            '  "change_type": "logic",\n'
            '  "used_indicators": ["rsi", "bollinger"],\n'
            '  "entry_long_logic": "when to BUY",\n'
            '  "entry_short_logic": "when to SELL",\n'
            '  "exit_logic": "when to close",\n'
            '  "risk_management": "SL/TP rules",\n'
            '  "default_params": {},\n'
            '  "parameter_specs": {}\n'
            "}"
        )
        sys_msg = LLMMessage(
            role="system",
            content=(
                "You are a quant trader. "
                "Reply ONLY with valid JSON. No commentary. No thinking."
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
            "        return {}\n\n"
            "    def generate_signals(self, df, indicators, params):\n"
            "        signals = pd.Series(0.0, index=df.index)\n"
            "        # FILL IN: actual trading logic using indicators\n"
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

        original_prop = type(instance).required_indicators

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

            try:
                # ── Phase 1 : Proposition ──
                logger.info("builder_iter_%d_proposal", i)
                ts.proposal_sent(has_previous=last_iteration is not None)
                t0 = time.perf_counter()
                proposal = self._ask_proposal(session, last_iteration)
                dt_proposal = time.perf_counter() - t0

                # Garde : proposition vide → retry avec prompt simplifié
                if _is_empty_proposal(proposal):
                    ts.warning(
                        "Proposition vide reçue — retry avec prompt simplifié"
                    )
                    ts.retry("proposition", 2)
                    logger.warning(
                        "builder_iter_%d_empty_proposal proposal=%s — retrying",
                        i, proposal,
                    )
                    t0 = time.perf_counter()
                    proposal = self._retry_proposal_simple(objective)
                    dt_proposal = time.perf_counter() - t0

                    if _is_empty_proposal(proposal):
                        ts.warning(
                            "Proposition toujours vide après retry — skip itération"
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
                iteration.change_type = proposal.get("change_type", "")
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
                code = self._ask_code(session, proposal)
                dt_code = time.perf_counter() - t0

                # Garde : code vide/trivial → retry avec prompt simplifié
                if _is_empty_code(code):
                    ts.warning(
                        f"Code vide/trivial ({len(code.strip().splitlines())} lignes)"
                        " — retry avec prompt simplifié"
                    )
                    ts.retry("code", 2)
                    logger.warning("builder_iter_%d_empty_code — retrying", i)
                    t0 = time.perf_counter()
                    code = self._retry_code_simple(proposal)
                    dt_code = time.perf_counter() - t0

                    if _is_empty_code(code):
                        ts.warning(
                            "Code toujours vide après retry — skip itération"
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
                    retry_code = self._retry_code_simple(proposal)
                    retry_code = _repair_code(retry_code)
                    is_valid_r, error_msg_r = validate_generated_code(retry_code)
                    if is_valid_r:
                        code = retry_code
                        iteration.code = code
                        is_valid, error_msg = True, ""
                    else:
                        error_msg = f"{error_msg} | retry: {error_msg_r}"

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
                bt_result = self._run_backtest(
                    strategy_cls, data, default_params, initial_capital,
                    symbol=session.symbol,
                    timeframe=session.timeframe,
                    fees_bps=session.fees_bps,
                    slippage_bps=session.slippage_bps,
                )
                iteration.backtest_result = bt_result
                ts.backtest_result(bt_result.metrics)

                # Backtest réussi → reset circuit breaker
                consecutive_failures = 0

                # ── Phase 6 : Mise à jour best ──
                sharpe = bt_result.metrics.get("sharpe_ratio", float("-inf"))
                if sharpe > session.best_sharpe:
                    session.best_sharpe = sharpe
                    session.best_iteration = iteration
                    ts.best_update(sharpe, i)

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
                    iteration.change_type = diag["change_type"]
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
                    session.status = "success"
                    break
                if decision == "stop":
                    session.status = (
                        "success" if session.best_sharpe > 0 else "failed"
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
    symbol: str = "BTCUSDC",
    timeframe: str = "1h",
    available_indicators: Optional[List[str]] = None,
) -> str:
    """Génère un objectif de stratégie aléatoire à partir de templates.

    Combine une famille de stratégie, des indicateurs du registry,
    des conditions d'entrée/sortie et du risk management.

    Returns:
        Objectif structuré en français prêt à être passé au StrategyBuilder.
    """
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
    symbol: str = "BTCUSDC",
    timeframe: str = "1h",
    available_indicators: Optional[List[str]] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Génère un objectif de stratégie via un appel LLM.

    Le LLM produit un objectif créatif et original en respectant
    le format attendu et les indicateurs disponibles.

    Returns:
        Objectif en texte libre généré par le LLM.
    """
    if available_indicators is None:
        available_indicators = list_indicators()

    indicators_list = ", ".join(sorted(available_indicators))

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
            f"Génère un objectif de stratégie de trading pour {symbol} en {timeframe}.\n\n"
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

    # Fallback si le LLM retourne du vide
    if not objective or len(objective) < 20:
        logger.warning("generate_llm_objective: résultat LLM vide, fallback template")
        return generate_random_objective(symbol, timeframe, available_indicators)

    return objective
