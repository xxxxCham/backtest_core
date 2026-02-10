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
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from agents.backtest_executor import BacktestExecutor, BacktestRequest, BacktestResult
from agents.base_agent import AgentRole, MetricsSnapshot
from agents.llm_client import LLMClient, LLMConfig, LLMMessage, create_llm_client
from indicators.registry import get_indicator, list_indicators
from utils.observability import get_obs_logger
from utils.template import render_prompt

logger = get_obs_logger(__name__)

# Dossier racine des sandbox
SANDBOX_ROOT = Path(__file__).resolve().parent.parent / "sandbox_strategies"

# Nom de classe standardisé attendu dans le code généré
GENERATED_CLASS_NAME = "BuilderGeneratedStrategy"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuilderIteration:
    """Résultat d'une itération du builder."""

    iteration: int
    hypothesis: str = ""
    code: str = ""
    backtest_result: Optional[BacktestResult] = None
    error: Optional[str] = None
    analysis: str = ""
    decision: str = ""  # "continue", "accept", "stop"
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
    """Extrait un bloc JSON depuis une réponse LLM (gère ```json ... ```)."""
    import json

    # Chercher bloc ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Essayer le texte brut
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Chercher premier { ... } englobant
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_python_from_response(text: str) -> str:
    """Extrait un bloc Python depuis une réponse LLM."""
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback : le texte entier
    return text.strip()


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
    ):
        if llm_client is not None:
            self.llm = llm_client
        elif llm_config is not None:
            self.llm = create_llm_client(llm_config)
        else:
            self.llm = create_llm_client(LLMConfig.from_env())

        self.available_indicators = list_indicators()

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
        }

        if last_iteration and last_iteration.backtest_result:
            metrics = last_iteration.backtest_result.metrics
            context["last_metrics"] = {
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "win_rate_pct": metrics.get("win_rate_pct", 0),
                "total_trades": metrics.get("total_trades", 0),
                "profit_factor": metrics.get("profit_factor", 0),
            }
            context["last_code"] = last_iteration.code
            context["last_analysis"] = last_iteration.analysis
            context["best_sharpe"] = session.best_sharpe

        if session.iterations:
            context["iteration_history"] = [
                {
                    "iteration": it.iteration,
                    "hypothesis": it.hypothesis,
                    "sharpe": (
                        it.backtest_result.metrics.get("sharpe_ratio", 0)
                        if it.backtest_result else None
                    ),
                    "error": it.error,
                }
                for it in session.iterations[-5:]
            ]

        prompt = render_prompt("strategy_builder_proposal.jinja2", context)

        response = self.llm.chat(
            messages=[
                LLMMessage(role="system", content=self._system_prompt_proposal()),
                LLMMessage(role="user", content=prompt),
            ],
            json_mode=True,
        )

        return _extract_json_from_response(response.content)

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
        }

        prompt = render_prompt("strategy_builder_code.jinja2", context)

        response = self.llm.chat(
            messages=[
                LLMMessage(role="system", content=self._system_prompt_code()),
                LLMMessage(role="user", content=prompt),
            ],
        )

        return _extract_python_from_response(response.content)

    def _ask_analysis(
        self,
        session: BuilderSession,
        iteration: BuilderIteration,
    ) -> tuple[str, str]:
        """Analyse le résultat et décide de continuer ou accepter.

        Returns:
            (analysis_text, decision) où decision ∈ {"continue", "accept", "stop"}
        """
        if not iteration.backtest_result:
            return "Pas de résultat de backtest disponible.", "continue"

        metrics = iteration.backtest_result.metrics
        prompt = f"""## Analyse de l'itération {iteration.iteration}

### Objectif
{session.objective}

### Hypothèse testée
{iteration.hypothesis}

### Résultats du backtest
- Sharpe Ratio: {metrics.get("sharpe_ratio", 0):.3f}
- Return: {metrics.get("total_return_pct", 0):.2f}%
- Max Drawdown: {metrics.get("max_drawdown_pct", 0):.2f}%
- Win Rate: {metrics.get("win_rate_pct", 0):.1f}%
- Trades: {metrics.get("total_trades", 0)}
- Profit Factor: {metrics.get("profit_factor", 0):.2f}

### Meilleur Sharpe jusqu'ici: {session.best_sharpe:.3f}
### Itération: {iteration.iteration}/{session.max_iterations}

Analyse la performance et décide:
- "accept" si Sharpe >= {session.target_sharpe} ET résultats robustes
- "continue" si amélioration possible avec modifications logiques
- "stop" si aucune amélioration ne semble possible

Réponds en JSON: {{"analysis": "...", "decision": "accept|continue|stop", "suggestions": ["..."]}}
"""
        response = self.llm.chat(
            messages=[
                LLMMessage(role="system", content=(
                    "You are an expert quantitative analyst. "
                    "Analyze backtest results and decide next steps. "
                    "Be concise. Respond in JSON."
                )),
                LLMMessage(role="user", content=prompt),
            ],
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
You design strategies using ONLY the available indicators from the registry.
You NEVER create new indicators — only combine existing ones with clever logic.

Your proposals must be JSON with these fields:
{
  "strategy_name": "descriptive_name",
  "hypothesis": "clear hypothesis being tested",
  "used_indicators": ["indicator1", "indicator2"],
  "indicator_params": {"indicator1": {"period": 14}, ...},
  "entry_long_logic": "description of long entry conditions",
  "entry_short_logic": "description of short entry conditions",
  "exit_logic": "description of exit conditions",
  "risk_management": "stop-loss / take-profit logic",
  "default_params": {"param1": value, ...},
  "parameter_specs": {"param1": {"min": 5, "max": 50, "default": 14, "type": "int"}, ...}
}

Focus on signal quality, risk management, and robustness."""

    @staticmethod
    def _system_prompt_code() -> str:
        return f"""You are an expert Python developer specializing in trading systems.
Generate a COMPLETE, WORKING Python strategy class.

CRITICAL RULES:
1. The class MUST be named '{GENERATED_CLASS_NAME}'
2. It MUST inherit from StrategyBase
3. It MUST implement generate_signals(self, df, indicators, params) -> pd.Series
4. Use indicators from the 'indicators' dict (pre-computed by the engine)
5. Return signals as pd.Series: 1.0=LONG, -1.0=SHORT, 0.0=FLAT
6. NEVER use os, subprocess, eval, exec, open, or __import__
7. ONLY import from: numpy, pandas, strategies.base, utils.parameters
8. Include required_indicators, default_params, and parameter_specs properties
9. Handle edge cases: NaN values, insufficient data, division by zero

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
        if cls is None:
            raise AttributeError(
                f"Classe '{GENERATED_CLASS_NAME}' absente du module généré"
            )

        return cls

    # ------------------------------------------------------------------
    # Core: run backtest on generated strategy
    # ------------------------------------------------------------------

    def _run_backtest(
        self,
        strategy_cls: type,
        data: pd.DataFrame,
        params: Dict[str, Any],
        initial_capital: float = 10000.0,
    ) -> BacktestResult:
        """Lance un backtest sur la stratégie générée.

        Utilise BacktestEngine directement avec la classe instanciée.
        """
        from backtest.engine import BacktestEngine
        from metrics_types import normalize_metrics
        from utils.observability import generate_run_id

        run_id = generate_run_id()
        engine = BacktestEngine(initial_capital=initial_capital, run_id=run_id)

        # Instancier la stratégie
        strategy_instance = strategy_cls()

        # Exécuter le backtest via l'engine (mode objet)
        result = engine.run(
            df=data,
            strategy=strategy_instance,
            params=params,
        )

        # Convertir en BacktestResult pour compatibilité
        metrics_pct = normalize_metrics(result.metrics, "pct")

        return BacktestResult(
            request_id=run_id,
            success=True,
            metrics=metrics_pct,
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
    ) -> BuilderSession:
        """
        Lance la boucle complète de construction de stratégie.

        Args:
            objective: Description textuelle de la stratégie souhaitée
            data: DataFrame OHLCV pour backtest
            max_iterations: Nombre max d'itérations
            target_sharpe: Sharpe cible pour acceptation automatique
            initial_capital: Capital initial pour les backtests

        Returns:
            BuilderSession avec l'historique complet et le meilleur résultat
        """
        session_id = self.create_session_id(objective)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        session = BuilderSession(
            session_id=session_id,
            objective=objective,
            session_dir=session_dir,
            available_indicators=self.available_indicators,
            max_iterations=max_iterations,
            target_sharpe=target_sharpe,
        )

        logger.info(
            "strategy_builder_start session=%s objective='%s' indicators=%d",
            session_id, objective, len(self.available_indicators),
        )

        last_iteration: Optional[BuilderIteration] = None

        for i in range(1, max_iterations + 1):
            iteration = BuilderIteration(iteration=i)

            try:
                # ── Phase 1 : Proposition ──
                logger.info("builder_iter_%d_proposal", i)
                proposal = self._ask_proposal(session, last_iteration)
                iteration.hypothesis = proposal.get(
                    "hypothesis", f"Itération {i}"
                )

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
                    # Filtrer les indicateurs inconnus
                    proposal["used_indicators"] = [
                        ind for ind in used if ind.lower() in
                        (x.lower() for x in self.available_indicators)
                    ]

                # ── Phase 2 : Génération de code ──
                logger.info("builder_iter_%d_codegen", i)
                code = self._ask_code(session, proposal)
                iteration.code = code

                # ── Phase 3 : Validation syntaxe + sécurité ──
                is_valid, error_msg = validate_generated_code(code)
                if not is_valid:
                    iteration.error = f"Validation échouée: {error_msg}"
                    logger.warning("builder_iter_%d_invalid code=%s", i, error_msg)
                    session.iterations.append(iteration)
                    last_iteration = iteration
                    continue

                # ── Phase 4 : Chargement dynamique ──
                logger.info("builder_iter_%d_load", i)
                strategy_cls = self._save_and_load(session, code, i)

                # ── Phase 5 : Backtest ──
                logger.info("builder_iter_%d_backtest", i)
                default_params = proposal.get("default_params", {})
                bt_result = self._run_backtest(
                    strategy_cls, data, default_params, initial_capital
                )
                iteration.backtest_result = bt_result

                # ── Phase 6 : Mise à jour best ──
                sharpe = bt_result.metrics.get("sharpe_ratio", float("-inf"))
                if sharpe > session.best_sharpe:
                    session.best_sharpe = sharpe
                    session.best_iteration = iteration

                # ── Phase 7 : Analyse + décision ──
                logger.info("builder_iter_%d_analysis", i)
                analysis, decision = self._ask_analysis(session, iteration)
                iteration.analysis = analysis
                iteration.decision = decision

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
                logger.error(
                    "builder_iter_%d_error error=%s\n%s",
                    i, e, traceback.format_exc(),
                )
                session.iterations.append(iteration)
                last_iteration = iteration

        else:
            session.status = "max_iterations"

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
        import json

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
                    "error": it.error,
                    "decision": it.decision,
                    "sharpe": (
                        it.backtest_result.metrics.get("sharpe_ratio", 0)
                        if it.backtest_result else None
                    ),
                    "trades": (
                        it.backtest_result.metrics.get("total_trades", 0)
                        if it.backtest_result else None
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
