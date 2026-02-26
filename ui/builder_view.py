"""
Module-ID: ui.builder_view

Purpose: Interface Streamlit pour le Strategy Builder — création de stratégies par IA.

Role in pipeline: UI / interaction utilisateur

Key components: render_builder_view, render_iteration_card, render_session_summary

Inputs: SidebarState (builder_objective, builder_model, etc.), DataFrame OHLCV

Outputs: Affichage interactif des itérations et résultats du builder

Dependencies: agents.strategy_builder, ui.helpers, ui.context

Conventions: Streamlit components, pas de logique de trading

Read-if: Modification de l'interface Strategy Builder

Skip-if: Logique backend du builder (voir agents/strategy_builder.py)
"""

from __future__ import annotations

import csv
import io
import logging
import random
import time
import traceback
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

import httpx

# ── Logging de diagnostic (TEMPORAIRE) ──
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Force INFO level
# Ajouter un handler console si absent
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s | %(message)s', datefmt='%H:%M:%S')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

from agents.llm_client import LLMConfig, LLMProvider, create_llm_client
from agents.ollama_manager import ensure_ollama_running
from agents.strategy_builder import (
    StrategyBuilder,
    generate_llm_objective,
    generate_parametric_catalog,
    generate_random_objective,
    get_catalog_coverage,
    get_next_catalog_objective,
    get_next_parametric_objective,
    mark_catalog_objective_explored,
    recommend_market_context,
    sanitize_objective_text,
)
from agents.thought_stream import STREAM_FILE
from ui.helpers import show_status


def render_iteration_card(
    iteration: Any,
    *,
    expanded: bool = False,
) -> None:
    """Affiche une carte enrichie pour une itération du builder."""
    it_num = iteration.iteration
    decision = getattr(iteration, "decision", "")
    error = getattr(iteration, "error", None)
    diag = getattr(iteration, "diagnostic_detail", {}) or {}
    phase_feedback = getattr(iteration, "phase_feedback", {}) or {}

    # Icône selon résultat
    if error:
        icon = "❌"
        label = f"Itération {it_num} — Erreur"
    elif decision == "accept":
        icon = "✅"
        label = f"Itération {it_num} — Acceptée"
    elif decision == "stop":
        icon = "🛑"
        label = f"Itération {it_num} — Arrêt"
    else:
        icon = "🔄"
        label = f"Itération {it_num} — Continue"

    with st.expander(f"{icon} {label}", expanded=expanded):
        # Diagnostic badge + type de modification
        diag_cat = getattr(iteration, "diagnostic_category", "")
        change_type = getattr(iteration, "change_type", "")
        severity = diag.get("severity", "")

        sev_icons = {
            "critical": "🔴", "warning": "🟡",
            "info": "🔵", "success": "🟢",
        }
        type_labels = {
            "logic": "🔀 Logique",
            "params": "🎛️ Paramètres",
            "both": "🔀🎛️ Logique + Params",
            "accept": "✅ Acceptation",
        }
        sev_icon = sev_icons.get(severity, "⚪")
        type_lbl = type_labels.get(change_type, "")
        cat_lbl = diag_cat.replace("_", " ").title() if diag_cat else ""

        if cat_lbl:
            st.caption(f"{sev_icon} **{cat_lbl}** — {type_lbl}")
        elif type_lbl:
            st.caption(type_lbl)

        # Hypothèse
        hypothesis = getattr(iteration, "hypothesis", "")
        if hypothesis:
            st.markdown(f"**💡 Hypothèse:** {hypothesis}")

        # Erreur
        if error:
            st.error(f"Erreur: {error}")

        # Résultats backtest
        bt = getattr(iteration, "backtest_result", None)
        if bt and hasattr(bt, "metrics"):
            metrics = bt.metrics
            score_info = diag.get("continuous_score")
            if score_info is None:
                score_info = (
                    (phase_feedback.get("scoring", {}) if phase_feedback else {}).get(
                        "continuous_score"
                    )
                )

            # Ligne 1: métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
            with col2:
                st.metric("Return", f"{metrics.get('total_return_pct', 0):+.2f}%")
            with col3:
                st.metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            with col4:
                st.metric("Trades", str(metrics.get("total_trades", 0)))
            with col5:
                if score_info is None:
                    st.metric("Score", "n/a")
                else:
                    st.metric("Score", f"{float(score_info):.2f}")

            # Ligne 2: métriques secondaires
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
            with c2:
                st.metric("PF", f"{metrics.get('profit_factor', 0):.2f}")
            with c3:
                st.metric("Sortino", f"{metrics.get('sortino_ratio', 0):.3f}")
            with c4:
                st.metric("Expectancy", f"{metrics.get('expectancy', 0):.2f}")

            # Bulletin de notes compact
            if diag and "score_card" in diag:
                sc = diag["score_card"]
                grades = " | ".join(
                    f"**{dim.replace('_', ' ').title()}**: {info['grade']}"
                    for dim, info in sc.items()
                )
                st.caption(f"📊 {grades}")

        # Actions recommandées (expandable)
        if diag and diag.get("actions"):
            with st.expander("💡 Actions recommandées", expanded=False):
                for act in diag["actions"]:
                    st.markdown(f"- {act}")
                for dont in diag.get("donts", []):
                    st.markdown(f"- ⚠️ {dont}")

        # Analyse LLM
        analysis = getattr(iteration, "analysis", "")
        if analysis:
            st.markdown(f"**📊 Analyse:** {analysis[:600]}")

        # Feedback structuré de phase (diagnostic d'orchestration)
        if phase_feedback:
            with st.expander("🧭 Feedback d'orchestration", expanded=False):
                pfb = phase_feedback.get("proposal", {}) or {}
                cfb = phase_feedback.get("code", {}) or {}
                bfb = phase_feedback.get("backtest", {}) or {}
                dfb = phase_feedback.get("decision", {}) or {}

                if pfb:
                    st.markdown("**Proposal phase**")
                    st.markdown(
                        f"- kind initial: `{pfb.get('initial_kind', '?')}` | "
                        f"realign attempts: `{pfb.get('realign_attempts', 0)}` | "
                        f"realign success: `{pfb.get('realign_success', False)}` | "
                        f"valid: `{pfb.get('final_valid', False)}`"
                    )
                    issues = pfb.get("issues") or pfb.get("issues_after_retry")
                    if issues:
                        st.markdown(f"- issues: `{issues}`")
                    if pfb.get("fallback_retry_used"):
                        st.markdown("- fallback retry utilisé")
                    ct_override = pfb.get("change_type_overridden")
                    if isinstance(ct_override, dict):
                        st.markdown(
                            "- change_type overridé: "
                            f"`{ct_override.get('from', '?')}` → "
                            f"`{ct_override.get('to', '?')}` "
                            f"(`{ct_override.get('reason', '?')}`)"
                        )

                if cfb:
                    st.markdown("**Code phase**")
                    st.markdown(
                        f"- kind initial: `{cfb.get('initial_kind', '?')}` | "
                        f"realign attempts: `{cfb.get('realign_attempts', 0)}` | "
                        f"realign success: `{cfb.get('realign_success', False)}` | "
                        f"valid: `{cfb.get('final_valid', False)}`"
                    )
                    if cfb.get("source"):
                        st.markdown(f"- source: `{cfb.get('source')}`")
                    if cfb.get("fallback_retry_used"):
                        st.markdown("- fallback retry utilisé")
                    if cfb.get("fallback_deterministic_used"):
                        st.markdown("- fallback déterministe appliqué")

                if bfb:
                    st.markdown("**Backtest phase**")
                    if bfb.get("runtime_error"):
                        st.markdown(f"- runtime_error: `{bfb.get('runtime_error')}`")
                    if bfb.get("runtime_traceback_tail"):
                        st.markdown("- traceback (tail):")
                        st.code(str(bfb.get("runtime_traceback_tail")), language="text")
                    if bfb.get("runtime_fix_applied"):
                        st.markdown("- runtime_fix appliqué avec succès")
                    if bfb.get("runtime_fix_fallback_deterministic_used"):
                        st.markdown("- runtime_fix: fallback déterministe appliqué")
                    if bfb.get("runtime_fix_retry_error"):
                        st.markdown(
                            "- runtime_fix second échec: "
                            f"`{bfb.get('runtime_fix_retry_error')}`"
                        )
                    if bfb.get("runtime_fix_validation_error"):
                        st.markdown(
                            "- runtime_fix rejeté à la validation: "
                            f"`{bfb.get('runtime_fix_validation_error')}`"
                        )

                sfb = phase_feedback.get("scoring", {}) or {}
                if sfb:
                    st.markdown("**Scoring continu**")
                    score = sfb.get("continuous_score")
                    dd_excess = sfb.get("drawdown_excess_pct")
                    if score is not None:
                        st.markdown(f"- score: `{float(score):.2f}`")
                    if dd_excess is not None:
                        st.markdown(f"- drawdown excess: `{float(dd_excess):.2f}%`")

                if dfb:
                    st.markdown("**Decision policy**")
                    if dfb.get("stop_overridden"):
                        st.markdown("- stop overridé en `continue`")
                    if dfb.get("accept_overridden"):
                        st.markdown("- accept overridé en `continue`")

        # Code (collapsible)
        code = getattr(iteration, "code", "")
        if code:
            with st.expander("📝 Code généré", expanded=False):
                st.code(code, language="python")


def render_session_summary(session: Any) -> None:
    """Affiche le résumé final de la session builder."""
    status = getattr(session, "status", "unknown")
    best_sharpe = getattr(session, "best_sharpe", float("-inf"))
    best_score = getattr(session, "best_score", float("-inf"))
    n_iters = len(getattr(session, "iterations", []))

    # Statut global
    status_map = {
        "success": ("🏆", "Stratégie acceptée"),
        "max_iterations": ("⏱️", "Itérations max atteintes"),
        "failed": ("❌", "Échec - aucune stratégie viable"),
        "running": ("🔄", "En cours..."),
    }
    icon, label = status_map.get(status, ("❓", status))

    st.markdown(f"### {icon} {label}")
    score_txt = "n/a" if best_score == float("-inf") else f"{best_score:.2f}"
    st.markdown(
        f"**Itérations:** {n_iters} | **Meilleur Sharpe:** {best_sharpe:.3f} | "
        f"**Meilleur Score:** {score_txt}"
    )

    # Synthèse orchestration: réalignements et overrides
    total_realign = 0
    stop_overrides = 0
    accept_overrides = 0
    for it in getattr(session, "iterations", []):
        fb = getattr(it, "phase_feedback", {}) or {}
        total_realign += int((fb.get("proposal", {}) or {}).get("realign_attempts", 0))
        total_realign += int((fb.get("code", {}) or {}).get("realign_attempts", 0))
        dec_fb = fb.get("decision", {}) or {}
        stop_overrides += 1 if dec_fb.get("stop_overridden") else 0
        accept_overrides += 1 if dec_fb.get("accept_overridden") else 0

    if total_realign or stop_overrides or accept_overrides:
        st.caption(
            "Orchestration: "
            f"realignements={total_realign} | "
            f"stop_overrides={stop_overrides} | "
            f"accept_overrides={accept_overrides}"
        )

    # Détails du meilleur résultat
    best = getattr(session, "best_iteration", None)
    if best and hasattr(best, "backtest_result") and best.backtest_result:
        metrics = best.backtest_result.metrics
        st.markdown("#### 🥇 Meilleur résultat")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
        with col2:
            st.metric("Return", f"{metrics.get('total_return_pct', 0):+.2f}%")
        with col3:
            st.metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
        with col5:
            st.metric("PF", f"{metrics.get('profit_factor', 0):.2f}")

        # Hypothèse gagnante
        hyp = getattr(best, "hypothesis", "")
        if hyp:
            st.info(f"**Hypothèse gagnante:** {hyp}")

        # Code final
        code = getattr(best, "code", "")
        if code:
            with st.expander("📝 Code de la stratégie gagnante", expanded=True):
                st.code(code, language="python")

    # Chemin sandbox
    session_dir = getattr(session, "session_dir", None)
    if session_dir:
        st.caption(f"📁 Fichiers de session: `{session_dir}`")


# ---------------------------------------------------------------------------
# Warmup Ollama — préchargement du modèle en VRAM
# ---------------------------------------------------------------------------

def _is_local_ollama_host(ollama_host: str) -> bool:
    """Retourne True si l'host Ollama est local (localhost/127.0.0.1/0.0.0.0)."""
    try:
        parsed = urlparse(ollama_host)
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "0.0.0.0"}
    except Exception:
        return False


def _model_matches(model_name: str, requested_model: str) -> bool:
    """Matching tolérant entre nom de modèle demandé et liste Ollama."""
    model_name_l = model_name.lower()
    requested_l = requested_model.lower()

    if model_name_l == requested_l:
        return True
    if model_name_l.startswith(requested_l):
        return True
    if requested_l.startswith(model_name_l):
        return True

    req_base = requested_l.split(":", 1)[0]
    model_base = model_name_l.split(":", 1)[0]
    if model_base == req_base:
        return True

    req_compact = re.sub(r"[^a-z0-9]", "", req_base)
    model_compact = re.sub(r"[^a-z0-9]", "", model_base)
    return bool(req_compact) and req_compact == model_compact


def _extract_model_size_b(model_name: str) -> float:
    """Extrait la taille d'un modèle en milliards de paramètres (ex: 32b)."""
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if not m:
        return -1.0
    try:
        return float(m.group(1))
    except Exception:
        return -1.0


def _resolve_requested_model(requested_model: str, installed_models: List[str]) -> tuple[str, str]:
    """Résout un modèle demandé vers un modèle installé (avec fallback robuste)."""
    if not installed_models:
        return requested_model, "Aucun modèle installé détecté."

    # 1) Match direct/normalisé
    for name in installed_models:
        if _model_matches(name, requested_model):
            return name, ""

    # 2) Match par taille (ex: 32b) pour garder un profil proche
    requested_size = _extract_model_size_b(requested_model)
    if requested_size > 0:
        same_size = [
            n for n in installed_models
            if _extract_model_size_b(n) == requested_size
        ]
        if same_size:
            return (
                same_size[0],
                (
                    f"Modèle `{requested_model}` absent. "
                    f"Fallback auto vers `{same_size[0]}` (taille {requested_size:.0f}B)."
                ),
            )

    # 3) Priorité à quelques modèles robustes si présents
    preferred = [
        "deepseek-r1-32b-local:latest",
        "deepseek-r1:32b",
        "qwq-32b-local:latest",
        "qwq:32b",
        "qwen3-48b-savant:latest",
    ]
    installed_lower = {m.lower(): m for m in installed_models}
    for pref in preferred:
        if pref.lower() in installed_lower:
            chosen = installed_lower[pref.lower()]
            return (
                chosen,
                f"Modèle `{requested_model}` absent. Fallback auto vers `{chosen}`.",
            )

    # 4) Dernier recours: premier installé
    return (
        installed_models[0],
        (
            f"Modèle `{requested_model}` absent. "
            f"Fallback auto vers `{installed_models[0]}`."
        ),
    )


def _warmup_ollama_model(
    *,
    model: str,
    ollama_host: str,
    keep_alive_minutes: int,
    timeout: float = 120.0,
) -> bool:
    """Précharge un modèle Ollama en VRAM via un prompt court.

    Envoie un prompt minimal à /api/generate pour forcer le chargement
    du modèle en mémoire GPU avant les vrais appels LLM.

    Returns:
        True si le modèle a été chargé avec succès.
    """
    try:
        keep_alive = f"{max(1, int(keep_alive_minutes))}m"
        resp = httpx.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": "Ready.",
                "keep_alive": keep_alive,
                "stream": False,
            },
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _unload_ollama_model(*, model: str, ollama_host: str, timeout: float = 20.0) -> bool:
    """Décharge un modèle Ollama de la mémoire."""
    try:
        resp = httpx.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "keep_alive": 0,
                "stream": False,
            },
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _prepare_builder_llm(
    *,
    model: str,
    ollama_host: str,
    preload_model: bool,
    keep_alive_minutes: int,
    auto_start_ollama: bool,
) -> tuple[bool, str, str]:
    """Prépare Ollama + modèle pour le Strategy Builder (check + warmup)."""
    if auto_start_ollama and _is_local_ollama_host(ollama_host):
        ok, msg = ensure_ollama_running()
        if not ok:
            return False, msg, model

    try:
        tags = httpx.get(f"{ollama_host}/api/tags", timeout=8.0)
    except Exception as exc:
        return False, f"Ollama inaccessible ({ollama_host}): {exc}", model

    if tags.status_code != 200:
        return (
            False,
            f"Ollama indisponible sur {ollama_host} (status={tags.status_code})",
            model,
        )

    models = [m.get("name", "") for m in tags.json().get("models", []) if m.get("name")]
    resolved_model, resolve_note = _resolve_requested_model(model, models)

    if not preload_model:
        msg = f"Ollama OK ({ollama_host}) — warmup désactivé."
        if resolve_note:
            msg = f"{resolve_note} {msg}"
        return True, msg, resolved_model

    if _warmup_ollama_model(
        model=resolved_model,
        ollama_host=ollama_host,
        keep_alive_minutes=keep_alive_minutes,
    ):
        msg = (
            f"Modèle `{resolved_model}` chargé en mémoire "
            f"({keep_alive_minutes} min keep-alive)."
        )
        if resolve_note:
            msg = f"{resolve_note} {msg}"
        return (
            True,
            msg,
            resolved_model,
        )

    return (
        False,
        f"Impossible de précharger `{resolved_model}` sur {ollama_host}.",
        resolved_model,
    )


def _dedupe_keep_order(values: List[str], *, upper: bool = False) -> List[str]:
    """Supprime les doublons en conservant l'ordre d'apparition."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if upper:
            value = value.upper()
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _is_builder_supported_timeframe(raw_tf: str) -> bool:
    """Filtre des TF supportés en mode Builder (rejette les TF mensuels)."""
    tf = str(raw_tf or "").strip()
    if not tf:
        return False
    m = re.fullmatch(r"(\d+)([mhdwM])", tf)
    if not m:
        return False
    amount = int(m.group(1))
    unit = m.group(2)
    if amount <= 0:
        return False
    if unit == "M":
        return False
    return True


def _sanitize_builder_timeframes(
    timeframes: List[str],
    *,
    fallback: str = "1h",
) -> List[str]:
    """Normalise la liste de TF pour le Builder en retirant les entrées non supportées."""
    cleaned = [
        str(tf or "").strip()
        for tf in (timeframes or [])
        if _is_builder_supported_timeframe(str(tf or "").strip())
    ]
    cleaned = _dedupe_keep_order(cleaned, upper=False)
    if cleaned:
        return cleaned
    return [fallback] if _is_builder_supported_timeframe(fallback) else ["1h"]


def _get_builder_compatible_indicators(df: Any) -> List[str]:
    """Retourne les indicateurs calculables avec les colonnes réellement présentes."""
    try:
        from indicators.registry import get_indicator, list_indicators
    except Exception:
        return ["ema", "rsi", "atr"]

    all_indicators = [str(x or "").strip().lower() for x in list_indicators()]
    all_indicators = [x for x in all_indicators if x]
    if not all_indicators:
        return ["ema", "rsi", "atr"]

    # Si df indisponible, partir sur OHLCV standard.
    raw_cols = getattr(df, "columns", None)
    if raw_cols is None:
        raw_cols = []
    df_cols = {
        str(col or "").strip().lower()
        for col in list(raw_cols)
        if str(col or "").strip()
    }
    if not df_cols:
        df_cols = {"open", "high", "low", "close", "volume"}

    # Indicateurs à colonne externe optionnelle/non-OHLCV.
    # fear_greed exige une colonne dédiée "fear_greed" et échoue sinon.
    custom_column_requirements = {
        "fear_greed": "fear_greed",
    }

    compatible: List[str] = []
    for name in all_indicators:
        info = get_indicator(name)
        required = tuple(getattr(info, "required_columns", ()) or ())
        if any(str(col).lower() not in df_cols for col in required):
            continue
        extra_col = custom_column_requirements.get(name)
        if extra_col and extra_col not in df_cols:
            continue
        compatible.append(name)

    if compatible:
        return compatible
    return ["ema", "rsi", "atr"]


def _stable_random_pick(session_key: str, candidates: List[str], fallback: str) -> str:
    """Retourne un choix aléatoire stable sur la session Streamlit."""
    normalized = [str(v or "").strip() for v in candidates if str(v or "").strip()]
    if not normalized:
        return fallback

    cached = str(st.session_state.get(session_key, "") or "").strip()
    if cached in normalized:
        return cached

    picked = random.choice(normalized)
    st.session_state[session_key] = picked
    return picked


def _pick_non_recent_market(
    symbols: List[str],
    timeframes: List[str],
    recent_markets: List[Tuple[str, str]],
) -> Tuple[str, str]:
    """Choisit un couple marché de fallback en évitant d'abord les plus récents."""
    clean_symbols = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    clean_tfs = [str(tf or "").strip() for tf in timeframes if str(tf or "").strip()]
    if not clean_symbols:
        clean_symbols = ["BTCUSDC"]
    if not clean_tfs:
        clean_tfs = ["1h"]

    all_pairs = [(s, tf) for s in clean_symbols for tf in clean_tfs]
    if len(all_pairs) == 1:
        return all_pairs[0]

    recent_set = {
        (str(s or "").strip().upper(), str(tf or "").strip())
        for s, tf in (recent_markets or [])
        if str(s or "").strip() and str(tf or "").strip()
    }
    candidate_pairs = [pair for pair in all_pairs if pair not in recent_set]
    pool = candidate_pairs if candidate_pairs else all_pairs

    # Diversifier explicitement les TF pour éviter les longs blocs mono-timeframe.
    tf_usage = st.session_state.setdefault("_builder_tf_usage", {})
    for tf in clean_tfs:
        tf_usage.setdefault(tf, 0)
    min_tf_usage = min(tf_usage.get(tf, 0) for _, tf in pool)
    tf_pool = [
        tf for tf in clean_tfs
        if tf_usage.get(tf, 0) == min_tf_usage and any(pair_tf == tf for _, pair_tf in pool)
    ]
    chosen_tf = random.choice(tf_pool) if tf_pool else random.choice(clean_tfs)

    tf_pairs = [pair for pair in pool if pair[1] == chosen_tf]
    chosen_pair = random.choice(tf_pairs) if tf_pairs else random.choice(pool)
    tf_usage[chosen_pair[1]] = int(tf_usage.get(chosen_pair[1], 0)) + 1
    return chosen_pair


def _builder_market_candidates(
    state: Any,
    *,
    current_symbol: str,
    current_timeframe: str,
) -> Tuple[List[str], List[str]]:
    """Construit l'univers symbol/timeframe proposé au LLM."""
    auto_market_pick = bool(getattr(state, "builder_auto_market_pick", False))

    # En mode auto market pick, ignorer les valeurs bootstrap cachées (1 seul token/TF)
    # et ne conserver que les sélections explicitement faites dans les multiselects UI.
    if auto_market_pick:
        selected_symbols = [
            str(s or "").strip().upper()
            for s in (st.session_state.get("symbols_select", []) or [])
            if str(s or "").strip()
        ]
        selected_timeframes = [
            str(tf or "").strip()
            for tf in (st.session_state.get("timeframes_select", []) or [])
            if str(tf or "").strip()
        ]
    else:
        selected_symbols = list(getattr(state, "symbols", []) or [])
        selected_timeframes = list(getattr(state, "timeframes", []) or [])

    available_symbols = list(getattr(state, "available_tokens", []) or [])
    available_timeframes = list(getattr(state, "available_timeframes", []) or [])

    # Priorité: sélection utilisateur -> marché courant -> univers complet.
    symbols = _dedupe_keep_order(
        [*selected_symbols, current_symbol, *available_symbols],
        upper=True,
    )
    if selected_timeframes:
        # Source de vérité: quand l'utilisateur a sélectionné des TF,
        # ne pas réinjecter un current_timeframe externe potentiellement obsolète.
        tf_candidates = [str(tf or "").strip() for tf in selected_timeframes if str(tf or "").strip()]
        if current_timeframe and current_timeframe in tf_candidates:
            tf_candidates = [current_timeframe, *[tf for tf in tf_candidates if tf != current_timeframe]]
    else:
        tf_candidates = [current_timeframe, *available_timeframes]

    timeframes = _dedupe_keep_order(tf_candidates, upper=False)
    timeframes = _sanitize_builder_timeframes(timeframes, fallback=current_timeframe or "1h")

    # Anti-biais: on garde des ancres utiles (marché courant + sélection utilisateur),
    # puis on complète aléatoirement pour éviter l'effet "premiers éléments de liste".
    symbol_anchors = _dedupe_keep_order(
        [current_symbol, *selected_symbols],
        upper=True,
    )
    symbol_anchors = [s for s in symbol_anchors if s in symbols][:6]
    symbol_pool = [s for s in symbols if s not in symbol_anchors]
    random.shuffle(symbol_pool)
    symbols = [*symbol_anchors, *symbol_pool]

    timeframe_anchors = _dedupe_keep_order(
        [current_timeframe, *selected_timeframes],
        upper=False,
    )
    timeframe_anchors = [
        tf for tf in timeframe_anchors
        if tf in timeframes and _is_builder_supported_timeframe(tf)
    ][:4]
    timeframe_pool = [tf for tf in timeframes if tf not in timeframe_anchors]
    random.shuffle(timeframe_pool)
    timeframes = [*timeframe_anchors, *timeframe_pool]

    # Limiter la taille du prompt LLM.
    return symbols[:24], timeframes[:12]


def _state_date_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return None
    text = str(value).strip()
    return text or None


def _load_builder_market_data(
    *,
    state: Any,
    symbol: str,
    timeframe: str,
    fallback_df: Any,
) -> tuple[Any, str | None, str]:
    """Charge les données pour un marché recommandé avec cache simple en session."""
    if not symbol or not timeframe:
        return fallback_df, "Marché invalide", "fallback_current_df"

    base_symbol = str(getattr(state, "symbol", "") or "").upper()
    base_timeframe = str(getattr(state, "timeframe", "") or "")
    if symbol.upper() == base_symbol and timeframe == base_timeframe:
        return fallback_df, None, "current_df"

    use_date_filter = bool(getattr(state, "use_date_filter", False))
    start = _state_date_to_iso(getattr(state, "start_date", None)) if use_date_filter else None
    end = _state_date_to_iso(getattr(state, "end_date", None)) if use_date_filter else None
    cache_key = f"{symbol}|{timeframe}|{start}|{end}"

    cache = st.session_state.setdefault("_builder_market_df_cache", {})
    if cache_key in cache:
        return cache[cache_key], None, "cache"

    try:
        from data.loader import load_ohlcv

        market_df = load_ohlcv(symbol, timeframe, start=start, end=end)
        cache[cache_key] = market_df
        if len(cache) > 6:
            first_key = next(iter(cache))
            if first_key != cache_key:
                cache.pop(first_key, None)
        return market_df, None, "loaded"
    except Exception as exc:
        return fallback_df, str(exc), "fallback_current_df"


def _pick_market_for_objective(
    *,
    state: Any,
    objective: str,
    llm_client: Any,
    default_symbol: str,
    default_timeframe: str,
    fallback_df: Any,
    recent_markets: list[tuple[str, str]] | None = None,
) -> tuple[str, str, Any, Dict[str, Any]]:
    """Demande au LLM le meilleur marché pour l'objectif puis charge les données."""
    symbols, timeframes = _builder_market_candidates(
        state,
        current_symbol=default_symbol,
        current_timeframe=default_timeframe,
    )
    pick = recommend_market_context(
        llm_client,
        objective=objective,
        candidate_symbols=symbols,
        candidate_timeframes=timeframes,
        default_symbol=default_symbol,
        default_timeframe=default_timeframe,
        recent_markets=recent_markets,
    )

    run_symbol = str(pick.get("symbol", default_symbol) or default_symbol).upper()
    run_timeframe = str(pick.get("timeframe", default_timeframe) or default_timeframe)
    run_df, load_error, data_source = _load_builder_market_data(
        state=state,
        symbol=run_symbol,
        timeframe=run_timeframe,
        fallback_df=fallback_df,
    )

    if load_error:
        run_symbol = default_symbol
        run_timeframe = default_timeframe

    pick["data_source"] = data_source
    pick["load_error"] = load_error
    pick["candidate_symbols"] = symbols
    pick["candidate_timeframes"] = timeframes
    return run_symbol, run_timeframe, run_df, pick


# ---------------------------------------------------------------------------
# Run unique d'une session builder (factorisé pour réutilisation)
# ---------------------------------------------------------------------------

def _run_single_builder_session(
    *,
    objective: str,
    model: str,
    ollama_host: str,
    preload_model: bool,
    keep_alive_minutes: int,
    unload_after_run: bool,
    auto_start_ollama: bool,
    max_iterations: int,
    target_sharpe: float,
    capital: float,
    symbol: str,
    timeframe: str,
    fees_bps: float,
    slippage_bps: float,
    df: Any,
    session_label: str = "",
    skip_llm_prepare: bool = False,
    show_config_caption: bool = True,
) -> Any:
    """Exécute une session builder unique et affiche les résultats.

    Args:
        skip_llm_prepare: Si True, ne pas refaire les checks/warmup (déjà fait en amont).

    Returns:
        BuilderSession ou None en cas d'erreur/interruption.
    """
    if session_label:
        st.markdown(f"### {session_label}")
    st.markdown(f"**Objectif:** {objective}")
    if show_config_caption:
        st.caption(
            f"Modèle: `{model}` | Max itérations: {max_iterations} | "
            f"Sharpe cible: {target_sharpe} | Capital: ${capital:,.0f} | "
            f"Marché: {symbol} {timeframe} | "
            f"Données: {len(df):,} barres | "
            f"Frais: {fees_bps}bps + {slippage_bps}bps slip"
        )

    # Préchargement du modèle Ollama en VRAM
    runtime_model = model
    if not skip_llm_prepare:
        with st.spinner(f"⏳ Préparation LLM `{model}` ({ollama_host})…"):
            ok, msg, resolved_model = _prepare_builder_llm(
                model=model,
                ollama_host=ollama_host,
                preload_model=preload_model,
                keep_alive_minutes=keep_alive_minutes,
                auto_start_ollama=auto_start_ollama,
            )
            if ok:
                st.caption(f"✅ {msg}")
                runtime_model = resolved_model
                if runtime_model != model:
                    st.info(
                        f"ℹ️ Modèle effectif Builder: `{runtime_model}` "
                        f"(fallback depuis `{model}`)"
                    )
            else:
                st.error(f"❌ {msg}")
                return None

    st.session_state["builder_model_effective"] = runtime_model

    # Zone de streaming LLM
    stream_placeholder = st.empty()
    _stream_state: dict = {"text": "", "phase": "", "active": False}

    _PHASE_LABELS = {
        "proposal": ("💡", "Proposition", "json"),
        "code": ("🔧", "Génération de code", "python"),
        "analysis": ("🤔", "Analyse", "json"),
        "retry_proposal": ("🔁", "Retry proposition", "json"),
        "retry_code": ("🔁", "Retry code", "python"),
        "objective_gen": ("🎯", "Génération d'objectif", "text"),
    }

    def _on_llm_stream(phase: str, chunk: str) -> None:
        if phase != _stream_state["phase"]:
            _stream_state["text"] = ""
            _stream_state["phase"] = phase
        _stream_state["text"] += chunk
        _stream_state["active"] = True

        icon, label, lang = _PHASE_LABELS.get(phase, ("🧠", phase, "text"))
        text = _stream_state["text"]

        try:
            with stream_placeholder.container():
                st.caption(f"{icon} **{label}** — streaming en cours…")
                display = text[-4000:] if len(text) > 4000 else text
                if len(text) > 4000:
                    display = "…(tronqué)…\n" + display
                st.code(display, language=lang)
        except Exception:
            pass

    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=runtime_model,
        ollama_host=ollama_host,
    )

    builder = StrategyBuilder(
        llm_config=llm_config,
        stream_callback=_on_llm_stream,
    )
    compatible_indicators = _get_builder_compatible_indicators(df)
    if compatible_indicators:
        builder.available_indicators = compatible_indicators

    progress_bar = st.progress(0.0, text="Initialisation...")
    iterations_container = st.container()
    summary_placeholder = st.empty()

    start_time = time.perf_counter()

    with st.status("🏗️ Construction en cours...", expanded=True) as live_status:
        try:
            session = builder.run(
                objective=objective,
                data=df,
                max_iterations=max_iterations,
                target_sharpe=target_sharpe,
                initial_capital=capital,
                symbol=symbol,
                timeframe=timeframe,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )
        except KeyboardInterrupt:
            live_status.update(label="⚠️ Interrompu", state="error")
            st.warning("Construction interrompue par l'utilisateur.")
            return None
        except Exception as exc:
            live_status.update(label=f"❌ Erreur: {exc}", state="error")
            show_status("error", f"Erreur Strategy Builder: {exc}")
            st.code(traceback.format_exc())
            return None

        elapsed = time.perf_counter() - start_time
        live_status.update(
            label=f"✅ Terminé en {elapsed:.1f}s — {len(session.iterations)} itérations",
            state="complete",
        )

    # Nettoyage
    try:
        stream_placeholder.empty()
    except Exception:
        pass

    progress_bar.progress(1.0, text="Terminé")

    with iterations_container:
        st.markdown("### 📋 Historique des itérations")
        for it in session.iterations:
            is_last = (it == session.iterations[-1])
            render_iteration_card(it, expanded=is_last)

    with summary_placeholder.container():
        st.markdown("---")
        render_session_summary(session)

    if unload_after_run:
        with st.spinner(f"💾 Déchargement du modèle `{runtime_model}`…"):
            if _unload_ollama_model(model=runtime_model, ollama_host=ollama_host):
                st.caption(f"✅ Modèle `{runtime_model}` déchargé")
            else:
                st.warning(f"⚠️ Impossible de décharger `{runtime_model}`")

    return session


# ---------------------------------------------------------------------------
# Tableau récapitulatif des sessions autonomes
# ---------------------------------------------------------------------------

def _render_autonomous_recap(history: List[Dict[str, Any]]) -> None:
    """Affiche un tableau récapitulatif de toutes les sessions autonomes."""
    if not history:
        return

    def _fmt_float(value: Any, pattern: str, default: str = "n/a") -> str:
        try:
            if value is None:
                return default
            return pattern.format(float(value))
        except Exception:
            return default

    def _fmt_int(value: Any, default: str = "n/a") -> str:
        try:
            if value is None:
                return default
            return str(int(value))
        except Exception:
            return default

    st.markdown("---")
    st.markdown("## 📊 Récapitulatif des sessions autonomes")

    header = "| # | Source | Objectif (résumé) | Statut | Score | Sharpe | Return | Max DD | Trades | Durée |"
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    export_rows: List[Dict[str, Any]] = []

    for i, h in enumerate(history, 1):
        objective_raw = str(h.get("objective", "") or "")
        objective_one_line = " ".join(objective_raw.split())
        obj_short = objective_one_line[:100]
        if len(objective_one_line) > 100:
            obj_short += "…"

        # Important: conserver le nom de source complet (pas de troncature).
        source = str(h.get("parametric_variant_id", "") or h.get("catalog_id", "") or "-")
        status = h.get("status", "?")
        sharpe = h.get("best_sharpe", 0)
        score = h.get("best_score")
        ret = h.get("best_return")
        max_dd = h.get("best_max_dd")
        trades = h.get("best_trades")
        duration = h.get("duration", 0)

        status_icon = {"success": "🏆", "max_iterations": "⏱️", "failed": "❌"}.get(
            status, "❓"
        )
        rows.append(
            f"| {i} | {source} | {obj_short} | {status_icon} {status} | "
            f"{_fmt_float(score, '{:.2f}')} | {_fmt_float(sharpe, '{:.3f}')} | "
            f"{_fmt_float(ret, '{:+.2f}%')} | {_fmt_float(max_dd, '{:.2f}%')} | "
            f"{_fmt_int(trades)} | {duration:.0f}s |"
        )
        export_rows.append(
            {
                "session_num": h.get("session_num"),
                "status": status,
                "best_score": score,
                "best_sharpe": sharpe,
                "best_return_pct": ret,
                "best_max_drawdown_pct": max_dd,
                "best_trades": trades,
                "duration_s": duration,
                "symbol": h.get("symbol"),
                "timeframe": h.get("timeframe"),
                "objective": objective_one_line,
                "source": source,
                "session_id": h.get("session_id"),
            }
        )

    st.markdown("\n".join(rows))

    # Meilleur global
    if history:
        best = max(history, key=lambda h: float(h.get("best_score", float("-inf")) or float("-inf")))
        st.success(
            f"**Meilleure session :** Score {_fmt_float(best.get('best_score'), '{:.2f}')} "
            f"(Sharpe {_fmt_float(best.get('best_sharpe'), '{:.3f}')}) — "
            f"{best.get('objective', '')[:80]}"
        )

    if export_rows:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(export_rows[0].keys()))
        writer.writeheader()
        writer.writerows(export_rows)
        st.download_button(
            "⬇️ Export leaderboard CSV",
            data=csv_buf.getvalue(),
            file_name="builder_autonomous_leaderboard.csv",
            mime="text/csv",
            key=f"builder_autonomous_leaderboard_export_{len(export_rows)}",
        )

    # Couverture catalogue
    try:
        cov = get_catalog_coverage()
        total = cov.get("total_objectives", 0)
        explored = cov.get("explored_count", 0)
        pct = cov.get("coverage_pct", 0.0)
        success = cov.get("success_count", 0)
        if total > 0:
            st.markdown(
                f"**Catalogue :** {explored}/{total} objectifs explores "
                f"({pct:.0f}%) — {success} avec Sharpe > 0"
            )
            st.progress(min(pct / 100.0, 1.0))
    except Exception:
        pass

    # Dernière fiche paramétrique (métadonnées utiles pour debug UI)
    parametric_runs = [h for h in history if h.get("parametric_variant_id")]
    if parametric_runs:
        last = parametric_runs[-1]
        st.markdown("**Dernier variant paramétrique**")
        st.json({
            "run_id": last.get("parametric_run_id", ""),
            "variant_id": last.get("parametric_variant_id", ""),
            "archetype_id": last.get("parametric_archetype_id", ""),
            "param_pack_id": last.get("parametric_param_pack_id", ""),
            "params": last.get("parametric_params", {}),
            "proposal": last.get("parametric_proposal", {}),
            "builder_text": last.get("parametric_builder_text", ""),
            "fingerprint": last.get("parametric_fingerprint", ""),
        })


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def render_builder_view(
    state: Any,
    df: Any,
    status_container: Any,
) -> None:
    """
    Rendu principal du mode Strategy Builder.

    Supporte deux modes :
    - **Manuel** : exécute une session unique avec l'objectif saisi
    - **Autonome 24/24** : génère des objectifs automatiquement et boucle
    """
    model = state.builder_model
    max_iterations = state.builder_max_iterations
    target_sharpe = state.builder_target_sharpe
    ollama_host = str(
        getattr(state, "builder_ollama_host", None)
        or "http://127.0.0.1:11434"
    ).strip()
    preload_model = bool(getattr(state, "builder_preload_model", True))
    keep_alive_minutes = int(getattr(state, "builder_keep_alive_minutes", 20))
    unload_after_run = bool(getattr(state, "builder_unload_after_run", False))
    auto_start_ollama = bool(getattr(state, "builder_auto_start_ollama", True))
    auto_market_pick = bool(getattr(state, "builder_auto_market_pick", False))
    capital_raw = getattr(state, "builder_capital", 10000.0)
    try:
        capital = float(capital_raw)
    except (TypeError, ValueError):
        capital = 10000.0

    # Contexte de marché — si rien n'est sélectionné, on pioche parmi les tokens disponibles
    available_tokens = list(getattr(state, "available_tokens", []) or [])
    available_tfs = _sanitize_builder_timeframes(
        list(getattr(state, "available_timeframes", []) or []),
        fallback="1h",
    )

    _raw_symbol = (
        getattr(state, "symbol", None)
        or st.session_state.get("selected_symbol")
        or ""
    )
    _raw_timeframe = (
        getattr(state, "timeframe", None)
        or st.session_state.get("selected_timeframe")
        or ""
    )

    # Fallback intelligent quand rien n'est sélectionné.
    # Évite le biais "premier élément de liste" en utilisant un bootstrap aléatoire stable de session.
    symbol = (
        str(_raw_symbol).strip().upper()
        if _raw_symbol
        else _stable_random_pick("_builder_startup_symbol", available_tokens, "BTCUSDC").upper()
    )
    timeframe = (
        str(_raw_timeframe).strip()
        if _raw_timeframe
        else _stable_random_pick(
            "_builder_startup_timeframe",
            available_tfs,
            "1h",
        )
    )
    if not _is_builder_supported_timeframe(timeframe):
        timeframe = _stable_random_pick(
            "_builder_startup_timeframe",
            available_tfs,
            "1h",
        )

    # ── DIAG: Mode auto-sélection marché ──
    logger.info(
        "🔍 [DIAG] auto_market_pick = %s | Symbole/TF par défaut: %s/%s",
        "✅ ACTIVÉ" if auto_market_pick else "❌ DÉSACTIVÉ",
        symbol,
        timeframe,
    )

    # Listes complètes pour le mode autonome (diversification multi-market).
    # En auto market pick, on ne traite comme "user_*" que les sélections explicites UI.
    if auto_market_pick:
        user_symbols = [
            str(s or "").strip().upper()
            for s in (st.session_state.get("symbols_select", []) or [])
            if str(s or "").strip()
        ]
        user_symbols = [s for s in user_symbols if s in available_tokens]
        user_timeframes = [
            str(tf or "").strip()
            for tf in (st.session_state.get("timeframes_select", []) or [])
            if str(tf or "").strip()
        ]
        user_timeframes = [tf for tf in user_timeframes if tf in available_tfs]
    else:
        user_symbols = list(getattr(state, "symbols", []) or [])
        user_timeframes = list(getattr(state, "timeframes", []) or [])

    # ── ROTATION DIVERSIFIÉE : éviter de toujours tester les mêmes tokens ──
    if not user_symbols and available_tokens:
        # Shuffle pour diversifier (évite biais alphabétique)
        shuffled_tokens = available_tokens.copy()
        random.shuffle(shuffled_tokens)

        # Tracker des marchés récemment testés (pour éviter répétitions)
        if "builder_tested_markets" not in st.session_state:
            st.session_state["builder_tested_markets"] = []

        tested_markets = st.session_state["builder_tested_markets"]
        recently_tested_tokens = {m["symbol"] for m in tested_markets[-20:]}  # 20 derniers

        # Priorité aux tokens NON récemment testés
        untested_tokens = [t for t in shuffled_tokens if t not in recently_tested_tokens]
        if untested_tokens:
            all_symbols = untested_tokens[:20]  # Top 20 non testés
        else:
            # Tous testés récemment → re-shuffle complet
            all_symbols = shuffled_tokens[:20]

        logger.info(
            "🔄 Rotation tokens: %d disponibles, %d récemment testés, %d sélectionnés pour ce run",
            len(available_tokens),
            len(recently_tested_tokens),
            len(all_symbols),
        )
    else:
        all_symbols = user_symbols if user_symbols else [symbol]

    all_timeframes_raw = user_timeframes if user_timeframes else (available_tfs or [timeframe])
    all_timeframes = _sanitize_builder_timeframes(all_timeframes_raw, fallback=timeframe or "1h")

    # ── DIAG: Univers de sélection ──
    logger.info(
        "🔍 [DIAG] Univers disponible: %d symboles × %d timeframes = %d combinaisons | "
        "Symboles: %s | Timeframes: %s",
        len(all_symbols), len(all_timeframes), len(all_symbols) * len(all_timeframes),
        ", ".join(all_symbols[:10]) + ("..." if len(all_symbols) > 10 else ""),
        ", ".join(all_timeframes),
    )

    fees_bps_raw = getattr(state, "fees_bps", None)
    if fees_bps_raw is None:
        fees_bps_raw = st.session_state.get("fees_bps", 10.0)
    try:
        fees_bps = float(fees_bps_raw)
    except (TypeError, ValueError):
        fees_bps = 10.0

    slippage_bps_raw = getattr(state, "slippage_bps", None)
    if slippage_bps_raw is None:
        slippage_bps_raw = st.session_state.get("slippage_bps", 5.0)
    try:
        slippage_bps = float(slippage_bps_raw)
    except (TypeError, ValueError):
        slippage_bps = 5.0

    autonomous = getattr(state, "builder_autonomous", False)

    if (df is None or len(df) == 0) and not autonomous:
        # Mode manuel : on a besoin des données pré-chargées
        # Tenter un chargement automatique si symbol/timeframe sont définis
        if symbol and timeframe and symbol != "UNKNOWN":
            try:
                from data.loader import load_ohlcv
                df = load_ohlcv(symbol, timeframe)
                st.caption(f"📥 Données chargées automatiquement: {symbol} {timeframe}")
            except Exception:
                pass
        if df is None or len(df) == 0:
            with status_container:
                show_status("error", "Aucune donnée OHLCV chargée — sélectionnez un token et timeframe, ou activez le mode autonome.")
            st.session_state.is_running = False
            return
    elif (df is None or len(df) == 0) and autonomous:
        # Mode autonome sans données pré-chargées : le builder chargera ses propres données
        # via auto_market_pick ou fallback aléatoire
        if symbol and timeframe and symbol != "UNKNOWN":
            try:
                from data.loader import load_ohlcv
                df = load_ohlcv(symbol, timeframe)
            except Exception:
                pass
        if df is None:
            # Précharge initiale: éviter tout biais "3 premiers tokens/2 premiers TF".
            probe_symbols = list(all_symbols)
            probe_timeframes = list(all_timeframes)
            if auto_market_pick:
                probe_symbols = _dedupe_keep_order(
                    [*probe_symbols, *available_tokens],
                    upper=True,
                )
                # Respect strict des TF utilisateur si explicitement fournis.
                if not user_timeframes:
                    probe_timeframes = _sanitize_builder_timeframes(
                        _dedupe_keep_order(
                            [*probe_timeframes, *available_tfs],
                            upper=False,
                        ),
                        fallback=timeframe or "1h",
                    )

            probe_pairs = [(s, tf) for s in probe_symbols for tf in probe_timeframes]
            if len(probe_pairs) > 1:
                random.shuffle(probe_pairs)
            max_probe_pairs = min(len(probe_pairs), 24)

            logger.info(
                "🔍 [DIAG] Startup data probe: trying %d/%d pairs (randomized order)",
                max_probe_pairs,
                len(probe_pairs),
            )

            for _try_sym, _try_tf in probe_pairs[:max_probe_pairs]:
                try:
                    from data.loader import load_ohlcv

                    df = load_ohlcv(_try_sym, _try_tf)
                    if df is not None and len(df) > 0:
                        symbol = _try_sym
                        timeframe = _try_tf
                        break
                except Exception:
                    continue
        if df is None or len(df) == 0:
            with status_container:
                show_status("error", "Aucune donnée OHLCV disponible pour démarrer le mode autonome.")
            st.session_state.is_running = False
            return
        st.caption(f"📥 Données initiales: {symbol} {timeframe} ({len(df)} barres)")

    # ══════════════════════════════════════════════════════════════════════
    # Mode MANUEL (comportement original)
    # ══════════════════════════════════════════════════════════════════════
    if not autonomous:
        raw_objective = str(getattr(state, "builder_objective", "") or "")
        objective = sanitize_objective_text(raw_objective)
        if raw_objective.strip() != objective:
            st.warning(
                "Objectif nettoyé automatiquement (des lignes de logs ont été retirées)."
            )
            st.session_state["builder_objective"] = objective
            # Ne pas modifier la clé widget après instanciation (StreamlitAPIException).
            # La sidebar appliquera cette synchro au prochain rerun, avant de créer le widget.
            st.session_state["_builder_objective_input_sync"] = objective
        if not objective or not objective.strip():
            with status_container:
                show_status("error", "Objectif vide — décrivez la stratégie souhaitée")
            st.session_state.is_running = False
            return

        st.markdown("## 🏗️ Strategy Builder")

        # Flux de pensée
        with st.expander("📂 Flux de pensée en terminal (optionnel)", expanded=False):
            st.code(
                f'Get-Content "{STREAM_FILE}" -Wait -Tail 80',
                language="powershell",
            )
            st.caption(
                f"📄 Fichier : `{STREAM_FILE}`  \n"
                "Alternative : surveiller ce fichier dans un terminal séparé."
            )

        st.markdown("---")

        run_model = model
        run_symbol = symbol
        run_timeframe = timeframe
        run_df = df
        skip_llm_prepare = False

        llm_client_for_market = None
        if auto_market_pick:
            with st.spinner(f"⏳ Préparation LLM `{model}` ({ollama_host})…"):
                ok, msg, resolved_model = _prepare_builder_llm(
                    model=model,
                    ollama_host=ollama_host,
                    preload_model=preload_model,
                    keep_alive_minutes=keep_alive_minutes,
                    auto_start_ollama=auto_start_ollama,
                )
                if ok:
                    st.caption(f"✅ {msg}")
                    run_model = resolved_model
                else:
                    st.error(f"❌ {msg}")
                    st.session_state.is_running = False
                    return

            llm_client_for_market = create_llm_client(
                LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=run_model,
                    ollama_host=ollama_host,
                ),
            )
            with st.spinner("🧭 Sélection automatique du marché (token/TF)…"):
                run_symbol, run_timeframe, run_df, market_pick = _pick_market_for_objective(
                    state=state,
                    objective=objective,
                    llm_client=llm_client_for_market,
                    default_symbol=symbol,
                    default_timeframe=timeframe,
                    fallback_df=df,
                )

            confidence = float(market_pick.get("confidence", 0.0) or 0.0)
            reason = str(market_pick.get("reason", "") or "").strip()
            source = str(market_pick.get("source", "") or "")
            data_source = str(market_pick.get("data_source", "") or "")
            if market_pick.get("load_error"):
                st.warning(
                    "⚠️ Choix marché LLM ignoré (chargement données impossible). "
                    f"Fallback sur {symbol} {timeframe}. "
                    f"Détail: {market_pick.get('load_error')}"
                )
            st.info(
                f"🧭 Marché sélectionné: `{run_symbol} {run_timeframe}` "
                f"(source: `{source}`, données: `{data_source}`, confiance: {confidence:.2f})."
            )
            if reason:
                st.caption(f"Raison LLM: {reason}")
            skip_llm_prepare = True

        session = _run_single_builder_session(
            objective=objective,
            model=run_model,
            ollama_host=ollama_host,
            preload_model=preload_model,
            keep_alive_minutes=keep_alive_minutes,
            unload_after_run=unload_after_run,
            auto_start_ollama=auto_start_ollama,
            max_iterations=max_iterations,
            target_sharpe=target_sharpe,
            capital=capital,
            symbol=run_symbol,
            timeframe=run_timeframe,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            df=run_df,
            skip_llm_prepare=skip_llm_prepare,
        )

        if session is not None:
            st.session_state["builder_session"] = session
            st.session_state["builder_last_objective"] = objective
            with status_container:
                show_status(
                    "success" if session.status == "success" else "info",
                    f"Builder terminé: {session.status} (Sharpe {session.best_sharpe:.3f})",
                )
        else:
            st.session_state.is_running = False

        return

    # ══════════════════════════════════════════════════════════════════════
    # Mode AUTONOME 24/24
    # ══════════════════════════════════════════════════════════════════════
    auto_pause = getattr(state, "builder_auto_pause", 10)
    use_llm_objectives = getattr(state, "builder_auto_use_llm", False)
    use_parametric_catalog = getattr(state, "builder_use_parametric_catalog", False)

    # Pré-génération du catalogue paramétrique si activé
    if use_parametric_catalog:
        with st.spinner("📐 Génération du catalogue paramétrique..."):
            n_variants = generate_parametric_catalog()
        st.caption(f"📐 {n_variants} fiches paramétriques générées")

    st.markdown("## 🔄 Strategy Builder — Mode Autonome 24/24")
    _market_display = (
        f"{', '.join(all_symbols)} × {', '.join(all_timeframes)}"
        if len(all_symbols) > 1 or len(all_timeframes) > 1
        else f"{symbol} {timeframe}"
    )
    st.caption(
        f"Configuration autonome | Max itérations/session: {max_iterations} | "
        f"Sharpe cible: {target_sharpe} | Capital: ${capital:,.0f} | "
        f"Marchés: {_market_display} | "
        f"Pause entre runs: {auto_pause}s | "
        f"Objectifs: {'Param.' if use_parametric_catalog else 'LLM' if use_llm_objectives else 'Templates'} | "
        f"Auto marché: {'ON' if auto_market_pick else 'OFF'}"
    )

    # Flux de pensée
    with st.expander("📂 Flux de pensée en terminal (optionnel)", expanded=False):
        st.code(
            f'Get-Content "{STREAM_FILE}" -Wait -Tail 80',
            language="powershell",
        )
        st.caption(
            f"📄 Fichier : `{STREAM_FILE}`  \n"
            "Alternative : surveiller ce fichier dans un terminal séparé."
        )

    st.markdown("---")

    # Préparer LLM une seule fois pour toute la boucle autonome
    with st.spinner(f"⏳ Préparation LLM `{model}` ({ollama_host})…"):
        ok, msg, resolved_model = _prepare_builder_llm(
            model=model,
            ollama_host=ollama_host,
            preload_model=preload_model,
            keep_alive_minutes=keep_alive_minutes,
            auto_start_ollama=auto_start_ollama,
        )
        if ok:
            st.caption(f"✅ {msg}")
            model = resolved_model
        else:
            show_status("error", msg)
            st.session_state.is_running = False
            return

    # Préparer le client LLM pour la génération d'objectifs (si mode LLM)
    llm_client_for_obj = None
    llm_client_for_market = None
    llm_config_shared = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=model,
        ollama_host=ollama_host,
    )
    if use_llm_objectives:
        llm_client_for_obj = create_llm_client(llm_config_shared)
    if auto_market_pick:
        llm_client_for_market = create_llm_client(llm_config_shared)

    objective_indicators = _get_builder_compatible_indicators(df)

    # Historique des sessions autonomes
    if "builder_autonomous_history" not in st.session_state:
        st.session_state["builder_autonomous_history"] = []
    history: List[Dict[str, Any]] = st.session_state["builder_autonomous_history"]

    # Compteur de session
    session_num = len(history)
    recap_placeholder = st.empty()
    session_placeholder = st.empty()

    while st.session_state.get("is_running", False):
        session_num += 1

        # ── Extraire les marchés récents pour forcer la diversité ──
        _recent_markets: list[tuple[str, str]] = [
            (str(h.get("symbol", "")), str(h.get("timeframe", "")))
            for h in history[-6:]
            if h.get("symbol") and h.get("timeframe")
        ]

        # ── DIAG: Historique de diversité ──
        logger.info(
            "🔍 [DIAG] Session #%d | Historique total: %d runs | Marchés récents (6 derniers): %s",
            session_num,
            len(history),
            _recent_markets if _recent_markets else "❌ VIDE (premier run ou historique perdu)",
        )

        # ── Générer l'objectif (multi-market : listes symbols/timeframes) ──
        current_catalog_id: Optional[str] = None
        current_parametric_meta: Dict[str, Any] = {}
        if use_parametric_catalog:
            # Priorité : fiches paramétriques (archetypes × param_packs)
            # Si auto_market_pick activé, ne PAS injecter symbol/TF dans le template
            parametric_result = get_next_parametric_objective(
                symbol=None if auto_market_pick else all_symbols,
                timeframe=None if auto_market_pick else all_timeframes,
            )
            if parametric_result is not None:
                objective = str(parametric_result.get("objective_text", "") or "")
                current_catalog_id = (
                    str(parametric_result.get("variant_id", "") or "").strip() or None
                )
                current_parametric_meta = dict(parametric_result)
            else:
                st.info("📐 Catalogue paramétrique vide — fallback templates")
                catalog_result = get_next_catalog_objective(
                    symbol=None if auto_market_pick else all_symbols,
                    timeframe=None if auto_market_pick else all_timeframes,
                )
                if catalog_result is not None:
                    objective, current_catalog_id = catalog_result
                else:
                    objective = generate_random_objective(
                        symbol=("{symbol}" if auto_market_pick else all_symbols),
                        timeframe=("{timeframe}" if auto_market_pick else all_timeframes),
                        available_indicators=objective_indicators,
                    )
        elif use_llm_objectives and llm_client_for_obj is not None:
            with st.spinner("🧠 Génération de l'objectif par LLM..."):
                objective = generate_llm_objective(
                    llm_client_for_obj,
                    symbol=None if auto_market_pick else all_symbols,
                    timeframe=None if auto_market_pick else all_timeframes,
                    available_indicators=objective_indicators,
                    recent_markets=_recent_markets or None,
                )
        else:
            # Catalogue systématique en priorité, fallback random
            catalog_result = get_next_catalog_objective(
                symbol=None if auto_market_pick else all_symbols,
                timeframe=None if auto_market_pick else all_timeframes,
            )
            if catalog_result is not None:
                objective, current_catalog_id = catalog_result
            else:
                st.info("📚 Catalogue épuisé — passage en mode aléatoire")
                objective = generate_random_objective(
                    symbol=("{symbol}" if auto_market_pick else all_symbols),
                    timeframe=("{timeframe}" if auto_market_pick else all_timeframes),
                    available_indicators=objective_indicators,
                )
        objective = sanitize_objective_text(objective)
        if current_parametric_meta:
            st.caption(
                "📐 Variant "
                f"{current_parametric_meta.get('variant_id', 'n/a')} | "
                f"archetype={current_parametric_meta.get('archetype_id', 'n/a')} | "
                f"pack={current_parametric_meta.get('param_pack_id', 'n/a')}"
            )

        session_symbol = symbol
        session_timeframe = timeframe
        if auto_market_pick:
            session_symbol, session_timeframe = _pick_non_recent_market(
                all_symbols,
                all_timeframes,
                _recent_markets,
            )
        default_session_symbol = session_symbol
        default_session_timeframe = session_timeframe
        session_df = df
        if auto_market_pick:
            session_df, pre_load_error, pre_data_source = _load_builder_market_data(
                state=state,
                symbol=default_session_symbol,
                timeframe=default_session_timeframe,
                fallback_df=df,
            )
            if pre_load_error:
                logger.warning(
                    "🔍 [DIAG] Default session market preload failed for %s %s: %s (source=%s)",
                    default_session_symbol,
                    default_session_timeframe,
                    pre_load_error,
                    pre_data_source,
                )
        market_pick: Dict[str, Any] = {}
        if auto_market_pick and llm_client_for_market is not None:
            with st.spinner("🧭 Sélection automatique du marché (token/TF)…"):
                session_symbol, session_timeframe, session_df, market_pick = _pick_market_for_objective(
                    state=state,
                    objective=objective,
                    llm_client=llm_client_for_market,
                    default_symbol=default_session_symbol,
                    default_timeframe=default_session_timeframe,
                    fallback_df=session_df,
                    recent_markets=_recent_markets or None,
                )
            confidence = float(market_pick.get("confidence", 0.0) or 0.0)
            source = str(market_pick.get("source", "") or "")
            data_source = str(market_pick.get("data_source", "") or "")
            reason = str(market_pick.get("reason", "") or "")

            # Détection override UI
            is_override = (
                default_session_symbol and default_session_timeframe and
                (
                    session_symbol != default_session_symbol
                    or session_timeframe != default_session_timeframe
                )
            )

            if is_override:
                # UI warning pour override explicite
                st.warning(
                    f"🔄 **Override LLM** : {default_session_symbol} {default_session_timeframe} → {session_symbol} {session_timeframe}\n\n"
                    f"**Raison:** {reason}\n\n"
                    f"*Source: {source} | Confidence: {confidence:.2f}*"
                )
                # Log structuré override
                logger.info(
                    "Market selection: source=llm_override, original=%s %s, final=%s %s, reason=%s, confidence=%.2f",
                    default_session_symbol,
                    default_session_timeframe,
                    session_symbol,
                    session_timeframe,
                    reason,
                    confidence,
                )
            else:
                st.caption(
                    f"🧭 Session #{session_num}: {session_symbol} {session_timeframe} "
                    f"(source={source}, data={data_source}, conf={confidence:.2f})"
                )

            # ── DIAG: Sélection finale ──
            logger.info(
                "🔍 [DIAG] Session #%d → Marché sélectionné: %s %s | "
                "Source: %s | Confidence: %.2f | Data: %s | "
                "Candidats: %d symbols × %d timeframes",
                session_num,
                session_symbol,
                session_timeframe,
                source,
                confidence,
                data_source,
                len(market_pick.get("candidate_symbols", [])),
                len(market_pick.get("candidate_timeframes", [])),
            )
        else:
            # ── DIAG: Mode auto-pick désactivé ──
            logger.info(
                "🔍 [DIAG] Session #%d → Marché PAR DÉFAUT (auto_market_pick=OFF): %s %s",
                session_num,
                session_symbol,
                session_timeframe,
            )

        # ── Remplacer les placeholders {symbol}/{timeframe} après sélection marché ──
        if "{symbol}" in objective or "{timeframe}" in objective:
            objective = objective.replace("{symbol}", session_symbol)
            objective = objective.replace("{timeframe}", session_timeframe)
            objective = sanitize_objective_text(objective)
            logger.info(
                "🔍 [DIAG] Placeholders remplacés dans objectif → %s %s",
                session_symbol,
                session_timeframe,
            )

        # ── Exécuter la session (remplace l'affichage précédent) ──
        with session_placeholder.container():
            t0 = time.perf_counter()
            session = _run_single_builder_session(
                objective=objective,
                model=model,
                ollama_host=ollama_host,
                preload_model=preload_model,
                keep_alive_minutes=keep_alive_minutes,
                unload_after_run=False,
                auto_start_ollama=auto_start_ollama,
                max_iterations=max_iterations,
                target_sharpe=target_sharpe,
                capital=capital,
                symbol=session_symbol,
                timeframe=session_timeframe,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
                df=session_df,
                session_label=f"🔄 Session autonome #{session_num}",
                skip_llm_prepare=True,
                show_config_caption=False,
            )
            duration = time.perf_counter() - t0

        # ── Enregistrer le résultat ──
        if session is not None:
            best_metrics = {}
            if session.best_iteration and session.best_iteration.backtest_result:
                best_metrics = session.best_iteration.backtest_result.metrics
            best_score = getattr(session, "best_score", float("-inf"))
            if best_score == float("-inf"):
                best_score = None

            history.append({
                "session_num": session_num,
                "objective": objective,
                "catalog_id": current_catalog_id or "",
                "parametric_run_id": current_parametric_meta.get("run_id", ""),
                "parametric_variant_id": current_parametric_meta.get("variant_id", ""),
                "parametric_archetype_id": current_parametric_meta.get("archetype_id", ""),
                "parametric_param_pack_id": current_parametric_meta.get("param_pack_id", ""),
                "parametric_params": current_parametric_meta.get("params", {}),
                "parametric_proposal": current_parametric_meta.get("proposal", {}),
                "parametric_builder_text": current_parametric_meta.get("builder_text", ""),
                "parametric_fingerprint": current_parametric_meta.get("fingerprint", ""),
                "status": session.status,
                "best_sharpe": session.best_sharpe,
                "best_score": best_score,
                "best_return": best_metrics.get("total_return_pct"),
                "best_max_dd": best_metrics.get("max_drawdown_pct"),
                "best_pf": best_metrics.get("profit_factor"),
                "best_trades": best_metrics.get("total_trades"),
                "n_iterations": len(session.iterations),
                "duration": duration,
                "session_id": session.session_id,
                "symbol": session_symbol,
                "timeframe": session_timeframe,
            })
            st.session_state["builder_autonomous_history"] = history
            st.session_state["builder_session"] = session

            # Marquer l'objectif catalogue comme exploré
            if current_catalog_id is not None:
                mark_catalog_objective_explored(
                    current_catalog_id,
                    status=session.status,
                    best_sharpe=session.best_sharpe,
                    session_id=session.session_id,
                    symbol=session_symbol,
                    timeframe=session_timeframe,
                )
        else:
            history.append({
                "session_num": session_num,
                "objective": objective,
                "catalog_id": current_catalog_id or "",
                "parametric_run_id": current_parametric_meta.get("run_id", ""),
                "parametric_variant_id": current_parametric_meta.get("variant_id", ""),
                "parametric_archetype_id": current_parametric_meta.get("archetype_id", ""),
                "parametric_param_pack_id": current_parametric_meta.get("param_pack_id", ""),
                "parametric_params": current_parametric_meta.get("params", {}),
                "parametric_proposal": current_parametric_meta.get("proposal", {}),
                "parametric_builder_text": current_parametric_meta.get("builder_text", ""),
                "parametric_fingerprint": current_parametric_meta.get("fingerprint", ""),
                "status": "error",
                "best_sharpe": None,
                "best_score": None,
                "best_return": None,
                "best_max_dd": None,
                "best_pf": None,
                "best_trades": None,
                "n_iterations": 0,
                "duration": duration,
                "session_id": "",
                "symbol": session_symbol,
                "timeframe": session_timeframe,
            })
            st.session_state["builder_autonomous_history"] = history

            # Marquer l'objectif catalogue comme exploré (même en erreur)
            if current_catalog_id is not None:
                mark_catalog_objective_explored(
                    current_catalog_id,
                    status="error",
                    best_sharpe=0.0,
                    session_id="",
                    symbol=session_symbol,
                    timeframe=session_timeframe,
                )

        # ── Afficher le récap mis à jour ──
        with recap_placeholder.container():
            _render_autonomous_recap(history)

        # ── Vérifier si on doit continuer ──
        if not st.session_state.get("is_running", False):
            break

        # ── Pause configurable avec countdown ──
        if auto_pause > 0:
            countdown_placeholder = st.empty()
            for remaining in range(auto_pause, 0, -1):
                if not st.session_state.get("is_running", False):
                    break
                countdown_placeholder.info(
                    f"⏱️ Prochaine session dans **{remaining}s**..."
                )
                time.sleep(1)
            try:
                countdown_placeholder.empty()
            except Exception:
                pass

    # ── Fin de la boucle autonome ──
    with recap_placeholder.container():
        _render_autonomous_recap(history)

    with status_container:
        n = len(history)
        best_ever = max(
            (h.get("best_sharpe", 0) for h in history), default=0,
        )
        show_status(
            "success" if best_ever > 0 else "info",
            f"Mode autonome terminé : {n} sessions | Meilleur Sharpe: {best_ever:.3f}",
        )

    if unload_after_run:
        with st.spinner(f"💾 Déchargement du modèle `{model}`…"):
            if _unload_ollama_model(model=model, ollama_host=ollama_host):
                st.caption(f"✅ Modèle `{model}` déchargé")
            else:
                st.warning(f"⚠️ Impossible de décharger `{model}`")

    st.session_state.is_running = False
