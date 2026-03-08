"""
Module-ID: cli.cross_token

Purpose: Validation cross-token des sessions Builder.

Role in pipeline: reload sessions -> replay best builder strategy -> score robustness

Key components: collect_builder_candidates(), evaluate_cross_token_candidates(),
build_cross_token_report(), run_cross_token_command()

Inputs: sandbox_strategies/*/session_summary.json + OHLCV basket

Outputs: JSON report + optional catalog promotion
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_STATUSES = ("success", "max_iterations")
DEFAULT_LIQUID_TOKENS = (
    "BTCUSDC",
    "ETHUSDC",
    "SOLUSDC",
    "DOGEUSDC",
    "ADAUSDC",
    "BNBUSDC",
    "XRPUSDC",
    "LINKUSDC",
    "AVAXUSDC",
    "HBARUSDC",
    "TRXUSDC",
    "ALGOUSDC",
)
MIN_TRADES_BY_TIMEFRAME = {
    "1w": 3,
    "1d": 5,
    "4h": 8,
    "1h": 10,
    "30m": 12,
    "15m": 15,
    "5m": 20,
    "3m": 25,
    "1m": 30,
}
DEFAULT_CHUNK_SIZE = 50
TIMEFRAME_RE = re.compile(r"\b(1m|3m|5m|15m|30m|1h|4h|1d|1w|1M)\b", re.I)
ID_RE = re.compile(r"\bid\s*:\s*([a-z0-9_]+)", re.I)
SYMBOL_RE = re.compile(r"\b([a-z0-9]{2,12}usd[ct])\b", re.I)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _split_multi_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    items: List[str] = []
    for raw in values:
        if raw is None:
            continue
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
        items.extend(parts)
    return items


def _safe_num(value: Any, default: float = float("-inf")) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _resolve_output_path(output: Optional[str]) -> Path:
    if output:
        return Path(output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"cross_token_{timestamp}.json"


def _resolve_data_dir() -> Path:
    from data.loader import _get_data_dir

    return _get_data_dir()


def resolve_unique_timeframe(summary: Mapping[str, Any]) -> tuple[Optional[str], str]:
    text = f"{summary.get('objective') or ''} {summary.get('session_id') or ''}".lower()
    found = sorted({match.lower() for match in TIMEFRAME_RE.findall(text)})
    if len(found) == 1:
        return found[0], "unique"
    if len(found) > 1:
        return None, "ambiguous"
    return None, "missing"


def infer_strategy_id(summary: Mapping[str, Any]) -> Optional[str]:
    objective = str(summary.get("objective") or "")
    match = ID_RE.search(objective)
    if not match:
        return None
    return match.group(1).lower()


def infer_source_symbol(summary: Mapping[str, Any]) -> Optional[str]:
    text = f"{summary.get('objective') or ''} {summary.get('session_id') or ''}"
    match = SYMBOL_RE.search(text)
    if not match:
        return None
    return match.group(1).upper()


def select_best_iteration(session_dir: Path, summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    candidates: List[tuple[Any, ...]] = []
    total_iterations = int(summary.get("total_iterations") or 0)
    for iteration in summary.get("iterations") or []:
        if not isinstance(iteration, dict):
            continue
        number = iteration.get("iteration")
        if not isinstance(number, int):
            continue
        strategy_path = session_dir / f"strategy_v{number}.py"
        if not strategy_path.exists():
            if number == total_iterations and (session_dir / "strategy.py").exists():
                strategy_path = session_dir / "strategy.py"
            else:
                continue

        total_return = _safe_num(iteration.get("return_pct"))
        sharpe = _safe_num(iteration.get("sharpe"))
        continuous_score = _safe_num(iteration.get("continuous_score"))
        profit_factor = _safe_num(iteration.get("profit_factor"))
        drawdown = _safe_num(iteration.get("max_drawdown_pct"))
        trades = _safe_num(iteration.get("trades"))
        positive = 1 if total_return > 0 else 0
        not_ruined = 1 if drawdown > -100 else 0
        fallback_score = (sharpe * 10.0) + (total_return * 0.05) + (profit_factor * 5.0) + (trades * 0.01)
        effective_score = continuous_score if continuous_score != float("-inf") else fallback_score
        candidates.append(
            (
                positive,
                not_ruined,
                effective_score,
                sharpe,
                total_return,
                profit_factor,
                trades,
                number,
                strategy_path,
                iteration,
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)
    chosen = candidates[0]
    return {
        "iteration": chosen[7],
        "strategy_path": str(chosen[8]),
        "iteration_metrics": chosen[9],
    }


def _available_basket(tokens: Sequence[str], timeframe: str, data_dir: Path) -> List[str]:
    from data.loader import is_usable_dataset

    basket: List[str] = []
    for token in tokens:
        if (data_dir / f"{token}_{timeframe}.parquet").exists() and is_usable_dataset(token, timeframe):
            basket.append(token)
    return basket


def collect_builder_candidates(
    *,
    sandbox_root: Path,
    statuses: Sequence[str],
    session_ids: Sequence[str],
    strategy_ids: Sequence[str],
    timeframe_filters: Sequence[str],
    tokens: Sequence[str],
    min_basket_size: int,
    max_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    data_dir = _resolve_data_dir()
    allowed_statuses = {status.strip() for status in statuses if status and str(status).strip()}
    allowed_session_ids = {session_id.strip() for session_id in session_ids if session_id and str(session_id).strip()}
    allowed_strategy_ids = {strategy_id.strip().lower() for strategy_id in strategy_ids if strategy_id and str(strategy_id).strip()}
    allowed_timeframes = {timeframe.strip().lower() for timeframe in timeframe_filters if timeframe and str(timeframe).strip()}

    skip_reasons: Counter[str] = Counter()
    candidates: List[Dict[str, Any]] = []

    for summary_path in sorted(sandbox_root.rglob("session_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            skip_reasons["json_error"] += 1
            continue

        session_id = str(summary.get("session_id") or summary_path.parent.name)
        status = str(summary.get("status") or "")
        if allowed_statuses and status not in allowed_statuses:
            continue
        if allowed_session_ids and session_id not in allowed_session_ids:
            continue

        timeframe, timeframe_status = resolve_unique_timeframe(summary)
        if timeframe_status != "unique":
            skip_reasons[f"timeframe_{timeframe_status}"] += 1
            continue
        if allowed_timeframes and timeframe not in allowed_timeframes:
            continue

        strategy_id = infer_strategy_id(summary)
        if allowed_strategy_ids and (strategy_id or "").lower() not in allowed_strategy_ids:
            continue

        chosen = select_best_iteration(summary_path.parent, summary)
        if not chosen:
            skip_reasons["no_strategy_iteration"] += 1
            continue

        basket = _available_basket(tokens, timeframe, data_dir)
        if len(basket) < int(min_basket_size):
            skip_reasons["basket_too_small"] += 1
            continue

        objective = " ".join(str(summary.get("objective") or "").split())
        candidates.append(
            {
                "session_id": session_id,
                "status": status,
                "timeframe": timeframe,
                "strategy_id": strategy_id,
                "source_symbol": infer_source_symbol(summary),
                "objective": objective[:280],
                "strategy_path": chosen["strategy_path"],
                "best_iteration": chosen["iteration"],
                "source_metrics": chosen["iteration_metrics"],
                "basket": basket,
            }
        )
        if max_candidates and len(candidates) >= max_candidates:
            break

    return {
        "candidates": candidates,
        "skip_reasons": dict(skip_reasons),
        "data_dir": str(data_dir),
    }


def _load_builder_strategy_class(strategy_path: Path):
    module_name = f"cross_token_{strategy_path.stem}_{abs(hash(str(strategy_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de créer spec pour {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    strategy_cls = getattr(module, "BuilderGeneratedStrategy", None)
    if strategy_cls is None:
        raise AttributeError("BuilderGeneratedStrategy absent")
    return strategy_cls


def classify_metrics(metrics: Mapping[str, Any], timeframe: str) -> Dict[str, Any]:
    min_trades = MIN_TRADES_BY_TIMEFRAME.get(timeframe, 10)
    total_return = float(metrics.get("total_return_pct", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    drawdown = float(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0)) or 0.0)
    trades = int(metrics.get("total_trades", 0) or 0)
    ruined = bool(metrics.get("account_ruined", False)) or drawdown <= -100.0
    alive = (not ruined) and total_return > 0 and profit_factor > 1.0
    robust = alive and sharpe > 0 and drawdown > -50.0 and trades >= min_trades
    return {
        "return_pct": total_return,
        "sharpe": sharpe,
        "drawdown_pct": drawdown,
        "profit_factor": profit_factor,
        "trades": trades,
        "alive": alive,
        "robust": robust,
    }


def _evaluate_cross_token_candidates_direct(
    *,
    candidates: Sequence[Mapping[str, Any]],
    capital: float,
    fees_bps: int,
    slippage_bps: Optional[float],
) -> Dict[str, Any]:
    from backtest.engine import BacktestEngine
    from data.loader import load_ohlcv_file
    from utils.config import Config

    config_kwargs: Dict[str, Any] = {"fees_bps": fees_bps}
    if slippage_bps is not None:
        config_kwargs["slippage_bps"] = slippage_bps
    engine = BacktestEngine(initial_capital=capital, config=Config(**config_kwargs))
    data_dir = _resolve_data_dir()
    cache: Dict[tuple[str, str], Any] = {}
    results: List[Dict[str, Any]] = []
    errors = {"load_strategy": 0, "run_backtest": 0}

    def load_df(token: str, timeframe: str):
        key = (token, timeframe)
        if key not in cache:
            cache[key], _ = load_ohlcv_file(
                data_dir / f"{token}_{timeframe}.parquet",
                symbol=token,
                timeframe=timeframe,
            )
        return cache[key]

    for index, candidate in enumerate(candidates):
        strategy_path = Path(str(candidate["strategy_path"]))
        try:
            strategy_cls = _load_builder_strategy_class(strategy_path)
            strategy = strategy_cls()
            params = dict(getattr(strategy, "default_params", {}) or {})
        except BaseException as exc:
            errors["load_strategy"] += 1
            results.append(
                {
                    **dict(candidate),
                    "error": f"load:{type(exc).__name__}:{exc}",
                    "tested": 0,
                    "alive_count": 0,
                    "robust_count": 0,
                    "alive_ratio": 0.0,
                    "robust_ratio": 0.0,
                    "avg_return": None,
                    "source_params": {},
                    "token_results": [],
                }
            )
            continue

        token_results: List[Dict[str, Any]] = []
        for token in candidate.get("basket") or []:
            try:
                df = load_df(str(token), str(candidate["timeframe"]))
                run_result = engine.run(
                    df=df,
                    strategy=strategy_cls(),
                    params=params,
                    symbol=str(token),
                    timeframe=str(candidate["timeframe"]),
                    silent_mode=True,
                )
                raw_metrics = run_result.metrics.to_dict() if hasattr(run_result.metrics, "to_dict") else dict(run_result.metrics)
                scored = classify_metrics(raw_metrics, str(candidate["timeframe"]))
                token_results.append({"token": token, **scored})
            except BaseException as exc:
                errors["run_backtest"] += 1
                token_results.append(
                    {
                        "token": token,
                        "error": f"{type(exc).__name__}:{exc}",
                        "alive": False,
                        "robust": False,
                    }
                )

        tested = len(token_results)
        alive_count = sum(1 for item in token_results if item.get("alive"))
        robust_count = sum(1 for item in token_results if item.get("robust"))
        numeric_returns = [float(item["return_pct"]) for item in token_results if "return_pct" in item]
        results.append(
            {
                **dict(candidate),
                "tested": tested,
                "alive_count": alive_count,
                "robust_count": robust_count,
                "alive_ratio": (alive_count / tested) if tested else 0.0,
                "robust_ratio": (robust_count / tested) if tested else 0.0,
                "avg_return": (sum(numeric_returns) / len(numeric_returns)) if numeric_returns else None,
                "source_params": params,
                "token_results": token_results,
            }
        )

    return {"results": results, "errors": errors, "data_dir": str(data_dir)}


def _evaluate_cross_token_chunk_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    return _evaluate_cross_token_candidates_direct(
        candidates=payload.get("candidates") or [],
        capital=float(payload.get("capital") or 10000.0),
        fees_bps=int(payload.get("fees_bps") or 10),
        slippage_bps=payload.get("slippage_bps"),
    )


def evaluate_cross_token_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    capital: float,
    fees_bps: int,
    slippage_bps: Optional[float],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, Any]:
    if chunk_size <= 0 or len(candidates) <= chunk_size:
        return _evaluate_cross_token_candidates_direct(
            candidates=candidates,
            capital=capital,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
        )

    tasks: List[Dict[str, Any]] = []
    for start in range(0, len(candidates), chunk_size):
        chunk = list(candidates[start:start + chunk_size])
        tasks.append(
            {
                "candidates": chunk,
                "capital": capital,
                "fees_bps": fees_bps,
                "slippage_bps": slippage_bps,
            }
        )

    aggregated_results: List[Dict[str, Any]] = []
    aggregated_errors = {"load_strategy": 0, "run_backtest": 0}
    data_dir = _resolve_data_dir()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        for chunk_result in pool.imap(_evaluate_cross_token_chunk_worker, tasks):
            aggregated_results.extend(chunk_result.get("results") or [])
            errors = chunk_result.get("errors") or {}
            aggregated_errors["load_strategy"] += int(errors.get("load_strategy") or 0)
            aggregated_errors["run_backtest"] += int(errors.get("run_backtest") or 0)

    return {
        "results": aggregated_results,
        "errors": aggregated_errors,
        "data_dir": str(data_dir),
    }


def _required_robust_count(result: Mapping[str, Any], min_robust_count: int, min_robust_ratio: float) -> int:
    tested = max(1, int(result.get("tested") or 0))
    ratio_count = int(math.ceil(min_robust_ratio * tested))
    return max(int(min_robust_count), ratio_count)


def build_cross_token_report(
    *,
    results: Sequence[Mapping[str, Any]],
    skip_reasons: Mapping[str, Any],
    errors: Mapping[str, Any],
    min_robust_count: int,
    min_robust_ratio: float,
    top: int,
    sandbox_root: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    interesting: List[Dict[str, Any]] = []
    family = defaultdict(lambda: {"count": 0, "interesting": 0, "best_ratio": 0.0, "avg_ratio": 0.0})

    for raw_result in results:
        result = dict(raw_result)
        family_key = str(result.get("strategy_id") or "non_canonical")
        stats = family[family_key]
        stats["count"] += 1
        stats["avg_ratio"] += float(result.get("robust_ratio") or 0.0)
        stats["best_ratio"] = max(stats["best_ratio"], float(result.get("robust_ratio") or 0.0))
        if int(result.get("robust_count") or 0) >= _required_robust_count(result, min_robust_count, min_robust_ratio):
            interesting.append(result)
            stats["interesting"] += 1

    def sort_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
        source_metrics = result.get("source_metrics") or {}
        return (
            float(result.get("robust_ratio") or 0.0),
            int(result.get("robust_count") or 0),
            float(result.get("alive_ratio") or 0.0),
            _safe_num(result.get("avg_return"), -999999.0),
            _safe_num(source_metrics.get("return_pct"), -999999.0),
        )

    sorted_results = sorted(results, key=sort_key, reverse=True)
    sorted_interesting = sorted(interesting, key=sort_key, reverse=True)

    report = {
        "generated_at": _now_iso(),
        "sandbox_root": str(sandbox_root),
        "data_dir": str(data_dir),
        "evaluated": len(results),
        "skip_reasons": dict(skip_reasons),
        "errors": dict(errors),
        "thresholds": {
            "min_robust_count": int(min_robust_count),
            "min_robust_ratio": float(min_robust_ratio),
        },
        "bands": {
            "robust_ge_50pct": sum(1 for result in results if float(result.get("robust_ratio") or 0.0) >= 0.50),
            "robust_25_to_49pct": sum(1 for result in results if 0.25 <= float(result.get("robust_ratio") or 0.0) < 0.50),
            "robust_01_to_24pct": sum(1 for result in results if 0.0 < float(result.get("robust_ratio") or 0.0) < 0.25),
            "robust_0pct": sum(1 for result in results if float(result.get("robust_ratio") or 0.0) == 0.0),
        },
        "interesting_count": len(sorted_interesting),
        "family_top": [
            {
                "strategy_id": key,
                "count": value["count"],
                "interesting": value["interesting"],
                "best_ratio": round(value["best_ratio"], 3),
                "avg_ratio": round((value["avg_ratio"] / value["count"]) if value["count"] else 0.0, 3),
            }
            for key, value in sorted(
                family.items(),
                key=lambda item: (item[1]["interesting"], item[1]["best_ratio"], item[1]["count"]),
                reverse=True,
            )[:20]
        ],
        "top_survivors": [
            {
                "session_id": result.get("session_id"),
                "status": result.get("status"),
                "timeframe": result.get("timeframe"),
                "strategy_id": result.get("strategy_id"),
                "source_symbol": result.get("source_symbol"),
                "best_iteration": result.get("best_iteration"),
                "source_return_pct": round(_safe_num((result.get("source_metrics") or {}).get("return_pct"), 0.0), 2),
                "source_sharpe": round(_safe_num((result.get("source_metrics") or {}).get("sharpe"), 0.0), 3),
                "robust_count": int(result.get("robust_count") or 0),
                "tested": int(result.get("tested") or 0),
                "robust_ratio": round(float(result.get("robust_ratio") or 0.0), 3),
                "alive_count": int(result.get("alive_count") or 0),
                "alive_ratio": round(float(result.get("alive_ratio") or 0.0), 3),
                "avg_return": None if result.get("avg_return") is None else round(float(result.get("avg_return")), 2),
                "robust_tokens": [item["token"] for item in result.get("token_results") or [] if item.get("robust")],
                "error": result.get("error"),
            }
            for result in sorted_results[: max(1, int(top))]
        ],
        "interesting_survivors": [
            {
                "session_id": result.get("session_id"),
                "status": result.get("status"),
                "timeframe": result.get("timeframe"),
                "strategy_id": result.get("strategy_id"),
                "source_symbol": result.get("source_symbol"),
                "best_iteration": result.get("best_iteration"),
                "robust_count": int(result.get("robust_count") or 0),
                "tested": int(result.get("tested") or 0),
                "robust_ratio": round(float(result.get("robust_ratio") or 0.0), 3),
                "alive_count": int(result.get("alive_count") or 0),
                "alive_ratio": round(float(result.get("alive_ratio") or 0.0), 3),
                "avg_return": None if result.get("avg_return") is None else round(float(result.get("avg_return")), 2),
                "robust_tokens": [item["token"] for item in result.get("token_results") or [] if item.get("robust")],
            }
            for result in sorted_interesting
        ],
        "results": sorted_results,
    }
    return report


def _merge_extra_catalog_tags(entry: Dict[str, Any], extra_tags: Sequence[str], path: Optional[Path] = None) -> Dict[str, Any]:
    if not extra_tags:
        return entry

    from catalog.strategy_catalog import get_entry, upsert_entry

    existing = get_entry(entry["id"], path=path) or {}
    merged: List[str] = []
    for group in (existing.get("tags") or [], entry.get("tags") or [], extra_tags):
        for tag in group:
            value = str(tag or "").strip()
            if value and value not in merged:
                merged.append(value)
    updated = dict(entry)
    updated["tags"] = merged
    return upsert_entry(updated, path=path)


def promote_cross_token_survivors(
    report: Mapping[str, Any],
    *,
    target_category: str,
    extra_tags: Sequence[str],
) -> List[Dict[str, Any]]:
    from catalog.strategy_catalog import upsert_from_cross_token_result

    promoted: List[Dict[str, Any]] = []
    interesting_ids = {item.get("session_id") for item in report.get("interesting_survivors") or []}
    for result in report.get("results") or []:
        if result.get("session_id") not in interesting_ids:
            continue
        entry = upsert_from_cross_token_result(result, target_category=target_category)
        promoted.append(_merge_extra_catalog_tags(entry, extra_tags))
    return promoted


def _format_ratio(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _format_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    from . import formatters as _formatters

    return _formatters.format_table(list(headers), [list(row) for row in rows], indent=2, padding=2)


def _print_report_text(report: Mapping[str, Any], output_path: Path, promoted: Sequence[Mapping[str, Any]]) -> None:
    from . import formatters as _formatters

    _formatters.print_header("Cross-Token Validation")
    print(f"  Sandbox   : {report.get('sandbox_root')}")
    print(f"  Data dir  : {report.get('data_dir')}")
    print(f"  Evalues   : {report.get('evaluated')}")
    print(f"  Survivants: {report.get('interesting_count')}")
    print(f"  JSON      : {output_path}")

    skip = report.get("skip_reasons") or {}
    if skip:
        pairs = ", ".join(f"{key}={value}" for key, value in sorted(skip.items()))
        print(f"  Skips     : {pairs}")

    errors = report.get("errors") or {}
    if errors:
        pairs = ", ".join(f"{key}={value}" for key, value in sorted(errors.items()))
        print(f"  Erreurs   : {pairs}")

    print()
    survivors = report.get("interesting_survivors") or []
    if survivors:
        headers = ["session", "status", "tf", "family", "robust", "alive", "src_ret", "avg_ret", "tokens"]
        rows = []
        for survivor in survivors[:20]:
            source = next(
                (
                    item
                    for item in report.get("top_survivors") or []
                    if item.get("session_id") == survivor.get("session_id")
                ),
                {},
            )
            rows.append(
                [
                    str(survivor.get("session_id") or "")[:46],
                    str(survivor.get("status") or ""),
                    str(survivor.get("timeframe") or ""),
                    str(survivor.get("strategy_id") or "non_canonical"),
                    f"{int(survivor.get('robust_count') or 0)}/{int(survivor.get('tested') or 0)}",
                    f"{int(survivor.get('alive_count') or 0)}/{int(survivor.get('tested') or 0)}",
                    _format_float(source.get("source_return_pct"), 2),
                    _format_float(survivor.get("avg_return"), 2),
                    ",".join((survivor.get("robust_tokens") or [])[:4]),
                ]
            )
        print(_render_table(headers, rows))
    else:
        print("  Aucun survivant au seuil demande.")

    if promoted:
        print()
        print(f"  Catalogues: {len(promoted)} entree(s) promues")


def run_cross_token_command(args) -> int:
    from . import formatters as _formatters

    if getattr(args, "no_color", False):
        _formatters.Colors.disable()

    previous_disable = logging.root.manager.disable
    if not getattr(args, "verbose", False):
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

    try:
        sandbox_root = Path(getattr(args, "sandbox_root", "sandbox_strategies"))
        if not sandbox_root.exists():
            print(f"❌ Sandbox introuvable: {sandbox_root}")
            return 1

        statuses = _split_multi_args(getattr(args, "status", None)) or list(DEFAULT_STATUSES)
        session_ids = _split_multi_args(getattr(args, "session_id", None))
        strategy_ids = _split_multi_args(getattr(args, "strategy_id", None))
        timeframe_filters = _split_multi_args(getattr(args, "timeframe_filter", None))
        tokens = [token.upper() for token in (_split_multi_args(getattr(args, "tokens", None)) or list(DEFAULT_LIQUID_TOKENS))]
        extra_catalog_tags = _split_multi_args(getattr(args, "catalog_tag", None))

        manifest = collect_builder_candidates(
            sandbox_root=sandbox_root,
            statuses=statuses,
            session_ids=session_ids,
            strategy_ids=strategy_ids,
            timeframe_filters=timeframe_filters,
            tokens=tokens,
            min_basket_size=int(getattr(args, "min_basket_size", 3) or 3),
            max_candidates=getattr(args, "max_candidates", None),
        )

        evaluation = evaluate_cross_token_candidates(
            candidates=manifest["candidates"],
            capital=float(getattr(args, "capital", 10000.0) or 10000.0),
            fees_bps=int(getattr(args, "fees_bps", 10) or 10),
            slippage_bps=getattr(args, "slippage_bps", None),
            chunk_size=int(getattr(args, "chunk_size", DEFAULT_CHUNK_SIZE) or 0),
        )

        report = build_cross_token_report(
            results=evaluation["results"],
            skip_reasons=manifest["skip_reasons"],
            errors=evaluation["errors"],
            min_robust_count=int(getattr(args, "min_robust_count", 2) or 2),
            min_robust_ratio=float(getattr(args, "min_robust_ratio", 0.25) or 0.25),
            top=int(getattr(args, "top", 20) or 20),
            sandbox_root=sandbox_root,
            data_dir=Path(str(evaluation["data_dir"])),
        )

        output_path = _resolve_output_path(getattr(args, "output", None))
        _json_dump(report, output_path)

        promoted: List[Dict[str, Any]] = []
        if getattr(args, "promote", False):
            promoted = promote_cross_token_survivors(
                report,
                target_category=str(getattr(args, "catalog_category", "p2_cross_token_survivors")),
                extra_tags=extra_catalog_tags,
            )

        if getattr(args, "json", False):
            console_payload = dict(report)
            console_payload.pop("results", None)
            print(json.dumps(console_payload, indent=2, ensure_ascii=False, default=str))
        elif not getattr(args, "quiet", False):
            _print_report_text(report, output_path, promoted)

        return 0
    finally:
        logging.disable(previous_disable)
