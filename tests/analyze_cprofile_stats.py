from __future__ import annotations

import argparse
import pstats
from pathlib import Path
from typing import Dict, List, Tuple


def _load_entries(stats: pstats.Stats, filter_text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (filename, lineno, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        if filter_text:
            if filter_text not in filename and filter_text not in func:
                continue
        rows.append(
            {
                "file": filename,
                "line": lineno,
                "func": func,
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
            }
        )
    return rows


def _summarize_by_key(
    rows: List[Dict[str, object]],
    key_fn,
) -> List[Tuple[str, Dict[str, float]]]:
    totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        key = key_fn(row)
        entry = totals.setdefault(
            key,
            {"total_time": 0.0, "cumulative_time": 0.0, "calls": 0.0},
        )
        entry["total_time"] += float(row["total_time"])
        entry["cumulative_time"] += float(row["cumulative_time"])
        entry["calls"] += float(row["calls"])
    return list(totals.items())


def _format_rows(rows: List[Dict[str, object]], top_n: int, sort_by: str) -> str:
    sort_key = "cumulative_time" if sort_by == "cumulative" else "total_time"
    ordered = sorted(rows, key=lambda r: float(r[sort_key]), reverse=True)
    lines = []
    lines.append(
        f"{'cumulative(s)':>14} {'total(s)':>10} {'calls':>8}  location"
    )
    lines.append("-" * 70)
    for row in ordered[:top_n]:
        location = f"{row['func']} ({Path(str(row['file'])).name}:{row['line']})"
        lines.append(
            f"{float(row['cumulative_time']):14.4f} "
            f"{float(row['total_time']):10.4f} "
            f"{int(row['calls']):8d}  {location}"
        )
    return "\n".join(lines)


def _format_grouped(
    grouped: List[Tuple[str, Dict[str, float]]],
    top_n: int,
    sort_by: str,
) -> str:
    sort_key = "cumulative_time" if sort_by == "cumulative" else "total_time"
    ordered = sorted(grouped, key=lambda r: r[1][sort_key], reverse=True)
    lines = []
    lines.append(
        f"{'cumulative(s)':>14} {'total(s)':>10} {'calls':>8}  key"
    )
    lines.append("-" * 70)
    for key, values in ordered[:top_n]:
        lines.append(
            f"{values['cumulative_time']:14.4f} "
            f"{values['total_time']:10.4f} "
            f"{int(values['calls']):8d}  {key}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a .pstats file and summarize top costs.",
    )
    parser.add_argument("input")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--sort", choices=["cumulative", "time"], default="cumulative")
    parser.add_argument("--filter", default="")
    parser.add_argument(
        "--group-by",
        choices=["function", "module", "file"],
        default="function",
    )
    parser.add_argument("--strip-dirs", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = pstats.Stats(args.input)
    if args.strip_dirs:
        stats.strip_dirs()

    rows = _load_entries(stats, args.filter)

    header = []
    header.append("PROFILE ANALYSIS")
    header.append("=" * 60)
    header.append(f"input: {args.input}")
    header.append(f"functions: {len(rows)}")
    header.append(f"total_time_sec: {stats.total_tt:.4f}")
    header.append("")

    output_lines = header

    if args.group_by == "function":
        output_lines.append("Top functions")
        output_lines.append(_format_rows(rows, args.top, args.sort))
    else:
        if args.group_by == "module":
            grouped = _summarize_by_key(
                rows, lambda r: Path(str(r["file"])).name
            )
        else:
            grouped = _summarize_by_key(rows, lambda r: str(r["file"]))
        output_lines.append(f"Top {args.group_by}s")
        output_lines.append(_format_grouped(grouped, args.top, args.sort))

    output = "\n".join(output_lines)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
