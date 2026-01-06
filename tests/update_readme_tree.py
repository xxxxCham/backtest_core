#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

START_MARKER = "<!-- TREE:START -->"
END_MARKER = "<!-- TREE:END -->"

DEFAULT_LIMIT = 1
DETAIL_LIMIT = 2
DETAIL_DIRS = {
    "agents",
    "backtest",
    "cli",
    "config",
    "indicators",
    "performance",
    "strategies",
    "t_core",
    "templates",
    "tools",
    "ui",
    "utils",
}
EXTRA_LIMITS = {
    "ui/components": 3,
}

HIDE_NAMES = {
    "__pycache__",
    "backtest_core.egg-info",
    "backtest_results",
    "desktop.ini",
    "logs",
    "nul",
    "profile.pstats",
    "profiling_results",
    "runs",
}


def _limit_for(rel_path: str) -> int:
    if rel_path in EXTRA_LIMITS:
        return EXTRA_LIMITS[rel_path]
    if rel_path in DETAIL_DIRS:
        return DETAIL_LIMIT
    return DEFAULT_LIMIT


def _should_hide(entry: Path) -> bool:
    name = entry.name
    if name.startswith("."):
        return True
    return name in HIDE_NAMES


def _sorted_entries(path: Path) -> list[Path]:
    entries = [p for p in path.iterdir() if not _should_hide(p)]
    entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    return entries


def _build_tree_lines(path: Path, rel_path: str, prefix: str, depth: int) -> list[str]:
    entries = _sorted_entries(path)
    lines: list[str] = []
    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1
        connector = "\\--" if is_last else "|--"
        name = f"{entry.name}/" if entry.is_dir() else entry.name
        lines.append(f"{prefix}{connector} {name}")

        if entry.is_dir():
            child_rel = entry.name if not rel_path else f"{rel_path}/{entry.name}"
            child_limit = _limit_for(child_rel)
            if depth + 1 < child_limit:
                next_prefix = f"{prefix}{'    ' if is_last else '|   '}"
                lines.extend(_build_tree_lines(entry, child_rel, next_prefix, depth + 1))
    return lines


def _render_tree() -> str:
    lines = [f"{ROOT.name}/"]
    lines.extend(_build_tree_lines(ROOT, "", "", 0))
    return "\n".join(lines)


def _update_readme(tree_text: str) -> None:
    text = README.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise SystemExit("Missing README tree markers.")

    newline = "\r\n" if "\r\n" in text else "\n"
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    block = (
        f"{START_MARKER}{newline}"
        f"```{newline}"
        f"{tree_text}{newline}"
        f"```{newline}"
        f"{END_MARKER}"
    )
    README.write_text(f"{before}{block}{after}", encoding="utf-8", newline=newline)


def main() -> None:
    tree_text = _render_tree()
    _update_readme(tree_text)


if __name__ == "__main__":
    main()
