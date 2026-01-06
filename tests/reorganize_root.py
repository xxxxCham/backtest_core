"""Réorganisation automatique de la racine du dépôt.

Usage :
    python tools/reorganize_root.py           # dry-run (par défaut)
    python tools/reorganize_root.py --execute # applique les déplacements

Le script applique la cartographie décrite dans docs/ROOT_REORGANISATION.md.
Il n'est **pas exécuté automatiquement** : lancez-le manuellement depuis votre poste après vérification de vos sauvegardes.
Il ignore les entrées inexistantes et protège les fichiers déjà présents en créant
un backup horodaté en cas de collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class MoveRule:
    patterns: tuple[str, ...]
    destination: Path
    description: str

    def matches(self, name: str) -> bool:
        return any(fnmatch(name, pattern) for pattern in self.patterns)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATIONS = {
    "docs/legacy/",
    "docs/performance/",
    "docs/guides/",
    "config/",
    "scripts/windows/",
    "sandbox/tests_manuel/",
    "artifacts/reports/",
    "artifacts/logs/",
    "archive/trash/",
    ".cache/",
}

RULES: List[MoveRule] = [
    MoveRule(
        patterns=(
            "CRITICAL_FIXES_2025-12-16.md",
            "STRUCTURE_ANALYSIS.md",
            "RESTRUCTURATION_REPORT.md",
            "BILAN_PROGRESSION.md",
            "LIVRABLE_FINAL_ANALYSE.md",
        ),
        destination=Path("docs/legacy"),
        description="Rapports historiques",
    ),
    MoveRule(
        patterns=(
            "PERFORMANCE_INDEX.md",
            "PERFORMANCE_QUICKSTART.md",
            "PERFORMANCE_SUMMARY.md",
            "PERFORMANCE_REPORT.md",
            "PERFORMANCE_OPTIMIZATIONS.md",
        ),
        destination=Path("docs/performance"),
        description="Documentation performance",
    ),
    MoveRule(
        patterns=(
            "TEMPLATES_SYSTEM.md",
            "PYDANTIC_REFACTORING.md",
            "LLM_INTEGRATION_README.md",
            "STORAGE_README.md",
            "DOCUMENTATION_SUMMARY.md",
            "ENVIRONMENT.md",
            "GPU_DECISION_GUIDE.md",
            "PYTEST_WATCH_GUIDE.md",
        ),
        destination=Path("docs/guides"),
        description="Guides techniques",
    ),
    MoveRule(
        patterns=(
            "requirements.txt",
            "requirements-gpu.txt",
            "pyproject.toml",
            ".env.example",
            "pyrightconfig.json",
            ".pytest-watch.cfg",
            "pytest-watch.ini",
            "backtest_core.code-workspace",
        ),
        destination=Path("config"),
        description="Configurations et workspace",
    ),
    MoveRule(
        patterns=("*.ps1", "*.bat", "*.lnk"),
        destination=Path("scripts/windows"),
        description="Scripts Windows/PowerShell",
    ),
    MoveRule(
        patterns=("run_tests.py", "test_*.py", "debug_*.py"),
        destination=Path("sandbox/tests_manuel"),
        description="Tests manuels et scripts de debug",
    ),
    MoveRule(
        patterns=(
            "backtest_results.*",
            "rapport_sweep.*",
            "sweep_*",
            "report_ema.html",
            "profile.pstats",
            "sharpe_coverage_summary.txt",
        ),
        destination=Path("artifacts/reports"),
        description="Rapports et artefacts",
    ),
    MoveRule(
        patterns=("*.log",),
        destination=Path("artifacts/logs"),
        description="Logs d'exécution",
    ),
    MoveRule(
        patterns=("nul",),
        destination=Path("archive/trash"),
        description="Entrées invalides",
    ),
    MoveRule(
        patterns=(
            ".coverage",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "__pycache__",
            "htmlcov",
            "_runs",
            "backtest_results",
            ".venv",
        ),
        destination=Path(".cache"),
        description="Caches ou artefacts à ignorer",
    ),
]


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_move(source: Path, destination_dir: Path, *, dry_run: bool = True) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if dry_run:
        print(f"[DRY-RUN] {source.name} -> {destination_dir}/")
        return

    if target.exists():
        backup = target.with_name(f"{target.name}.bak_{timestamp}")
        target.replace(backup)
        print(f"  Backup créé : {backup}")

    source.replace(target)
    print(f"  Déplacé : {source} -> {target}")


def apply_rules(items: Iterable[Path], *, dry_run: bool = True) -> int:
    moved = 0
    for item in items:
        # ne pas toucher aux fichiers du contrôle de version ou aux répertoires internes
        if item.name in {".git", "tools", "docs"}:
            continue
        for rule in RULES:
            if rule.matches(item.name):
                safe_move(item, ROOT / rule.destination, dry_run=dry_run)
                moved += 1
                break
    return moved


def main() -> None:
    ensure_directories(DEFAULT_DESTINATIONS)
    dry_run = "--execute" not in {arg.lower() for arg in __import__("sys").argv[1:]}
    items = list(ROOT.iterdir())
    moved = apply_rules(items, dry_run=dry_run)
    mode = "DRY-RUN" if dry_run else "APPLIQUÉ"
    print(f"\nRésumé ({mode}) : {moved} élément(s) concerné(s) sur {len(items)} entrées racine.")


if __name__ == "__main__":
    main()
