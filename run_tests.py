#!/usr/bin/env python
"""
Script pour exécuter tous les tests unitaires.
==============================================

Usage:
    python run_tests.py
    python run_tests.py -v          # Verbose
    python run_tests.py --coverage  # Avec couverture
"""

import subprocess
import sys
from pathlib import Path


def run_tests(verbose: bool = False, coverage: bool = False):
    """Exécute les tests unitaires."""
    project_root = Path(__file__).parent

    # Construire la commande
    if coverage:
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=backtest",
            "--cov=indicators",
            "--cov=strategies",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
        ]
    else:
        cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    cmd.append("tests/")

    print(f"🧪 Exécution des tests: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=project_root)

    return result.returncode


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    coverage = "--coverage" in sys.argv or "-c" in sys.argv

    exit_code = run_tests(verbose=verbose, coverage=coverage)
    sys.exit(exit_code)
