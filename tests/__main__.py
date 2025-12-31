"""
Module-ID: __main__

Purpose: Point d'entrée CLI du moteur de backtest - dispatcher de commandes (list, info, backtest, sweep, validate, export).

Role in pipeline: entry point

Key components: _run(), dispatch, CLI parser

Inputs: Arguments sys.argv (--strategy, --data, --params, etc.)

Outputs: Exécution commandes, messages stdout

Dependencies: sys, pathlib, cli.commands

Conventions: Commandes via `python -m backtest_core [CMD]`

Read-if: Modification entry point ou ajout nouvelles commandes CLI.

Skip-if: Développement stratégies ou indicateurs (pas besoin CLI).
"""

import sys
from pathlib import Path


def _run() -> int:
    # Ajouter le répertoire courant au path pour les imports
    sys.path.insert(0, str(Path(__file__).parent))

    from cli import main  # pylint: disable=import-outside-toplevel

    return int(main())


if __name__ == "__main__":
    raise SystemExit(_run())
