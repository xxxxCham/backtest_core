"""
Backtest Core - Point d'entrée CLI
==================================

Usage:
    python -m backtest_core [COMMANDE] [OPTIONS]

Commandes disponibles:
    list        Lister stratégies, indicateurs ou données
    info        Informations détaillées sur une ressource
    backtest    Exécuter un backtest
    sweep       Optimisation paramétrique
    validate    Valider configuration
    export      Exporter résultats

Exemples:
    python -m backtest_core list strategies
    python -m backtest_core info strategy bollinger_dual
    python -m backtest_core backtest --strategy ema_cross --data data/BTCUSDT_1h.parquet
"""

import sys
from pathlib import Path

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from cli import main

if __name__ == "__main__":
    sys.exit(main())
