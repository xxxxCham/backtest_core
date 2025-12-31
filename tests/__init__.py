"""
Module-ID: backtest_core

Purpose: Package racine - orchestration globale du moteur de backtest (engine, strategies, indicators, utils, ui).

Role in pipeline: package initialization

Key components: Version, imports, package structure

Inputs: Modules enfants (agents, backtest, cli, data, indicators, strategies, ui, utils)

Outputs: Namespace unifié backtest_core

Dependencies: Python 3.9+, librairies externes (numpy, pandas, streamlit, etc.)

Conventions: Package structure; version sémantique

Read-if: Modification structure package ou version.

Skip-if: Vous modifiez juste les stratégies ou indicateurs.
"""

__version__ = "1.0.0"
__author__ = "ThreadX Framework"
