"""
Module-ID: tests.conftest

Purpose: Fixtures pytest partagées.

Role in pipeline: testing

Key components: logger()

Inputs: tmp_path fixture

Outputs: OrchestrationLogger isolé (save_path temporaire)

Dependencies: pytest, agents.orchestration_logger

Conventions: Évite l'écriture dans le repo; une session_id stable.

Read-if: Un test demande une fixture globale.

Skip-if: Tests n'utilisent pas de fixtures custom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.orchestration_logger import OrchestrationLogger


@pytest.fixture
def logger(tmp_path: Path) -> OrchestrationLogger:
    return OrchestrationLogger(
        session_id="pytest_session",
        auto_save=True,
        save_path=tmp_path / "trace.jsonl",
    )
