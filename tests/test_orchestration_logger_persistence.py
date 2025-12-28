"""Tests de persistance JSONL pour OrchestrationLogger.

But: éviter un trace.jsonl qui ne contient que le header.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_orchestration_logger_writes_events_to_jsonl(tmp_path: Path) -> None:
    from agents.orchestration_logger import OrchestrationLogger

    save_path = tmp_path / "trace.jsonl"
    logger = OrchestrationLogger(session_id="testsession", auto_save=False, save_path=save_path)

    # Ajouter un événement via l'API générique utilisée par Orchestrator
    logger.log("run_start", {"event_type": "run_start", "session_id": "testsession", "timestamp": "2025-01-01T00:00:00"})
    logger.log("error", {"event_type": "error", "session_id": "testsession", "message": "boom", "timestamp": "2025-01-01T00:00:01"})

    logger.save_to_jsonl(save_path)

    assert save_path.exists()
    lines = save_path.read_text(encoding="utf-8").splitlines()

    # 1 header + au moins 2 événements
    assert len(lines) >= 3

    header = json.loads(lines[0])
    assert header.get("event_type") == "session_header"

    evt1 = json.loads(lines[1])
    evt2 = json.loads(lines[2])
    # Format attendu: OrchestrationLogEntry sérialisé
    assert evt1.get("action_type") == "run_start"
    assert evt2.get("action_type") in {"error", "warning"}
    assert "details" in evt1
    assert "details" in evt2


def test_create_orchestrator_with_backtest_uses_injected_logger_and_session_id(monkeypatch) -> None:
    """Vérifie que l'orchestrateur multi-agents réutilise le logger UI (session_id cohérent)."""

    from agents.llm_client import LLMConfig, LLMProvider
    import agents.orchestrator as orchestrator_module

    class _DummyLLMClient:
        def __init__(self, config: LLMConfig):
            self.config = config

        def is_available(self) -> bool:  # pragma: no cover
            return True

    def _fake_create_llm_client(config: LLMConfig):
        return _DummyLLMClient(config)

    monkeypatch.setattr(orchestrator_module, "create_llm_client", _fake_create_llm_client)

    class _FakeOrchLogger:
        def __init__(self, session_id: str):
            self.session_id = session_id

        def log(self, action_type: str, details: dict) -> None:  # pragma: no cover
            pass

    fake_logger = _FakeOrchLogger(session_id="ui_session_123")

    from agents.integration import create_orchestrator_with_backtest

    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [100, 110, 120],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D")
    )

    orchestrator = create_orchestrator_with_backtest(
        strategy_name="ema_cross",
        data=df,
        initial_params={"fast_period": 10, "slow_period": 21},
        llm_config=LLMConfig(provider=LLMProvider.OLLAMA, model="dummy"),
        use_walk_forward=True,
        orchestration_logger=fake_logger,
        max_iterations=1,
    )

    assert orchestrator.session_id == "ui_session_123"
    assert getattr(orchestrator, "_orch_logger", None) is fake_logger
