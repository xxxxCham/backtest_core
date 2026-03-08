from cli.cross_token import (
    _available_basket,
    build_cross_token_report,
    resolve_unique_timeframe,
    select_best_iteration,
)


def test_resolve_unique_timeframe_detects_unique_and_ambiguous():
    unique_tf, unique_state = resolve_unique_timeframe(
        {"objective": "Strategie momentum BTCUSDC 1h"}
    )
    ambiguous_tf, ambiguous_state = resolve_unique_timeframe(
        {"objective": "Strategie multi-timeframe 1h et 4h"}
    )

    assert unique_tf == "1h"
    assert unique_state == "unique"
    assert ambiguous_tf is None
    assert ambiguous_state == "ambiguous"


def test_select_best_iteration_falls_back_to_strategy_py(tmp_path):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    (session_dir / "strategy.py").write_text("class BuilderGeneratedStrategy:\n    pass\n", encoding="utf-8")

    summary = {
        "total_iterations": 2,
        "iterations": [
            {"iteration": 1, "return_pct": -10.0, "sharpe": 0.2, "trades": 12},
            {"iteration": 2, "return_pct": 35.0, "sharpe": 1.1, "trades": 48},
        ],
    }

    selected = select_best_iteration(session_dir, summary)

    assert selected is not None
    assert selected["iteration"] == 2
    assert selected["strategy_path"].endswith("strategy.py")


def test_build_cross_token_report_flags_interesting_survivors(tmp_path):
    report = build_cross_token_report(
        results=[
            {
                "session_id": "sess-a",
                "status": "success",
                "timeframe": "1h",
                "strategy_id": "trend_supertrend",
                "source_symbol": "AAVEUSDC",
                "best_iteration": 1,
                "source_metrics": {"return_pct": 228.5, "sharpe": 1.07},
                "tested": 12,
                "alive_count": 9,
                "robust_count": 4,
                "alive_ratio": 0.75,
                "robust_ratio": 4 / 12,
                "avg_return": 1078.11,
                "token_results": [
                    {"token": "SOLUSDC", "alive": True, "robust": True},
                    {"token": "LINKUSDC", "alive": True, "robust": True},
                ],
            },
            {
                "session_id": "sess-b",
                "status": "max_iterations",
                "timeframe": "1h",
                "strategy_id": "momentum_macd",
                "source_symbol": "BTCUSDC",
                "best_iteration": 5,
                "source_metrics": {"return_pct": 109.3, "sharpe": 0.93},
                "tested": 12,
                "alive_count": 3,
                "robust_count": 1,
                "alive_ratio": 0.25,
                "robust_ratio": 1 / 12,
                "avg_return": -42.69,
                "token_results": [
                    {"token": "BNBUSDC", "alive": True, "robust": True},
                ],
            },
        ],
        skip_reasons={"timeframe_missing": 2},
        errors={"load_strategy": 0, "run_backtest": 1},
        min_robust_count=2,
        min_robust_ratio=0.25,
        top=10,
        sandbox_root=tmp_path / "sandbox_strategies",
        data_dir=tmp_path / "data",
    )

    assert report["evaluated"] == 2
    assert report["interesting_count"] == 1
    assert report["interesting_survivors"][0]["session_id"] == "sess-a"
    assert report["top_survivors"][0]["session_id"] == "sess-a"


def test_available_basket_ignores_invalid_datasets(tmp_path, monkeypatch):
    (tmp_path / "GOODUSDC_1h.parquet").write_text("", encoding="utf-8")
    (tmp_path / "BADUSDC_1h.parquet").write_text("", encoding="utf-8")

    def fake_is_usable_dataset(symbol: str, timeframe: str) -> bool:
        return symbol == "GOODUSDC" and timeframe == "1h"

    monkeypatch.setattr("data.loader.is_usable_dataset", fake_is_usable_dataset)

    basket = _available_basket(["GOODUSDC", "BADUSDC"], "1h", tmp_path)

    assert basket == ["GOODUSDC"]
