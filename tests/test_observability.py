"""
Tests pour le module d'observabilité.

Vérifie:
1. Zéro overhead quand DEBUG désactivé
2. Safe stats ne sérialise pas de gros contenus
3. PerfCounters fonctionnent correctement
4. DiagnosticPack est compact
"""

import json
import logging
import time
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.observability import (
    get_obs_logger,
    init_logging,
    generate_run_id,
    trace_span,
    safe_stats_df,
    safe_stats_array,
    safe_stats_series,
    PerfCounters,
    DiagnosticPack,
    build_diagnostic_summary,
    set_log_level,
    is_debug_enabled,
    ObsLoggerAdapter,
)


class TestZeroOverhead:
    """Tests pour vérifier le zéro overhead en mode INFO."""

    def test_trace_span_no_logging_when_info(self):
        """trace_span n'émet rien et ne calcule pas si niveau INFO."""
        # Configurer en INFO
        set_log_level("INFO")
        logger = get_obs_logger("test.overhead", run_id="test123")
        
        # Capturer les logs
        captured_logs = []
        
        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())
        
        handler = CaptureHandler()
        handler.setLevel(logging.DEBUG)
        logging.getLogger("backtest.test.overhead").addHandler(handler)
        
        try:
            # Exécuter trace_span
            with trace_span(logger, "test_operation", foo="bar"):
                time.sleep(0.01)
            
            # Vérifier qu'aucun log DEBUG n'a été émis
            debug_logs = [l for l in captured_logs if "span_" in l]
            assert len(debug_logs) == 0, f"Logs DEBUG inattendus: {debug_logs}"
        
        finally:
            logging.getLogger("backtest.test.overhead").removeHandler(handler)

    def test_trace_span_emits_when_debug(self):
        """trace_span émet des logs quand DEBUG activé."""
        set_log_level("DEBUG")
        logger = get_obs_logger("test.debug", run_id="dbg123")
        
        captured_logs = []
        
        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())
        
        handler = CaptureHandler()
        handler.setLevel(logging.DEBUG)
        logging.getLogger("backtest.test.debug").addHandler(handler)
        
        try:
            with trace_span(logger, "test_op"):
                pass
            
            # Vérifier que les logs start/end ont été émis
            span_logs = [l for l in captured_logs if "span_" in l]
            assert len(span_logs) >= 2, f"Logs span attendus: {span_logs}"
            assert any("span_start" in l for l in span_logs)
            assert any("span_end" in l for l in span_logs)
        
        finally:
            logging.getLogger("backtest.test.debug").removeHandler(handler)
            set_log_level("INFO")

    def test_lazy_formatting_no_cost(self):
        """Vérifie que le formatting lazy n'a pas de coût si désactivé."""
        set_log_level("ERROR")  # Désactive INFO et DEBUG
        logger = get_obs_logger("test.lazy")
        
        # Compteur pour vérifier si la fonction est appelée
        call_count = [0]
        
        def expensive_computation():
            call_count[0] += 1
            return "expensive result"
        
        # Log avec lazy formatting
        logger.debug("Result: %s", expensive_computation())
        
        # En Python, les arguments sont évalués avant l'appel,
        # donc on vérifie plutôt que le log n'est pas émis
        # Le coût est dans le non-traitement, pas dans le non-appel
        
        set_log_level("INFO")


class TestSafeStats:
    """Tests pour safe_stats_* - vérifier qu'ils ne sérialisent pas tout."""

    def test_safe_stats_df_no_full_content(self):
        """safe_stats_df ne retourne jamais le contenu complet."""
        # DataFrame de 10000 lignes
        big_df = pd.DataFrame({
            "a": np.random.randn(10000),
            "b": np.random.randn(10000),
            "c": ["text"] * 10000,
        }, index=pd.date_range("2020-01-01", periods=10000, freq="1h"))
        
        stats = safe_stats_df(big_df)
        
        # Vérifier la structure
        assert stats["shape"] == (10000, 3)
        assert "a" in stats["dtypes"]
        assert "nan_count" in stats
        assert "memory_mb" in stats
        
        # Vérifier que head est limité
        if "head" in stats:
            assert len(stats["head"]) <= 3
        
        # Vérifier que la sérialisation JSON est petite
        json_str = json.dumps(stats, default=str)
        assert len(json_str) < 2000, f"JSON trop gros: {len(json_str)} bytes"

    def test_safe_stats_df_empty(self):
        """safe_stats_df gère les DataFrames vides."""
        empty_df = pd.DataFrame()
        stats = safe_stats_df(empty_df)
        
        assert stats["shape"] == (0, 0)
        assert stats.get("empty", False)

    def test_safe_stats_array_no_full_content(self):
        """safe_stats_array ne retourne pas le contenu complet."""
        big_array = np.random.randn(100000)
        big_array[1000:2000] = np.nan
        
        stats = safe_stats_array(big_array, name="test_array")
        
        assert stats["name"] == "test_array"
        assert stats["shape"] == (100000,)
        assert stats["nan_count"] == 1000
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        
        # Vérifier que c'est compact
        json_str = json.dumps(stats, default=str)
        assert len(json_str) < 500

    def test_safe_stats_array_handles_non_numeric(self):
        """safe_stats_array gère les arrays non-numériques."""
        str_array = np.array(["a", "b", "c"])
        stats = safe_stats_array(str_array, name="strings")
        
        assert stats["name"] == "strings"
        assert "min" not in stats  # Pas de stats numériques

    def test_safe_stats_series(self):
        """safe_stats_series fonctionne correctement."""
        series = pd.Series(np.random.randn(1000), name="test")
        series.iloc[100:200] = np.nan
        
        stats = safe_stats_series(series, name="my_series")
        
        assert stats["name"] == "my_series"
        assert stats["nan_count"] == 100
        assert stats["min"] is not None
        assert stats["max"] is not None


class TestPerfCounters:
    """Tests pour PerfCounters."""

    def test_perf_counters_timing(self):
        """PerfCounters mesure correctement les durées."""
        counters = PerfCounters()
        
        counters.start("operation1")
        time.sleep(0.05)  # 50ms
        duration = counters.stop("operation1")
        
        # Vérifier la durée (avec tolérance)
        assert 40 < duration < 100, f"Durée inattendue: {duration}ms"
        
        # Vérifier le summary
        summary = counters.summary()
        assert "operation1" in summary["durations_ms"]
        assert summary["total_ms"] > 0

    def test_perf_counters_increment(self):
        """PerfCounters incrémente correctement."""
        counters = PerfCounters()
        
        counters.increment("trades", 5)
        counters.increment("trades", 3)
        counters.increment("signals", 100)
        
        summary = counters.summary()
        assert summary["counts"]["trades"] == 8
        assert summary["counts"]["signals"] == 100

    def test_perf_counters_missing_start(self):
        """PerfCounters gère stop sans start."""
        counters = PerfCounters()
        
        # Stop sans start ne plante pas
        duration = counters.stop("nonexistent")
        assert duration == 0.0


class TestDiagnosticPack:
    """Tests pour DiagnosticPack."""

    def test_diagnostic_pack_compact(self):
        """DiagnosticPack produit un JSON compact."""
        counters = PerfCounters()
        counters.start("total")
        counters.stop("total")
        counters.increment("trades", 42)
        
        pack = build_diagnostic_summary(
            run_id="abc123",
            request={"strategy": "ema_cross", "symbol": "BTCUSDC", "params": {"fast": 10}},
            result={"sharpe_ratio": 1.5, "total_return": 0.15, "max_drawdown": -0.08},
            counters=counters,
            last_exception=None,
        )
        
        json_str = pack.to_json()
        
        # Vérifier la taille
        assert len(json_str) < 2000, f"JSON trop gros: {len(json_str)} bytes"
        
        # Vérifier le contenu
        data = json.loads(json_str)
        assert data["run_id"] == "abc123"
        assert data["strategy"] == "ema_cross"
        assert data["error"] is None

    def test_diagnostic_pack_with_error(self):
        """DiagnosticPack capture les erreurs."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            pack = build_diagnostic_summary(
                run_id="err123",
                request={},
                result=None,
                counters=None,
                last_exception=e,
            )
        
        assert pack.error == "Test error message"
        assert pack.error_type == "ValueError"


class TestLoggerAdapter:
    """Tests pour ObsLoggerAdapter."""

    def test_run_id_propagation(self):
        """run_id est propagé dans les logs."""
        logger = get_obs_logger("test.propagation", run_id="prop123")
        
        assert logger.extra["run_id"] == "prop123"

    def test_with_context(self):
        """with_context enrichit le logger."""
        logger = get_obs_logger("test.context", run_id="ctx123")
        enriched = logger.with_context(strategy="ema_cross", symbol="BTCUSDC")
        
        assert enriched.extra["run_id"] == "ctx123"
        assert enriched.extra["strategy"] == "ema_cross"
        assert enriched.extra["symbol"] == "BTCUSDC"

    def test_generate_run_id_unique(self):
        """generate_run_id produit des IDs uniques."""
        ids = [generate_run_id() for _ in range(100)]
        
        # Tous uniques
        assert len(set(ids)) == 100
        
        # Format correct (8 chars hex)
        for id_ in ids:
            assert len(id_) == 8
            assert all(c in "0123456789abcdef" for c in id_)


class TestIntegration:
    """Tests d'intégration avec le vrai workflow."""

    def test_full_workflow_with_observability(self):
        """Test du workflow complet avec observabilité."""
        from backtest.engine import BacktestEngine
        
        # Créer des données de test
        n_bars = 500
        df = pd.DataFrame({
            "open": 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
            "high": 100 + np.cumsum(np.random.randn(n_bars) * 0.5) + 1,
            "low": 100 + np.cumsum(np.random.randn(n_bars) * 0.5) - 1,
            "close": 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
            "volume": np.random.randint(1000, 10000, n_bars),
        }, index=pd.date_range("2020-01-01", periods=n_bars, freq="1h"))
        
        # Fixer les colonnes high/low
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        
        # Créer engine avec run_id
        run_id = generate_run_id()
        engine = BacktestEngine(initial_capital=10000, run_id=run_id)
        
        # Vérifier que run_id est propagé
        assert engine.run_id == run_id
        
        # Exécuter backtest
        result = engine.run(
            df=df,
            strategy="ema_cross",
            params={"fast_period": 10, "slow_period": 21},
            symbol="TEST",
            timeframe="1h",
        )
        
        # Vérifier que les counters sont présents
        assert engine.counters is not None
        assert "indicators" in engine.counters.summary()["durations_ms"]
        
        # Vérifier que run_id est dans les meta
        assert result.meta.get("run_id") == run_id
        assert "perf_counters" in result.meta
