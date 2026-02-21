"""
Tests pour le Strategy Builder (agents/strategy_builder.py).

Couvre :
- Validation du code généré (syntaxe, sécurité, structure)
- Création de session (ID, dossier)
- Extraction JSON/Python depuis réponses LLM
- Chargement dynamique de stratégie
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.strategy_builder import (
    GENERATED_CLASS_NAME,
    SANDBOX_ROOT,
    BuilderIteration,
    BuilderSession,
    StrategyBuilder,
    _build_deterministic_fallback_code,
    _validate_llm_logic_block,
    _repair_code,
    _is_accept_candidate,
    _policy_change_type_override,
    _ranking_sharpe,
    _extract_json_from_response,
    _extract_python_from_response,
    normalize_variant_for_builder,
    recommend_market_context,
    sanitize_objective_text,
    validate_generated_code,
)


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def valid_strategy_code():
    """Code Python valide d'une stratégie générée."""
    return textwrap.dedent(f"""\
        from typing import Any, Dict, List
        import numpy as np
        import pandas as pd
        from strategies.base import StrategyBase
        from utils.parameters import ParameterSpec

        class {GENERATED_CLASS_NAME}(StrategyBase):
            \"\"\"Stratégie générée par le builder.\"\"\"

            def __init__(self):
                super().__init__(name="TestBuilder")

            @property
            def required_indicators(self) -> List[str]:
                return ["rsi", "atr"]

            @property
            def default_params(self) -> Dict[str, Any]:
                return {{"rsi_period": 14, "atr_period": 14}}

            @property
            def parameter_specs(self) -> Dict[str, ParameterSpec]:
                return {{
                    "rsi_period": ParameterSpec(
                        name="rsi_period", min_val=5, max_val=30,
                        default=14, param_type="int",
                    ),
                }}

            def generate_signals(
                self, df: pd.DataFrame,
                indicators: Dict[str, Any],
                params: Dict[str, Any],
            ) -> pd.Series:
                n = len(df)
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                rsi = indicators.get("rsi")
                if rsi is not None:
                    signals[rsi < 30] = 1.0
                    signals[rsi > 70] = -1.0
                return signals
    """)


@pytest.fixture
def sample_ohlcv():
    """DataFrame OHLCV minimal pour tests."""
    n = 200
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


# ─── Tests validate_generated_code ───────────────────────────────────────

class TestValidateCode:
    """Tests de la fonction validate_generated_code."""

    def test_valid_code(self, valid_strategy_code):
        is_valid, msg = validate_generated_code(valid_strategy_code)
        assert is_valid, f"Code devrait être valide: {msg}"

    def test_syntax_error(self):
        code = "class Foo(\n  def bar(self):\n    pass"
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "syntaxe" in msg.lower() or "syntax" in msg.lower()

    def test_missing_class(self):
        code = textwrap.dedent("""\
            import numpy as np
            class WrongName:
                def generate_signals(self, df, indicators, params):
                    pass
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert GENERATED_CLASS_NAME in msg

    def test_missing_generate_signals(self):
        code = textwrap.dedent(f"""\
            class {GENERATED_CLASS_NAME}:
                def some_other_method(self):
                    pass
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "generate_signals" in msg

    def test_reject_nameerror_df_not_defined(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["rsi"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, data, indicators, params):
                    signals = pd.Series(0.0, index=data.index)
                    close = df["close"].values
                    signals[close > 0] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "nameerror" in msg.lower()
        assert "df" in msg.lower()

    def test_reject_nameerror_warmup_not_defined(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["rsi"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    signals.iloc[:warmup] = 0.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "nameerror" in msg.lower()
        assert "warmup" in msg.lower()

    def test_dangerous_os_system(self):
        code = textwrap.dedent(f"""\
            import os
            class {GENERATED_CLASS_NAME}:
                def generate_signals(self, df, indicators, params):
                    os.system("rm -rf /")
                    return df["close"] * 0
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "dangereux" in msg.lower() or "dangerous" in msg.lower()

    def test_dangerous_subprocess(self):
        code = textwrap.dedent(f"""\
            import subprocess
            class {GENERATED_CLASS_NAME}:
                def generate_signals(self, df, indicators, params):
                    subprocess.run(["ls"])
                    return df["close"] * 0
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid

    def test_dangerous_eval(self):
        code = textwrap.dedent(f"""\
            class {GENERATED_CLASS_NAME}:
                def generate_signals(self, df, indicators, params):
                    return eval("df['close']")
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid

    def test_reject_iloc_on_indicator_array(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["rsi"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    for i in range(len(df)):
                        if indicators["rsi"].iloc[i] < 30:
                            signals.iloc[i] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        # May be rejected for iloc or for-range (both are banned patterns)
        assert "iloc" in msg.lower() or "range" in msg.lower()

    def test_reject_unknown_indicator_alias(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["rsi", "bollinger_upper"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    upper = indicators.get("bollinger_upper")
                    rsi = indicators["rsi"]
                    if upper is not None:
                        signals[(rsi > 70) & (df["close"].values > upper)] = -1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "indicateur" in msg.lower() or "inconnu" in msg.lower()

    def test_reject_array_indicator_subkey_access(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["ema"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    ema_21 = np.nan_to_num(indicators["ema"]["ema_21"])
                    signals[df["close"].values > ema_21] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "ndarray" in msg.lower() or "ema" in msg.lower()

    def test_reject_dict_indicator_direct_comparison(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["adx"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    adx = indicators["adx"]
                    signals[adx > 25] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "dict" in msg.lower() and "adx" in msg.lower()

    def test_reject_unknown_supertrend_subkey(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["supertrend"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    st = indicators["supertrend"]
                    upper = np.nan_to_num(st["upper"])
                    signals[df["close"].values > upper] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "supertrend" in msg.lower() and "sous-cl" in msg.lower()

    def test_reject_dict_indicator_any_method(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["adx"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    signals = pd.Series(0.0, index=df.index)
                    adx = indicators["adx"]
                    if adx.any():
                        signals.iloc[:10] = 1.0
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "indicator dict" in msg.lower() or "dict" in msg.lower()

    def test_reject_overwrite_np_alias(self):
        code = textwrap.dedent(f"""\
            from typing import Any, Dict, List
            import numpy as np
            import pandas as pd
            from strategies.base import StrategyBase

            class {GENERATED_CLASS_NAME}(StrategyBase):
                @property
                def required_indicators(self) -> List[str]:
                    return ["rsi"]

                @property
                def default_params(self) -> Dict[str, Any]:
                    return {{}}

                def generate_signals(self, df, indicators, params):
                    np = 1
                    signals = pd.Series(0.0, index=df.index)
                    return signals
        """)
        is_valid, msg = validate_generated_code(code)
        assert not is_valid
        assert "alias réservé `np`".lower() in msg.lower()


# ─── Tests extraction LLM ─────────────────────────────────────────────────

class TestExtractResponse:
    """Tests des helpers d'extraction de réponse LLM."""

    def test_extract_json_from_code_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _extract_json_from_response(text)
        assert result == {"key": "value"}

    def test_extract_json_bare(self):
        text = '{"strategy_name": "test"}'
        result = _extract_json_from_response(text)
        assert result["strategy_name"] == "test"

    def test_extract_json_embedded_in_text(self):
        text = 'Here is the result: {"hypothesis": "test RSI"} and done.'
        result = _extract_json_from_response(text)
        assert result["hypothesis"] == "test RSI"

    def test_extract_json_invalid(self):
        text = "No JSON here, just text."
        result = _extract_json_from_response(text)
        assert result == {}

    def test_extract_python_from_code_block(self):
        text = 'Sure:\n```python\nimport numpy as np\nprint("hello")\n```'
        result = _extract_python_from_response(text)
        assert "import numpy" in result
        assert "print" in result

    def test_extract_python_fallback(self):
        text = "import pandas as pd\ndf = pd.DataFrame()"
        result = _extract_python_from_response(text)
        assert "import pandas" in result


class TestLogicBlockValidation:
    def test_llm_logic_allows_boolean_constants_outside_signals(self):
        logic = textwrap.dedent("""\
            long_prev = np.roll(long_mask, 1)
            long_prev[:1] = False
            long_entry = long_mask & (~long_prev)
            signals[long_entry] = 1.0
        """)
        ok, err = _validate_llm_logic_block(logic)
        assert ok, err

    def test_llm_logic_rejects_true_false_in_signal_assignments(self):
        logic = "signals[long_mask] = True"
        ok, err = _validate_llm_logic_block(logic)
        assert not ok
        assert "true/false" in err.lower()


class TestCodeRepair:
    def test_repair_normalizes_indicator_key_case(self):
        raw = "x = indicators['SMA']\ny = indicators.get('ADX', None)\n"
        repaired = _repair_code(raw)
        assert "indicators['sma']" in repaired
        assert "indicators.get('adx'" in repaired

    def test_repair_rewrites_dict_dot_notation(self):
        raw = "x = donchian.upper\ny = adx.adx\n"
        repaired = _repair_code(raw)
        assert "indicators['donchian']['upper']" in repaired
        assert "indicators['adx']['adx']" in repaired

    def test_repair_rewrites_indicators_close_to_df_close(self):
        raw = "price = indicators['close']\nsl = indicators.get('bb_stop_long', np.nan)\n"
        repaired = _repair_code(raw)
        assert "df['close']" in repaired
        assert "df['bb_stop_long']" in repaired


class _DummyLLMClient:
    def __init__(self, response: str):
        self._response = response

    def chat(self, messages, max_tokens=0):  # noqa: ANN001
        return self._response


class TestMarketRecommendation:
    def test_recommend_market_context_valid(self):
        llm = _DummyLLMClient(
            '{"symbol":"DOGEUSDC","timeframe":"5m","confidence":0.82,"reason":"Scalp court terme."}',
        )
        result = recommend_market_context(
            llm,
            objective="Scalp de continuation agressif",
            candidate_symbols=["BTCUSDC", "DOGEUSDC"],
            candidate_timeframes=["5m", "15m"],
            default_symbol="BTCUSDC",
            default_timeframe="15m",
        )
        assert result["source"] == "llm"
        assert result["symbol"] == "DOGEUSDC"
        assert result["timeframe"] == "5m"
        assert float(result["confidence"]) > 0.8

    def test_recommend_market_context_out_of_universe_fallback(self):
        llm = _DummyLLMClient(
            '{"symbol":"ETHUSDC","timeframe":"1m","confidence":0.9,"reason":"Test"}',
        )
        result = recommend_market_context(
            llm,
            objective="Scalp",
            candidate_symbols=["BTCUSDC", "DOGEUSDC"],
            candidate_timeframes=["5m", "15m"],
            default_symbol="BTCUSDC",
            default_timeframe="15m",
        )
        assert result["source"] == "fallback_out_of_universe"
        assert result["symbol"] == "BTCUSDC"
        assert result["timeframe"] == "15m"

    def test_recommend_market_context_invalid_json_fallback(self):
        llm = _DummyLLMClient("pas de json ici")
        result = recommend_market_context(
            llm,
            objective="Scalp",
            candidate_symbols=["BTCUSDC"],
            candidate_timeframes=["5m"],
            default_symbol="BTCUSDC",
            default_timeframe="5m",
        )
        assert result["source"] == "fallback_invalid_json"
        assert result["symbol"] == "BTCUSDC"
        assert result["timeframe"] == "5m"


# ─── Tests session ─────────────────────────────────────────────────────────

class TestSession:
    """Tests de gestion de session."""

    def test_create_session_id(self):
        sid = StrategyBuilder.create_session_id("Trend BTC 30m Bollinger")
        assert "trend_btc_30m_bollinger" in sid
        # Contient un timestamp
        assert "_" in sid
        parts = sid.split("_")
        assert len(parts) >= 3

    def test_get_session_dir(self):
        sdir = StrategyBuilder.get_session_dir("test_session_123")
        assert sdir == SANDBOX_ROOT / "test_session_123"

    def test_builder_session_defaults(self):
        session = BuilderSession(
            session_id="test",
            objective="Trend following",
            session_dir=Path("/tmp/test"),
        )
        assert session.status == "running"
        assert session.best_sharpe == float("-inf")
        assert session.iterations == []

    def test_builder_iteration_defaults(self):
        it = BuilderIteration(iteration=1)
        assert it.hypothesis == ""
        assert it.error is None
        assert it.decision == ""


class TestObjectiveSanitizer:
    """Tests du nettoyage d'objectif Builder."""

    def test_preserve_clean_objective(self):
        objective = (
            "Scalp de continuation sur DOGEUSDC 5m. "
            "Indicateurs: EMA + RSI + Bollinger. "
            "Entrées pullback EMA21. Sorties ATR."
        )
        assert sanitize_objective_text(objective) == objective

    def test_extract_objective_from_contaminated_logs(self):
        raw = textwrap.dedent("""\
            19:24:49 | INFO | agents.ollama_manager | démarrage
            19:25:00 | INFO | backtest.agents.strategy_builder | strategy_builder_start session=abc objective='19:24:49 | INFO | noise
            19:25:00 | INFO | backtest.agents.strategy_builder | strategy_builder_start session=prev objective='[Scalp de continuation / micro-retournement] sur [crypto liquide] [5m ou 15m].
            Indicateurs : [EMA 9/21/50] + [RSI 14] + [Bandes de Bollinger 20,2].' indicators=31
            Traceback (most recent call last):
              File "D:\\backtest_core\\agents\\strategy_builder.py", line 1
            ' indicators=31
        """)
        cleaned = sanitize_objective_text(raw)
        assert cleaned.startswith("[Scalp de continuation / micro-retournement]")
        assert "strategy_builder_start" not in cleaned
        assert "| INFO |" not in cleaned

    def test_drop_pipe_warning_and_traceback_blob(self):
        raw = textwrap.dedent("""\
            | WARNING | data.loader | Plus gros gap : 2019-05-15 02:30:00+00:00 → 2019-05-15 13:00:00+00:00 (20 barres)
            ────────────────────────── Traceback (most recent call last) ───────────────────────────
            C:\\Program Files\\Python312\\Lib\\site-packages\\streamlit\\runtime\\scriptrunner\\exec_code.py
            StreamlitAPIException: st.session_state.builder_objective_input cannot be modified
        """)
        cleaned = sanitize_objective_text(raw)
        assert cleaned == ""


class TestBuilderRobustnessGate:
    """Tests des garde-fous robustesse pour acceptance/ranking."""

    def test_ranking_penalizes_ruined_metrics(self):
        metrics = {
            "sharpe_ratio": 1.8,
            "total_return_pct": -35000.0,
            "max_drawdown_pct": -100.0,
            "total_trades": 1200,
        }
        assert _ranking_sharpe(metrics) == -20.0

    def test_ranking_penalizes_no_trades(self):
        metrics = {
            "sharpe_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
        }
        assert _ranking_sharpe(metrics) <= -5.0

    def test_accept_candidate_requires_robustness(self):
        metrics = {
            "sharpe_ratio": 1.2,
            "total_return_pct": 10.0,
            "max_drawdown_pct": -30.0,
            "total_trades": 40,
        }
        ok, reason = _is_accept_candidate(metrics, target_sharpe=1.0)
        assert ok is True
        assert reason == "ok"

    def test_accept_candidate_rejects_ruined(self):
        metrics = {
            "sharpe_ratio": 1.5,
            "total_return_pct": -200.0,
            "max_drawdown_pct": -99.0,
            "total_trades": 60,
        }
        ok, reason = _is_accept_candidate(metrics, target_sharpe=1.0)
        assert ok is False
        assert reason == "ruined_metrics"

    def test_policy_change_type_overrides_to_logic_on_ruined_no_trades_cycle(self):
        session = BuilderSession(
            session_id="test",
            objective="test",
            session_dir=Path("/tmp/test_policy_logic"),
        )
        it1 = BuilderIteration(
            iteration=1,
            diagnostic_category="ruined",
            diagnostic_detail={"severity": "critical"},
        )
        it2 = BuilderIteration(
            iteration=2,
            diagnostic_category="no_trades",
            diagnostic_detail={"severity": "critical"},
        )
        session.iterations = [it1, it2]

        override = _policy_change_type_override(session=session, last_iteration=it2)
        assert override == "logic"

    def test_policy_change_type_overrides_to_params_near_target(self):
        session = BuilderSession(
            session_id="test",
            objective="test",
            session_dir=Path("/tmp/test_policy_params"),
        )
        it1 = BuilderIteration(
            iteration=1,
            diagnostic_category="approaching_target",
            diagnostic_detail={"severity": "info"},
        )
        session.iterations = [it1]

        override = _policy_change_type_override(session=session, last_iteration=it1)
        assert override == "params"


class TestDeterministicFallbackCode:
    """Vérifie le fallback de code déterministe en dernier recours."""

    def test_deterministic_fallback_is_valid_python(self):
        proposal = {
            "strategy_name": "Fallback Test",
            "used_indicators": ["rsi", "bollinger", "adx"],
            "default_params": {"rsi_period": 14, "stop_atr_mult": 1.5},
        }
        code = _build_deterministic_fallback_code(proposal)
        is_valid, msg = validate_generated_code(code)
        assert is_valid, msg

    def test_deterministic_fallback_conservative_logic(self):
        proposal = {"strategy_name": "Fallback Test", "used_indicators": ["rsi"]}
        code = _build_deterministic_fallback_code(proposal)
        assert "rsi_oversold" in code
        assert "rsi_overbought" in code
        assert "bollinger" in code

    def test_deterministic_fallback_breakout_variant_for_donchian_adx(self):
        proposal = {
            "strategy_name": "Fallback Breakout",
            "used_indicators": ["donchian", "adx", "atr"],
        }
        code = _build_deterministic_fallback_code(proposal)
        assert "dc_upper" in code
        assert "dc_lower" in code
        assert "adx_threshold" in code


# ─── Tests bridge paramétrique ───────────────────────────────────────────

class TestParametricVariantNormalization:
    """Vérifie la normalisation/gating des variants paramétriques."""

    def test_normalize_variant_rewrites_crosses_and_exposes_metadata(self):
        variant = {
            "run_id": "cat_test_001",
            "variant_id": "variant_a",
            "archetype_id": "arch_a",
            "param_pack_id": "pack_a",
            "params": {"bb_period": 20},
            "proposal": {
                "entry_long_logic": "close > bollinger.upper",
                "entry_short_logic": "close < bollinger.lower",
                "exit_logic": "close crosses bollinger.middle",
            },
            "builder_text": (
                "FICHE_STRATEGIE v1\n"
                "exit:\n"
                "  - condition: close crosses bollinger.middle\n"
                "market: {symbol} {timeframe}\n"
            ),
            "fingerprint": "fp_1",
        }

        normalized = normalize_variant_for_builder(
            variant,
            symbol="BTCUSDC",
            timeframe="1h",
        )
        assert normalized is not None
        assert normalized["run_id"] == "cat_test_001"
        assert normalized["variant_id"] == "variant_a"
        assert normalized["archetype_id"] == "arch_a"
        assert normalized["param_pack_id"] == "pack_a"
        assert normalized["params"] == {"bb_period": 20}
        assert "cross_any(close, bollinger.middle)" in normalized["proposal"]["exit_logic"]
        assert "cross_any(close, bollinger.middle)" in normalized["builder_text"]
        assert "crosses" not in normalized["objective_text"].lower()
        assert "BTCUSDC" in normalized["objective_text"]
        assert "1h" in normalized["objective_text"]

    def test_normalize_variant_rejects_forbidden_tokens(self):
        variant = {
            "variant_id": "variant_bad",
            "archetype_id": "arch_bad",
            "param_pack_id": "pack_bad",
            "params": {},
            "proposal": {
                "entry_long_logic": "df['close'] > bollinger.upper",
                "entry_short_logic": "close < bollinger.lower",
                "exit_logic": "close > bollinger.middle",
            },
            "builder_text": "FICHE_STRATEGIE v1\nentry:\n  - long: df['close'] > bollinger.upper\n",
            "fingerprint": "fp_bad",
        }
        assert normalize_variant_for_builder(variant) is None


# ─── Tests chargement dynamique ──────────────────────────────────────────

class TestDynamicLoad:
    """Tests du chargement dynamique de stratégie."""

    def test_save_and_load(self, valid_strategy_code, tmp_path):
        """Vérifie que le code valide peut être chargé dynamiquement."""
        builder = StrategyBuilder.__new__(StrategyBuilder)
        builder.available_indicators = ["rsi", "atr"]

        session = BuilderSession(
            session_id="test_dynamic",
            objective="Test",
            session_dir=tmp_path / "test_dynamic",
        )
        session.session_dir.mkdir(parents=True, exist_ok=True)

        cls = builder._save_and_load(session, valid_strategy_code, 1)
        assert cls.__name__ == GENERATED_CLASS_NAME

        # Instancier et vérifier les propriétés
        instance = cls()
        assert "rsi" in instance.required_indicators
        assert "rsi_period" in instance.default_params

    def test_save_creates_versioned_copy(self, valid_strategy_code, tmp_path):
        builder = StrategyBuilder.__new__(StrategyBuilder)
        builder.available_indicators = ["rsi", "atr"]

        session = BuilderSession(
            session_id="test_versioned",
            objective="Test",
            session_dir=tmp_path / "test_versioned",
        )
        session.session_dir.mkdir(parents=True, exist_ok=True)

        builder._save_and_load(session, valid_strategy_code, 3)

        assert (session.session_dir / "strategy.py").exists()
        assert (session.session_dir / "strategy_v3.py").exists()


# ─── Tests indicateurs disponibles ───────────────────────────────────────

class TestIndicators:
    """Vérifie que le builder voit bien le registry."""

    def test_available_indicators_not_empty(self):
        from indicators.registry import list_indicators
        indicators = list_indicators()
        assert len(indicators) > 10
        assert "bollinger" in indicators
        assert "atr" in indicators
        assert "rsi" in indicators
        assert "ema" in indicators

    def test_builder_gets_indicators(self):
        # On ne peut pas instancier le builder sans LLM, mais on peut
        # vérifier la liste statiquement
        from indicators.registry import list_indicators
        assert "macd" in list_indicators()
        assert "supertrend" in list_indicators()


# ─── Tests templates ──────────────────────────────────────────────────────

class TestTemplates:
    """Vérifie que les templates Jinja2 sont accessibles et rendables."""

    def test_proposal_template_renders(self):
        from utils.template import render_prompt
        context = {
            "objective": "Trend following BTC",
            "available_indicators": ["rsi", "bollinger", "atr"],
            "iteration": 1,
            "max_iterations": 5,
        }
        result = render_prompt("strategy_builder_proposal.jinja2", context)
        assert "Trend following BTC" in result
        assert "rsi" in result
        assert "ITERATION 1" in result

    def test_code_template_renders(self):
        from utils.template import render_prompt
        context = {
            "objective": "Mean reversion ETH",
            "proposal": {
                "strategy_name": "test_strat",
                "used_indicators": ["rsi", "bollinger"],
                "entry_long_logic": "RSI < 30",
                "entry_short_logic": "RSI > 70",
                "exit_logic": "RSI crosses 50",
                "risk_management": "ATR stop",
                "default_params": {"rsi_period": 14},
            },
            "available_indicators": ["rsi", "bollinger", "atr"],
            "class_name": GENERATED_CLASS_NAME,
        }
        result = render_prompt("strategy_builder_code.jinja2", context)
        assert GENERATED_CLASS_NAME in result
        assert "Mean reversion ETH" in result
        assert "rsi, bollinger" in result


class TestMarketRecommendationDiversity:
    def test_recommend_market_context_diversity_overrides_repeated_pair(self):
        llm = _DummyLLMClient(
            '{"symbol":"0GUSDC","timeframe":"1h","confidence":0.92,"reason":"objectif explicite"}',
        )
        result = recommend_market_context(
            llm,
            objective="Breakout Donchian sur 0GUSDC 1h",
            candidate_symbols=["0GUSDC", "BTCUSDC"],
            candidate_timeframes=["1h", "15m"],
            default_symbol="BTCUSDC",
            default_timeframe="15m",
            recent_markets=[("0GUSDC", "1h"), ("BTCUSDC", "1h"), ("0GUSDC", "15m")],
        )
        assert result["symbol"] == "BTCUSDC"
        assert result["timeframe"] == "15m"
        assert str(result["source"]).endswith("diversity_override")



    def test_recommend_market_context_rotates_when_all_pairs_recent(self):
        llm = _DummyLLMClient(
            '{"symbol":"0GUSDC","timeframe":"1h","confidence":0.90,"reason":"focus"}',
        )
        result = recommend_market_context(
            llm,
            objective="Breakout 0GUSDC 1h",
            candidate_symbols=["0GUSDC", "BTCUSDC"],
            candidate_timeframes=["1h"],
            default_symbol="0GUSDC",
            default_timeframe="1h",
            recent_markets=[("0GUSDC", "1h"), ("BTCUSDC", "1h")],
        )
        assert result["symbol"] == "BTCUSDC"
        assert result["timeframe"] == "1h"
        assert str(result["source"]).endswith("diversity_override")
class TestBuilderRobustnessProfitFactor:
    def test_accept_candidate_rejects_low_profit_factor(self):
        metrics = {
            "sharpe_ratio": 1.3,
            "total_return_pct": 12.0,
            "max_drawdown_pct": -18.0,
            "total_trades": 45,
            "profit_factor": 1.01,
        }
        ok, reason = _is_accept_candidate(metrics, target_sharpe=1.0)
        assert ok is False
        assert reason == "profit_factor_too_low"

