"""
Tests pour le Strategy Builder (agents/strategy_builder.py).

Couvre :
- Validation du code généré (syntaxe, sécurité, structure)
- Création de session (ID, dossier)
- Extraction JSON/Python depuis réponses LLM
- Chargement dynamique de stratégie
"""

import ast
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
    _extract_json_from_response,
    _extract_python_from_response,
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
