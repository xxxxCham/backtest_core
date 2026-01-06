"""
Tests unitaires pour les indicateurs FairValOseille.

Teste swing_high, swing_low, FVG bullish/bearish, FVA.
"""

import pandas as pd
import pytest

from indicators.fva import calculate_fva
from indicators.fvg import calculate_fvg_bearish, calculate_fvg_bullish
from indicators.swing import calculate_swing_high, calculate_swing_low


class TestSwingDetection:
    """Tests pour la detection des swing points."""

    def test_swing_high_basic(self):
        """Test detection swing high basique."""
        # Pattern: prix monte puis descend
        highs = [100, 105, 110, 108, 105]
        df = pd.DataFrame({'high': highs})

        result = calculate_swing_high(df)

        # Swing high detecte a l'index 2 (110 > 105 et 110 > 108)
        assert result[2]
        assert not result[0]  # Bord
        assert not result[4]  # Bord

    def test_swing_low_basic(self):
        """Test detection swing low basique."""
        # Pattern: prix baisse puis remonte
        lows = [100, 95, 90, 93, 96]
        df = pd.DataFrame({'low': lows})

        result = calculate_swing_low(df)

        # Swing low detecte a l'index 2 (90 < 95 et 90 < 93)
        assert result[2]
        assert not result[0]  # Bord
        assert not result[4]  # Bord

    def test_swing_no_detection(self):
        """Test cas sans swing (tendance monotone)."""
        # Tendance haussiere monotone
        highs = [100, 105, 110, 115, 120]
        df = pd.DataFrame({'high': highs})

        result = calculate_swing_high(df)

        # Aucun swing detecte
        assert not result.any()

    def test_swing_multiple(self):
        """Test detection multiples swings."""
        highs = [100, 105, 103, 108, 106, 110, 108]
        df = pd.DataFrame({'high': highs})

        result = calculate_swing_high(df)

        # Swings detectes aux indices 1, 3, 5
        assert result[1]  # 105 > 100 et 105 > 103
        assert result[3]  # 108 > 103 et 108 > 106
        assert result[5]  # 110 > 106 et 110 > 108


class TestFVGDetection:
    """Tests pour la detection des Fair Value Gaps."""

    def test_fvg_bullish_basic(self):
        """Test detection FVG bullish basique."""
        # FVG bullish: low[2] > high[0]
        highs = [100, 105, 112, 115, 118]
        lows = [98, 103, 110, 113, 116]
        df = pd.DataFrame({'high': highs, 'low': lows})

        result = calculate_fvg_bullish(df)

        # FVG bullish a l'index 2: low[2]=110 > high[0]=100
        assert result[2]
        assert not result[0]  # Pas de i-2
        assert not result[1]  # Pas de i-2

    def test_fvg_bearish_basic(self):
        """Test detection FVG bearish basique."""
        # FVG bearish: high[2] < low[0]
        highs = [100, 95, 88, 85, 82]
        lows = [98, 93, 86, 83, 80]
        df = pd.DataFrame({'high': highs, 'low': lows})

        result = calculate_fvg_bearish(df)

        # FVG bearish a l'index 2: high[2]=88 < low[0]=98
        assert result[2]

    def test_fvg_no_gap(self):
        """Test cas sans gap (prix chevauche)."""
        # Prix qui se chevauchent (low[i] <= high[i-2])
        highs = [100, 101, 102, 103, 104]
        lows = [95, 96, 97, 98, 99]  # lows[2]=97 < highs[0]=100, pas de gap
        df = pd.DataFrame({'high': highs, 'low': lows})

        result_bull = calculate_fvg_bullish(df)
        result_bear = calculate_fvg_bearish(df)

        # Aucun gap detecte (prix se chevauchent)
        assert not result_bull.any(), f"FVG bullish inattendu: {result_bull}"
        assert not result_bear.any(), f"FVG bearish inattendu: {result_bear}"


class TestFVADetection:
    """Tests pour la detection des Fair Value Areas."""

    def test_fva_basic(self):
        """Test detection FVA basique (inside bar)."""
        highs = [100, 110, 108, 115, 120]
        lows = [90, 95, 97, 100, 105]
        df = pd.DataFrame({'high': highs, 'low': lows})

        result = calculate_fva(df)

        # FVA a l'index 2: high[2]=108 < high[1]=110 ET low[2]=97 > low[1]=95
        assert result[2]

    def test_fva_no_consolidation(self):
        """Test cas sans consolidation (expansion)."""
        highs = [100, 110, 120, 130, 140]
        lows = [90, 95, 100, 105, 110]
        df = pd.DataFrame({'high': highs, 'low': lows})

        result = calculate_fva(df)

        # Aucune FVA (toujours expansion)
        assert not result.any()

    def test_fva_edge_case(self):
        """Test cas limite (egalite)."""
        highs = [100, 110, 110, 115, 120]
        lows = [90, 95, 95, 100, 105]
        df = pd.DataFrame({'high': highs, 'low': lows})

        result = calculate_fva(df)

        # Pas de FVA si egalite (besoin strict </>)
        assert not result[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
