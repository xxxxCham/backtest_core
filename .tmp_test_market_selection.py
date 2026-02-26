"""
Script de test pour vérifier le système de sélection intelligente de tokens/TF.

Usage:
    python .tmp_test_market_selection.py
"""

from config.market_selection import (
    rank_tokens_for_strategy,
    get_token_profile,
    get_strategy_requirements,
)

def test_token_ranking():
    """Test du tri des tokens selon différents types de stratégies."""

    # Tokens de test
    test_tokens = [
        "BTCUSDC", "ETHUSDC", "SOLUSDC", "AVAXUSDC",
        "XRPUSDC", "LTCUSDC", "APTUSDC", "TRXUSDC"
    ]

    print("=" * 80)
    print("TEST DE RANKING DES TOKENS PAR TYPE DE STRATÉGIE")
    print("=" * 80)

    strategy_types = ["scalping", "breakout", "momentum", "trend", "mean_reversion"]

    for strategy_type in strategy_types:
        print(f"\n📊 Type de stratégie: {strategy_type.upper()}")
        print("-" * 80)

        # Récupérer les exigences
        reqs = get_strategy_requirements(strategy_type)
        print(f"   Volatilité préférée: {', '.join(reqs.get('volatility_preferred', []))}")
        print(f"   Liquidité minimum: {reqs.get('liquidity_min', 'medium')}")
        print(f"   Timeframes recommandés: {', '.join(reqs.get('timeframes', ['1h']))}")

        # Trier les tokens
        ranked = rank_tokens_for_strategy(test_tokens, strategy_type)

        print(f"\n   Tokens triés par pertinence:")
        for i, token in enumerate(ranked[:5], 1):
            profile = get_token_profile(token)
            print(f"      {i}. {token:12} (vol={profile['volatility']:6}, liq={profile['liquidity']:6})")

        print()


def test_strategy_detection():
    """Test de la détection de type de stratégie dans les objectifs."""

    print("\n" + "=" * 80)
    print("TEST DE DÉTECTION DE TYPE DE STRATÉGIE")
    print("=" * 80)

    test_objectives = [
        ("Scalping rapide sur BTC 5m avec RSI", "scalping"),
        ("Breakout Donchian sur ETH 15m", "breakout"),
        ("Momentum EMA sur SOL 1h", "momentum"),
        ("Trend following sur AVAX 4h avec MACD", "trend"),
        ("Mean reversion Bollinger sur XRP 30m", "mean_reversion"),
        ("Stratégie quelconque", None),  # Pas de détection
    ]

    for objective, expected_type in test_objectives:
        print(f"\n📝 Objectif: {objective}")
        print(f"   Type attendu: {expected_type or 'UNKNOWN'}")

        # Simuler la détection (même logique que dans recommend_market_context)
        objective_lower = objective.lower()
        detected = None

        if any(kw in objective_lower for kw in ["scalp", "court terme", "rapide"]):
            detected = "scalping"
        elif any(kw in objective_lower for kw in ["breakout", "cassure", "sortie.*range", "donchian"]):
            detected = "breakout"
        elif any(kw in objective_lower for kw in ["momentum", "directionnel"]):
            detected = "momentum"
        elif any(kw in objective_lower for kw in ["tendance", "trend", "suivre"]):
            detected = "trend"
        elif any(kw in objective_lower for kw in ["mean", "reversion", "retour", "moyenne", "bollinger", "survente"]):
            detected = "mean_reversion"

        status = "✅" if detected == expected_type else "❌"
        print(f"   {status} Type détecté: {detected or 'UNKNOWN'}")


if __name__ == "__main__":
    test_token_ranking()
    test_strategy_detection()

    print("\n" + "=" * 80)
    print("✅ Tests terminés")
    print("=" * 80)
