"""
Test Complet : Presets + Orchestration LLM
===========================================

Valide le système complet de Presets et de logging d'orchestration.
"""

from utils.parameters import PRESETS
from utils.preset_validation import (
    validate_all_presets,
    format_validation_report,
    create_preset_from_strategy,
)
from agents.orchestration_logger import (
    OrchestrationLogger,
    OrchestrationActionType,
    OrchestrationStatus,
)


def test_presets_validation():
    """Test la validation de tous les Presets."""
    print("=" * 80)
    print("TEST 1: Validation des Presets")
    print("=" * 80)

    print(f"\n✓ {len(PRESETS)} Presets définis")

    # Valider tous les Presets
    results = validate_all_presets()

    # Afficher le rapport
    report = format_validation_report(results)
    print(f"\n{report}")

    # Compter les valides
    valid_count = sum(1 for r in results.values() if r.is_valid)
    total_count = len(results)

    return valid_count == total_count


def test_preset_auto_creation():
    """Test la création automatique de Presets."""
    print("\n" + "=" * 80)
    print("TEST 2: Création Automatique de Presets")
    print("=" * 80)

    # Tester avec une stratégie
    strategy_name = "bollinger_atr"
    print(f"\nCréation auto pour: {strategy_name}")

    preset = create_preset_from_strategy(
        strategy_name=strategy_name,
        preset_name="test_auto",
        description="Test de création automatique"
    )

    print(f"✓ Preset créé: {preset.name}")
    print(f"   Indicateurs: {preset.indicators}")
    print(f"   Paramètres: {len(preset.parameters)}")
    print(f"   Granularité: {preset.default_granularity}")

    # Valider que les indicateurs sont corrects
    from strategies.indicators_mapping import get_required_indicators
    expected = set(get_required_indicators(strategy_name))
    actual = set(preset.indicators)

    if expected == actual:
        print(f"✓ Indicateurs corrects: {preset.indicators}")
        return True
    else:
        print(f"✗ Incohérence indicateurs")
        print(f"   Attendu: {expected}")
        print(f"   Obtenu: {actual}")
        return False


def test_orchestration_logging():
    """Test le système de logging d'orchestration."""
    print("\n" + "=" * 80)
    print("TEST 3: Logging d'Orchestration")
    print("=" * 80)

    # Créer un logger
    logger = OrchestrationLogger(session_id="test_session")

    print(f"\n✓ Logger créé: session_id={logger.session_id}")

    # Simuler un workflow
    logger.log_analysis_start("Analyst", {"data": "market_data"})
    logger.log_analysis_complete("Analyst", {"trend": "bullish"})

    logger.log_strategy_selection("Strategist", "bollinger_atr", "Marché range")

    logger.log_indicator_values_change(
        "Strategist",
        "bollinger",
        old_values={"period": 20, "std": 2.0},
        new_values={"period": 25, "std": 2.5},
        reason="Augmenter sensibilité"
    )

    logger.log_backtest_launch(
        "Executor",
        params={"bb_period": 25, "atr_period": 14},
        combination_id=1,
        total_combinations=100
    )

    logger.log_backtest_complete(
        "Executor",
        params={"bb_period": 25, "atr_period": 14},
        results={"pnl": 150.5, "sharpe": 1.8},
        combination_id=1
    )

    logger.log_decision("Validator", "continue", "Résultats prometteurs")

    logger.next_iteration()

    # Vérifier les logs
    print(f"✓ {len(logger.logs)} logs enregistrés")

    # Afficher le résumé
    summary = logger.generate_summary()
    print(f"\n{summary}")

    # Tester les filtres
    analyst_logs = logger.get_logs_by_agent("Analyst")
    print(f"\n✓ Logs de l'Analyst: {len(analyst_logs)}")

    backtest_logs = logger.get_logs_by_type(OrchestrationActionType.BACKTEST_COMPLETE)
    print(f"✓ Backtests complétés: {len(backtest_logs)}")

    # Sauvegarder
    logger.save_to_file()
    print(f"\n✓ Logs sauvegardés dans orchestration_logs_{logger.session_id}.json")

    return len(logger.logs) > 0


def test_full_workflow():
    """Test un workflow complet de bout en bout."""
    print("\n" + "=" * 80)
    print("TEST 4: Workflow Complet End-to-End")
    print("=" * 80)

    # 1. Valider un Preset
    print("\n1. Validation Preset 'bollinger_atr'")
    from utils.preset_validation import validate_preset_against_strategy
    from utils.parameters import get_preset

    preset = get_preset("bollinger_dual")
    result = validate_preset_against_strategy(preset, "bollinger_dual")

    print(f"   {result.summary()}")

    # 2. Créer un logger
    print("\n2. Initialisation du logger d'orchestration")
    logger = OrchestrationLogger(session_id="workflow_test")

    # 3. Simuler une séquence d'optimisation
    print("\n3. Simulation d'une séquence d'optimisation")

    logger.log_strategy_selection("Analyst", "bollinger_dual", "Double condition efficace")

    logger.log_backtest_launch(
        "Executor",
        params={"bb_window": 20, "bb_std": 2.0},
        combination_id=1,
        total_combinations=10
    )

    logger.log_backtest_complete(
        "Executor",
        params={"bb_window": 20, "bb_std": 2.0},
        results={"pnl": 200.0, "sharpe": 2.1, "max_dd": -50.0},
        combination_id=1
    )

    logger.log_decision("Critic", "continue", "Sharpe > 2.0, continuer")

    logger.next_iteration()

    logger.log_indicator_add(
        "Strategist",
        "atr",
        params={"period": 14},
        reason="Ajouter filtre volatilité",
        status=OrchestrationStatus.PENDING
    )

    logger.log_indicator_validation(
        "Validator",
        "atr",
        is_valid=True,
        message="ATR compatible avec Bollinger"
    )

    logger.log_backtest_launch(
        "Executor",
        params={"bb_window": 20, "bb_std": 2.0, "atr_period": 14},
        combination_id=2,
        total_combinations=10
    )

    logger.log_backtest_complete(
        "Executor",
        params={"bb_window": 20, "bb_std": 2.0, "atr_period": 14},
        results={"pnl": 250.0, "sharpe": 2.5, "max_dd": -40.0},
        combination_id=2
    )

    logger.log_decision("Validator", "stop", "Sharpe optimal atteint")

    print(f"   ✓ Workflow simulé: {logger.current_iteration} itérations")
    print(f"   ✓ {len(logger.logs)} actions enregistrées")

    # 4. Afficher les logs de chaque agent
    print("\n4. Actions par agent:")
    for agent in ["Analyst", "Strategist", "Critic", "Validator", "Executor"]:
        agent_logs = logger.get_logs_by_agent(agent)
        if agent_logs:
            print(f"   - {agent}: {len(agent_logs)} actions")

    # 5. Générer le résumé
    summary = logger.generate_summary()
    print(f"\n{summary}")

    return True


def main():
    """Lance tous les tests."""
    print("\n🧪 TEST COMPLET: PRESETS + ORCHESTRATION LLM\n")

    results = []

    results.append(("Validation Presets", test_presets_validation()))
    results.append(("Création Auto Preset", test_preset_auto_creation()))
    results.append(("Logging Orchestration", test_orchestration_logging()))
    results.append(("Workflow E2E", test_full_workflow()))

    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TOUS LES TESTS RÉUSSIS!")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
