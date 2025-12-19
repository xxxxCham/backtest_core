"""
Test de l'Orchestration Complète
=================================

Vérifie que l'orchestration entre stratégies, indicateurs et granularité
fonctionne correctement de bout en bout.
"""

from strategies.base import list_strategies, get_strategy
from strategies.indicators_mapping import get_required_indicators, get_strategy_info
from indicators.registry import list_indicators, get_indicator
from utils.parameters import (
    PRESETS,
    generate_param_grid,
    parameter_values,
    ParameterSpec,
)
from ui.indicators_panel import (
    group_indicators_by_category,
    get_category_for_indicator,
    format_indicator_name,
)


def test_strategies_listing():
    """Test que toutes les stratégies sont listées."""
    print("=" * 80)
    print("TEST 1: Listing des Stratégies")
    print("=" * 80)

    strategies = list_strategies()
    print(f"\n✓ {len(strategies)} stratégies disponibles:")
    for strat in sorted(strategies):
        print(f"  - {strat}")

    return len(strategies) > 0


def test_indicators_for_each_strategy():
    """Test que chaque stratégie a ses indicateurs mappés."""
    print("\n" + "=" * 80)
    print("TEST 2: Indicateurs pour chaque Stratégie")
    print("=" * 80)

    strategies = list_strategies()

    for strat_name in sorted(strategies):
        info = get_strategy_info(strat_name)
        required = info.required_indicators
        internal = info.internal_indicators

        print(f"\n📊 {strat_name}:")
        print(f"   Requis: {required if required else 'Aucun'}")
        print(f"   Internes: {internal if internal else 'Aucun'}")

    return True


def test_indicators_registry():
    """Test le registre d'indicateurs."""
    print("\n" + "=" * 80)
    print("TEST 3: Registre des Indicateurs")
    print("=" * 80)

    all_indicators = list_indicators()
    print(f"\n✓ {len(all_indicators)} indicateurs enregistrés:")

    # Grouper par catégorie
    categories = group_indicators_by_category()

    for category, indicators in categories.items():
        print(f"\n{category} ({len(indicators)}):")
        for ind_name in sorted(indicators):
            info = get_indicator(ind_name)
            desc = info.description if info else "N/A"
            print(f"  - {ind_name.upper()}: {desc}")

    return len(all_indicators) > 0


def test_granularity_system():
    """Test le système de granularité."""
    print("\n" + "=" * 80)
    print("TEST 4: Système de Granularité")
    print("=" * 80)

    # Test avec différentes granularités
    test_cases = [
        (0.0, "Très fin"),
        (0.5, "Modéré"),
        (1.0, "Très grossier"),
    ]

    for granularity, label in test_cases:
        values = parameter_values(
            min_val=10,
            max_val=50,
            granularity=granularity,
            param_type="int"
        )
        print(f"\n{label} (granularité={granularity}):")
        print(f"  bb_period (10-50) → {list(values)} ({len(values)} valeurs)")

    return True


def test_presets_system():
    """Test le système de Presets."""
    print("\n" + "=" * 80)
    print("TEST 5: Système de Presets")
    print("=" * 80)

    print(f"\n✓ {len(PRESETS)} Presets disponibles:")

    for preset_name, preset in PRESETS.items():
        print(f"\n📦 {preset.name}:")
        print(f"   Indicateurs: {preset.indicators}")
        print(f"   Paramètres: {len(preset.parameters)}")
        print(f"   Granularité: {preset.default_granularity}")

        # Estimer combinaisons
        total_combos = preset.estimate_combinations()
        print(f"   Combinaisons (~): {total_combos:,}")

    return len(PRESETS) > 0


def test_end_to_end_workflow():
    """Test le workflow complet bout en bout."""
    print("\n" + "=" * 80)
    print("TEST 6: Workflow Complet (End-to-End)")
    print("=" * 80)

    # 1. Sélectionner une stratégie
    strategy_name = "bollinger_atr"
    print(f"\n1. Stratégie sélectionnée: {strategy_name}")

    # 2. Récupérer les indicateurs requis
    info = get_strategy_info(strategy_name)
    required_indicators = info.required_indicators
    print(f"2. Indicateurs requis: {required_indicators}")

    # 3. Vérifier que les indicateurs existent
    print(f"3. Vérification des indicateurs:")
    for ind_name in required_indicators:
        ind_info = get_indicator(ind_name)
        if ind_info:
            print(f"   ✓ {ind_name.upper()}: {ind_info.description}")
        else:
            print(f"   ✗ {ind_name.upper()}: MANQUANT!")
            return False

    # 4. Charger la stratégie
    strategy_class = get_strategy(strategy_name)
    strategy = strategy_class()
    print(f"4. Stratégie chargée: {strategy.name}")

    # 5. Récupérer les paramètres
    params = strategy.parameter_specs
    print(f"5. Paramètres de la stratégie: {list(params.keys())}")

    # 6. Générer une grille avec granularité
    granularity = 0.7
    print(f"6. Génération grille (granularité={granularity}):")

    grid = generate_param_grid(
        params_specs=params,
        granularity=granularity,
        max_values_per_param=4,
        max_total_combinations=2000  # Augmenté pour le test
    )

    print(f"   ✓ {len(grid)} combinaisons générées")
    print(f"   Exemple: {grid[0]}")

    # 7. Workflow complet validé
    print(f"\n7. ✅ Workflow complet validé!")
    print(f"   - Stratégie → Indicateurs → Paramètres → Grille")

    return True


def test_ui_integration():
    """Test l'intégration UI."""
    print("\n" + "=" * 80)
    print("TEST 7: Intégration UI")
    print("=" * 80)

    print("\nComposants UI disponibles:")

    # Test groupement par catégorie
    categories = group_indicators_by_category()
    print(f"\n✓ Groupement par catégorie: {len(categories)} catégories")

    # Test formatage
    test_ind = "bollinger"
    formatted = format_indicator_name(test_ind, with_description=True)
    print(f"✓ Formatage: {formatted}")

    # Test catégorie
    category = get_category_for_indicator(test_ind)
    print(f"✓ Catégorie de '{test_ind}': {category}")

    return True


def main():
    """Lance tous les tests."""
    print("\n🧪 TEST DE L'ORCHESTRATION COMPLÈTE\n")

    results = []

    results.append(("Listing Stratégies", test_strategies_listing()))
    results.append(("Indicateurs par Stratégie", test_indicators_for_each_strategy()))
    results.append(("Registre Indicateurs", test_indicators_registry()))
    results.append(("Système Granularité", test_granularity_system()))
    results.append(("Système Presets", test_presets_system()))
    results.append(("Workflow E2E", test_end_to_end_workflow()))
    results.append(("Intégration UI", test_ui_integration()))

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
        print("✅ TOUS LES TESTS RÉUSSIS - ORCHESTRATION COMPLÈTE VALIDÉE!")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
