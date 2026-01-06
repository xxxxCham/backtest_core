#!/usr/bin/env python
"""
Script CLI pour lancer une optimisation LLM directement depuis le terminal.

Usage:
    python run_llm_optimization.py --strategy bollinger_atr --symbol BTCUSDC --timeframe 30m --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from agents.integration import create_orchestrator_with_backtest
from agents.llm_client import LLMConfig
from data.loader import load_ohlcv
from strategies.base import get_strategy


def main():
    parser = argparse.ArgumentParser(description="Lancer une optimisation LLM")
    parser.add_argument("--strategy", default="bollinger_atr", help="Nom de la stratégie")
    parser.add_argument("--symbol", default="BTCUSDC", help="Symbole (ex: BTCUSDC)")
    parser.add_argument("--timeframe", default="30m", help="Timeframe (ex: 1h, 30m, 1d)")
    parser.add_argument("--start-date", default="2024-01-01", help="Date de début")
    parser.add_argument("--end-date", default="2024-12-31", help="Date de fin")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--max-iterations", type=int, default=10, help="Nombre max d'itérations LLM")
    parser.add_argument("--model", default="deepseek-r1-distill:14b", help="Modèle LLM à utiliser")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("🤖 OPTIMISATION LLM")
    print(f"{'='*80}")
    print(f"Stratégie: {args.strategy}")
    print(f"Symbole: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Période: {args.start_date} → {args.end_date}")
    print(f"Capital initial: ${args.initial_capital:,.2f}")
    print(f"Modèle LLM: {args.model}")
    print(f"Max itérations: {args.max_iterations}")
    print(f"{'='*80}\n")

    # Charger les données
    print("📥 Chargement des données...")
    try:
        df = load_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start_date,
            end=args.end_date
        )
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1

    print(f"✅ Données chargées: {len(df)} barres")
    print(f"   Période: {df.index[0]} → {df.index[-1]}\n")

    # Configuration LLM
    from agents.llm_client import LLMProvider

    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=args.model,
        temperature=0.7,
        max_tokens=4096,
        timeout_seconds=900,  # 15 minutes pour les modèles de raisonnement
    )

    # Récupérer les paramètres par défaut de la stratégie
    print("🔧 Récupération des paramètres par défaut...")
    strategy_class = get_strategy(args.strategy)
    strategy_instance = strategy_class()
    initial_params = strategy_instance.default_params

    print(f"   Paramètres initiaux: {list(initial_params.keys())}\n")

    # Créer l'orchestrateur
    print("🔧 Création de l'orchestrateur LLM...")
    try:
        orchestrator = create_orchestrator_with_backtest(
            strategy_name=args.strategy,
            data=df,
            initial_params=initial_params,
            data_symbol=args.symbol,
            data_timeframe=args.timeframe,
            llm_config=llm_config,
            use_walk_forward=True,
            max_iterations=args.max_iterations,
            initial_capital=args.initial_capital,
        )
    except Exception as e:
        print(f"❌ Erreur création orchestrateur: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("✅ Orchestrateur créé\n")

    # Lancer l'optimisation
    print(f"{'='*80}")
    print("🚀 DÉMARRAGE OPTIMISATION")
    print(f"{'='*80}\n")

    try:
        result = orchestrator.run()

        print(f"\n{'='*80}")
        print("📊 RÉSULTATS FINAUX")
        print(f"{'='*80}\n")

        if result and result.success:
            print("🏆 MEILLEURS PARAMÈTRES TROUVÉS:")
            for key, value in result.final_params.items():
                print(f"   {key}: {value}")

            if hasattr(result, 'final_metrics') and result.final_metrics:
                print("\n📈 MÉTRIQUES:")
                print(f"   PnL: ${result.final_metrics.get('total_pnl', 0):,.2f}")
                print(f"   Sharpe: {result.final_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"   Max DD: {result.final_metrics.get('max_drawdown_pct', 0):.2f}%")
                print(f"   Win Rate: {result.final_metrics.get('win_rate_pct', 0):.1f}%")
                print(f"   Trades: {result.final_metrics.get('total_trades', 0)}")

            print("\n📊 STATISTIQUES OPTIMISATION:")
            print(f"   Itérations: {getattr(result, 'iteration_count', 'N/A')}")
            print(f"   Succès: {result.success}")
        else:
            print("⚠️ Optimisation terminée sans succès")
            if result:
                print(f"   Message: {getattr(result, 'message', 'Aucun message')}")

        print(f"\n{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n❌ Erreur lors de l'optimisation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
