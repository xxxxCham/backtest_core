#!/usr/bin/env python
"""
Script pour charger et utiliser les configurations rentables validées.

Usage:
    python use_profitable_configs.py --preset ema_cross_champion
    python use_profitable_configs.py --list
    python use_profitable_configs.py --backtest ema_cross_champion
"""

from pathlib import Path

import pandas as pd
import toml

from backtest.engine import BacktestEngine


def load_profitable_presets():
    """Charge les presets rentables depuis le fichier TOML."""
    config_file = Path(__file__).parent / "config" / "profitable_presets.toml"
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_file}")

    return toml.load(config_file)


def list_presets():
    """Liste tous les presets disponibles."""
    presets = load_profitable_presets()

    print("\n" + "="*80)
    print("🏆 CONFIGURATIONS RENTABLES VALIDÉES")
    print("="*80)

    meta = presets.get("meta", {})
    print("\n📊 Données de test:")
    print(f"   Token: {meta.get('token', 'N/A')}")
    print(f"   Timeframe: {meta.get('timeframe', 'N/A')}")
    print(f"   Période: {meta.get('period_start', 'N/A')} → {meta.get('period_end', 'N/A')}")
    print(f"   Barres: {meta.get('bars_count', 'N/A')}")
    print(f"   Capital initial: ${meta.get('initial_capital', 0):,.0f}")

    print("\n🎯 Presets disponibles:\n")

    # Champions
    champions = [k for k in presets.keys() if k.endswith("_champion") or k.endswith("_bronze")]
    for preset_name in sorted(champions):
        preset = presets[preset_name]
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(preset.get("rank", 0), "")

        print(f"   {rank_emoji} {preset_name}")
        print(f"      Stratégie: {preset.get('strategy', 'N/A')}")
        print(f"      Description: {preset.get('description', 'N/A')}")

        perf = preset.get("performance", {})
        print(f"      Performance: ${perf.get('pnl', 0):,.2f} ({perf.get('total_return_pct', 0):+.2f}%)")
        print(f"      Trades: {perf.get('total_trades', 0)}, Win Rate: {perf.get('win_rate_pct', 0):.1f}%")
        print()

    # Échecs (pour référence)
    failed = [k for k in presets.keys() if k.endswith("_failed")]
    if failed:
        print("   ⚠️  Configurations NON rentables (à éviter):\n")
        for preset_name in sorted(failed):
            preset = presets[preset_name]
            print(f"   ❌ {preset_name}")
            print(f"      Stratégie: {preset.get('strategy', 'N/A')}")
            perf = preset.get("performance", {})
            print(f"      Performance: ${perf.get('pnl', 0):,.2f} ({perf.get('total_return_pct', 0):+.2f}%)")
            print(f"      Raison: {perf.get('reason', 'N/A')}")
            print()

    print("="*80)


def get_preset_params(preset_name: str):
    """Récupère les paramètres d'un preset."""
    presets = load_profitable_presets()

    if preset_name not in presets:
        available = [k for k in presets.keys() if not k.startswith("_") and k != "meta"]
        raise ValueError(f"Preset '{preset_name}' non trouvé. Disponibles: {available}")

    preset = presets[preset_name]
    return {
        "strategy": preset.get("strategy"),
        "params": preset.get("params", {}),
        "description": preset.get("description", ""),
        "performance": preset.get("performance", {}),
    }


def run_backtest_with_preset(preset_name: str, data_file: str = None):
    """Lance un backtest avec un preset."""
    print(f"\n{'='*80}")
    print(f"🚀 BACKTEST AVEC PRESET: {preset_name}")
    print(f"{'='*80}\n")

    # Charger le preset
    preset_data = get_preset_params(preset_name)
    strategy_name = preset_data["strategy"]
    params = preset_data["params"]

    print(f"📋 Stratégie: {strategy_name}")
    print(f"📝 Description: {preset_data['description']}")
    print(f"⚙️  Paramètres: {params}\n")

    # Charger les données
    if data_file is None:
        sample_file = Path(__file__).parent / "data" / "sample_data" / "BTCUSDT_1h_6months.csv"
        data_file = str(sample_file)

    print(f"📥 Chargement des données: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    print(f"✅ {len(df)} barres chargées ({df.index[0]} → {df.index[-1]})\n")

    # Lancer le backtest
    engine = BacktestEngine(initial_capital=10000.0)
    result = engine.run(df=df, strategy=strategy_name, params=params)

    # Afficher les résultats
    metrics = result.metrics

    # Extraire le PnL correctement
    pnl = metrics.get('total_pnl', metrics.get('pnl', 0))
    if pnl == 0:
        return_pct = metrics.get('total_return_pct', 0)
        pnl = (return_pct / 100.0) * 10000.0

    print(f"\n{'='*80}")
    print("📊 RÉSULTATS DU BACKTEST")
    print(f"{'='*80}\n")

    print(f"💰 PnL: ${pnl:,.2f}")
    print(f"📈 Return: {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"📉 Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"⚡ Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"🎯 Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"🔄 Trades: {metrics.get('total_trades', 0)}")
    print(f"💎 Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    # Comparaison avec performance attendue
    expected_perf = preset_data["performance"]
    print(f"\n{'='*80}")
    print("🔍 COMPARAISON AVEC PERFORMANCE ATTENDUE")
    print(f"{'='*80}\n")

    print(f"PnL attendu: ${expected_perf.get('pnl', 0):,.2f} | Obtenu: ${pnl:,.2f}")
    print(f"Return attendu: {expected_perf.get('total_return_pct', 0):+.2f}% | Obtenu: {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"Trades attendus: {expected_perf.get('total_trades', 0)} | Obtenu: {metrics.get('total_trades', 0)}")

    if abs(pnl - expected_perf.get('pnl', 0)) < 10:
        print("\n✅ Résultats cohérents avec la validation !")
    else:
        print("\n⚠️  Différence significative détectée - Vérifier les données")

    print(f"\n{'='*80}\n")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Utiliser les configurations rentables validées")
    parser.add_argument("--list", action="store_true", help="Lister tous les presets disponibles")
    parser.add_argument("--preset", type=str, help="Nom du preset à afficher")
    parser.add_argument("--backtest", type=str, help="Nom du preset à backtester")
    parser.add_argument("--data", type=str, help="Fichier de données personnalisé")

    args = parser.parse_args()

    if args.list:
        list_presets()
    elif args.preset:
        preset_data = get_preset_params(args.preset)
        print(f"\n🎯 Preset: {args.preset}")
        print(f"Stratégie: {preset_data['strategy']}")
        print(f"Description: {preset_data['description']}")
        print("\nParamètres:")
        for k, v in preset_data['params'].items():
            print(f"  {k}: {v}")
        print("\nPerformance attendue:")
        for k, v in preset_data['performance'].items():
            print(f"  {k}: {v}")
    elif args.backtest:
        run_backtest_with_preset(args.backtest, args.data)
    else:
        parser.print_help()
        print("\n💡 Exemples:")
        print("  python use_profitable_configs.py --list")
        print("  python use_profitable_configs.py --backtest ema_cross_champion")
        print("  python use_profitable_configs.py --preset rsi_reversal_champion")


if __name__ == "__main__":
    main()
