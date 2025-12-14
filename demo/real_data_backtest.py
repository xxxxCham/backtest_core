"""
Backtest sur données réelles Parquet
====================================

Script qui charge vos données Parquet existantes et exécute un backtest.

Usage:
    python demo/real_data_backtest.py
    python demo/real_data_backtest.py --symbol BTCUSDC --timeframe 15m
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import pandas as pd

from backtest.engine import BacktestEngine

# Chemin vers vos données Parquet
DATA_DIR = Path(r"D:\ThreadX_big\data\crypto\processed\parquet")


def list_available_data():
    """Liste les symboles et timeframes disponibles."""
    if not DATA_DIR.exists():
        print(f"❌ Répertoire non trouvé: {DATA_DIR}")
        return {}

    available = {}
    for f in DATA_DIR.glob("*.parquet"):
        # Format: BTCUSDC_15m.parquet
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            symbol, tf = parts
            if symbol not in available:
                available[symbol] = []
            available[symbol].append(tf)

    return available


def load_parquet_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Charge un fichier Parquet de données OHLCV."""
    filepath = DATA_DIR / f"{symbol}_{timeframe}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

    print(f"📂 Chargement: {filepath}")
    df = pd.read_parquet(filepath)

    # Afficher les colonnes disponibles
    print(f"   Colonnes: {list(df.columns)}")
    print(f"   Lignes: {len(df):,}")

    # S'assurer que l'index est datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        # Chercher une colonne datetime
        for col in ["timestamp", "datetime", "date", "time", "open_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break

    # Normaliser les noms de colonnes
    col_mapping = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    }
    df = df.rename(columns=col_mapping)

    # Vérifier les colonnes requises
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    print(f"   Période: {df.index.min()} → {df.index.max()}")

    return df


def run_backtest(symbol: str, timeframe: str, strategy: str = "bollinger_atr"):
    """Exécute un backtest sur les données réelles."""
    print("=" * 60)
    print(f"  BACKTEST: {symbol} {timeframe}")
    print("=" * 60)

    # 1. Charger les données
    df = load_parquet_data(symbol, timeframe)

    # 2. Créer le moteur
    engine = BacktestEngine(initial_capital=10000)

    # 3. Exécuter le backtest
    print(f"\n🎯 Stratégie: {strategy}")

    result = engine.run(
        df=df,
        strategy=strategy,
        params={
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "k_sl": 1.5,
            "leverage": 3
        },
        symbol=symbol,
        timeframe=timeframe
    )

    # 4. Afficher les résultats
    print("\n" + "=" * 60)
    print("  RÉSULTATS")
    print("=" * 60)

    metrics = result.metrics
    print(f"""
📊 Métriques de Performance:
   ─────────────────────────
   Trades totaux:     {metrics.get('total_trades', 0):>8}
   Win Rate:          {metrics.get('win_rate', 0):>8.1f}%

   P&L Total:         ${metrics.get('total_pnl', 0):>10,.2f}
   Rendement:         {metrics.get('total_return_pct', 0):>8.2f}%

   Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>8.2f}
   Sortino Ratio:     {metrics.get('sortino_ratio', 0):>8.2f}
   Max Drawdown:      {metrics.get('max_drawdown', 0):>8.1f}%

   Profit Factor:     {metrics.get('profit_factor', 0):>8.2f}
   Gain Moyen:        ${metrics.get('avg_win', 0):>10,.2f}
   Perte Moyenne:     ${metrics.get('avg_loss', 0):>10,.2f}
""")

    # 5. Premiers et derniers trades
    if not result.trades.empty:
        print("\n📋 Derniers trades:")
        print(result.trades[["entry_ts", "exit_ts", "side", "pnl", "exit_reason"]].tail(5).to_string())

    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest sur données Parquet réelles")
    parser.add_argument("--symbol", "-s", default="BTCUSDC", help="Symbole (ex: BTCUSDC)")
    parser.add_argument("--timeframe", "-t", default="15m", help="Timeframe (ex: 15m, 1h)")
    parser.add_argument("--strategy", default="bollinger_atr", help="Stratégie (bollinger_atr, ema_cross)")
    parser.add_argument("--list", "-l", action="store_true", help="Lister les données disponibles")

    args = parser.parse_args()

    if args.list:
        print("📁 Données disponibles:")
        available = list_available_data()
        for symbol, tfs in sorted(available.items())[:20]:
            print(f"   {symbol}: {', '.join(sorted(tfs))}")
        print(f"\n   ... et {len(available)} symboles au total")
        return

    try:
        run_backtest(args.symbol, args.timeframe, args.strategy)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nUtilisez --list pour voir les données disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
