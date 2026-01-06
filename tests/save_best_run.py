"""
Relancer et sauvegarder le meilleur run identifié dans les logs.

Best run from logs:
- PNL: 15757.85
- Strategy: rsi_reversal
- Symbol: BTCUSDC 1h
- Params: rsi_period=30, oversold_level=39, overbought_level=77, leverage=1
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine
from backtest.storage import ResultStorage
from strategies import get_strategy
from data.loader import load_ohlcv
from utils.log import get_logger

logger = get_logger(__name__)


def main():
    # Configuration du meilleur run
    strategy_key = "rsi_reversal"
    symbol = "BTCUSDC"
    timeframe = "1h"

    best_params = {
        "rsi_period": 30,
        "oversold_level": 39,
        "overbought_level": 77,
        "leverage": 1,
        "fees_bps": 10,
        "slippage_bps": 5,
    }

    initial_capital = 10000

    logger.info("=" * 80)
    logger.info("🏆 SAUVEGARDE DU MEILLEUR RUN")
    logger.info(f"Strategy: {strategy_key}")
    logger.info(f"Symbol: {symbol} {timeframe}")
    logger.info(f"Params: {best_params}")
    logger.info("=" * 80)

    # Charger les données
    data_file = f"data/{symbol}_{timeframe}.parquet"
    logger.info(f"📂 Chargement des données: {data_file}")

    try:
        df = load_ohlcv(symbol, timeframe)
        logger.info(f"✅ Données chargées: {len(df)} bars")
    except FileNotFoundError:
        logger.error(f"❌ Fichier non trouvé: {data_file}")
        logger.info("📋 Fichiers disponibles:")
        data_path = Path("data")
        if data_path.exists():
            for f in sorted(data_path.glob("*.parquet"))[:10]:
                logger.info(f"   - {f.name}")
        return

    # Créer la stratégie
    strategy_cls = get_strategy(strategy_key)
    if strategy_cls is None:
        logger.error(f"❌ Stratégie non trouvée: {strategy_key}")
        return

    strategy = strategy_cls()
    logger.info(f"✅ Stratégie chargée: {strategy.name}")

    # Exécuter le backtest
    logger.info("🚀 Lancement du backtest...")
    engine = BacktestEngine(initial_capital=initial_capital)

    try:
        result = engine.run(
            df=df,
            strategy=strategy,
            params=best_params,
            symbol=symbol,
            timeframe=timeframe,
        )

        logger.info("=" * 80)
        logger.info("📊 RÉSULTATS DU BACKTEST")
        logger.info(f"📋 Clés disponibles dans metrics: {list(result.metrics.keys())[:20]}")
        logger.info(f"PNL: {result.metrics.get('pnl', 0):.2f}")
        logger.info(f"Total PNL: {result.metrics.get('total_pnl', 0):.2f}")
        logger.info(f"Total Return: {result.metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Max Drawdown: {result.metrics.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"Win Rate: {result.metrics.get('win_rate', 0):.2f}%")
        logger.info(f"Win Rate Pct: {result.metrics.get('win_rate_pct', 0):.2f}%")
        logger.info(f"Nombre de trades: {result.metrics.get('total_trades', 0)}")
        logger.info("=" * 80)

        # Vérifier si le PNL correspond
        expected_pnl = 15757.85
        actual_pnl = result.metrics.get('pnl', 0)

        if abs(actual_pnl - expected_pnl) < 1.0:
            logger.info(f"✅ PNL confirmé: {actual_pnl:.2f} ≈ {expected_pnl:.2f}")
        else:
            logger.warning(f"⚠️ PNL différent: {actual_pnl:.2f} vs attendu {expected_pnl:.2f}")
            logger.warning("   Cela peut être dû à des données différentes ou des paramètres manquants")

        # Sauvegarder
        logger.info("\n💾 Sauvegarde du résultat...")
        storage = ResultStorage()

        run_id = storage.save_result(result)
        logger.info(f"✅ Run sauvegardé: {run_id}")

        # Vérifier
        logger.info("\n🔍 Vérification...")
        saved_meta = None
        for meta in storage.list_results():
            if meta.run_id == run_id:
                saved_meta = meta
                break

        if saved_meta:
            logger.info("✅ Métadonnées sauvegardées:")
            logger.info(f"   - PNL: {saved_meta.metrics.get('pnl', 0):.2f}")
            logger.info(f"   - Sharpe: {saved_meta.metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"   - Trades: {saved_meta.n_trades}")
        else:
            logger.error("❌ Run non trouvé dans l'index après sauvegarde")

    except Exception as exc:
        logger.error(f"❌ Erreur backtest: {exc}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
