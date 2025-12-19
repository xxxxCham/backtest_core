"""
Backtest Core - Trade Simulator
================================

Simulation de l'exécution des trades avec gestion des positions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """
    Représentation d'un trade exécuté.
    """
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    exit_reason: str
    leverage: float = 1.0
    fees_paid: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "side": self.side,
            "price_entry": self.entry_price,
            "price_exit": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "leverage_used": self.leverage,
            "fees_paid": self.fees_paid
        }


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    params: Dict[str, Any],
    execution_engine: Optional[Any] = None
) -> pd.DataFrame:
    """
    Simule l'exécution des trades basée sur les signaux.

    Features:
    - Gestion des positions (une seule à la fois)
    - Stop-loss configurable
    - Calcul des frais et slippage (fixe ou dynamique)
    - Exécution réaliste optionnelle (spread/slippage dynamique)
    - Sortie en fin de données si position ouverte

    Args:
        df: DataFrame OHLCV avec index datetime
        signals: Série de signaux (+1, -1, 0)
        params: Paramètres de trading:
            - leverage: Levier (défaut: 3)
            - k_sl: Multiplicateur stop-loss % (défaut: 1.5)
            - initial_capital: Capital initial (défaut: 10000)
            - fees_bps: Frais en bps (défaut: 10)
            - slippage_bps: Slippage en bps (défaut: 5)
            - execution_model: Modèle d'exécution ('fixed', 'dynamic', 'realistic')
        execution_engine: ExecutionEngine optionnel pour exécution réaliste

    Returns:
        DataFrame des trades avec colonnes:
        entry_ts, exit_ts, pnl, size, price_entry, price_exit, side, exit_reason, etc.
    """
    logger.debug("Début simulation des trades")

    trades: List[Trade] = []

    # État de position
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None

    # Paramètres avec défauts
    leverage = params.get("leverage", 3)
    k_sl = params.get("k_sl", 1.5)  # Stop loss en %
    initial_capital = params.get("initial_capital", 10000.0)
    fees_bps = params.get("fees_bps", 10.0)
    slippage_bps = params.get("slippage_bps", 5.0)
    
    # Mode d'exécution
    use_realistic_execution = execution_engine is not None
    if use_realistic_execution:
        execution_engine.prepare(df)
        logger.debug("Mode exécution réaliste activé")

    # Convertir en arrays pour performance
    timestamps = df.index.values
    closes = df["close"].values
    signal_values = signals.values if hasattr(signals, "values") else signals

    n_bars = len(closes)
    
    # Tracking des coûts d'exécution
    total_spread_cost = 0.0
    total_slippage_cost = 0.0

    for i in range(n_bars):
        timestamp = pd.Timestamp(timestamps[i])
        close_price = closes[i]
        signal = signal_values[i]

        # === Entrée en position ===
        if position == 0 and signal != 0:
            position = int(signal)
            
            # Calcul du prix d'entrée
            if use_realistic_execution:
                exec_result = execution_engine.execute_order(
                    price=close_price,
                    side=position,
                    bar_idx=i
                )
                entry_price = exec_result.executed_price
                total_spread_cost += exec_result.spread_cost
                total_slippage_cost += exec_result.slippage_cost
            else:
                # Mode simple: slippage fixe
                slip_factor = 1 + (slippage_bps * 0.0001 * position)
                entry_price = close_price * slip_factor
            
            entry_time = timestamp

            logger.debug(f"Entrée {'LONG' if position == 1 else 'SHORT'} @ {entry_price:.2f}")

        # === En position: vérifier sortie ===
        elif position != 0:
            exit_condition = False
            exit_reason = ""

            # 1. Signal opposé
            if signal != 0 and signal != position:
                exit_condition = True
                exit_reason = "signal_reverse"

            # 2. Stop-loss
            elif position == 1 and close_price <= entry_price * (1 - k_sl * 0.01):
                exit_condition = True
                exit_reason = "stop_loss"
            elif position == -1 and close_price >= entry_price * (1 + k_sl * 0.01):
                exit_condition = True
                exit_reason = "stop_loss"

            # === Exécuter sortie ===
            if exit_condition:
                # Calcul du prix de sortie
                if use_realistic_execution:
                    exec_result = execution_engine.execute_order(
                        price=close_price,
                        side=-position,  # Direction opposée pour la sortie
                        bar_idx=i
                    )
                    exit_price = exec_result.executed_price
                    total_spread_cost += exec_result.spread_cost
                    total_slippage_cost += exec_result.slippage_cost
                else:
                    # Mode simple: slippage fixe
                    slip_factor = 1 - (slippage_bps * 0.0001 * position)
                    exit_price = close_price * slip_factor

                # Calcul PnL
                if position == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price

                # Frais (aller-retour)
                total_fees_pct = fees_bps * 2 * 0.0001
                net_return = raw_return - total_fees_pct

                # PnL avec leverage
                pnl = net_return * leverage * initial_capital

                # Taille de position
                position_size = leverage * initial_capital / entry_price

                # Frais payés
                fees_paid = position_size * entry_price * total_fees_pct

                trade = Trade(
                    entry_ts=entry_time,
                    exit_ts=timestamp,
                    side="LONG" if position == 1 else "SHORT",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    pnl=pnl,
                    return_pct=net_return * 100,
                    exit_reason=exit_reason,
                    leverage=leverage,
                    fees_paid=fees_paid
                )
                trades.append(trade)

                logger.debug(f"Sortie {trade.side} @ {exit_price:.2f}, PnL: ${pnl:.2f}")

                # Reset position
                position = 0
                entry_price = 0.0
                entry_time = None

                # Nouvelle position si signal présent
                if signal != 0:
                    position = int(signal)
                    if use_realistic_execution:
                        exec_result = execution_engine.execute_order(
                            price=close_price,
                            side=position,
                            bar_idx=i
                        )
                        entry_price = exec_result.executed_price
                        total_spread_cost += exec_result.spread_cost
                        total_slippage_cost += exec_result.slippage_cost
                    else:
                        entry_price = close_price * (1 + slippage_bps * 0.0001 * position)
                    entry_time = timestamp

    # === Trade final si position ouverte ===
    if position != 0 and entry_time is not None:
        if use_realistic_execution:
            exec_result = execution_engine.execute_order(
                price=closes[-1],
                side=-position,
                bar_idx=n_bars - 1
            )
            final_price = exec_result.executed_price
            total_spread_cost += exec_result.spread_cost
            total_slippage_cost += exec_result.slippage_cost
        else:
            final_price = closes[-1] * (1 - slippage_bps * 0.0001 * position)
        
        final_time = pd.Timestamp(timestamps[-1])

        if position == 1:
            raw_return = (final_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - final_price) / entry_price

        total_fees_pct = fees_bps * 2 * 0.0001
        net_return = raw_return - total_fees_pct
        pnl = net_return * leverage * initial_capital
        position_size = leverage * initial_capital / entry_price

        trade = Trade(
            entry_ts=entry_time,
            exit_ts=final_time,
            side="LONG" if position == 1 else "SHORT",
            entry_price=entry_price,
            exit_price=final_price,
            size=position_size,
            pnl=pnl,
            return_pct=net_return * 100,
            exit_reason="end_of_data",
            leverage=leverage,
            fees_paid=position_size * entry_price * total_fees_pct
        )
        trades.append(trade)

    # Construire DataFrame
    if trades:
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
    else:
        # DataFrame vide avec colonnes requises
        trades_df = pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "pnl", "size", "price_entry", "price_exit",
            "side", "exit_reason", "return_pct", "leverage_used", "fees_paid"
        ])

    logger.info(f"Simulation terminée: {len(trades)} trades")

    return trades_df


def calculate_equity_curve(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float = 10000.0
) -> pd.Series:
    """
    Calcule la courbe d'équité à partir des trades.

    Args:
        df: DataFrame OHLCV (pour l'index temporel)
        trades_df: DataFrame des trades
        initial_capital: Capital initial

    Returns:
        pd.Series de l'équité au cours du temps
    """
    equity = pd.Series(initial_capital, index=df.index, dtype=np.float64)

    if trades_df.empty:
        return equity

    # ⚠️ FIX: Harmoniser les timezones AVANT la boucle
    exit_ts_series = pd.to_datetime(trades_df["exit_ts"])
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        if exit_ts_series.dt.tz is None:
            # exit_ts naive, df.index aware → localiser
            exit_ts_series = exit_ts_series.dt.tz_localize(df.index.tz)
        elif exit_ts_series.dt.tz != df.index.tz:
            # Timezones différentes → convertir
            exit_ts_series = exit_ts_series.dt.tz_convert(df.index.tz)

    capital = initial_capital

    for idx, (_, trade) in enumerate(trades_df.iterrows()):
        exit_ts = exit_ts_series.iloc[idx]
        pnl = trade["pnl"]

        # Mettre à jour l'équité à partir de la sortie du trade
        mask = (df.index >= exit_ts)
        capital += pnl
        equity[mask] = capital

    return equity


def calculate_returns(equity: pd.Series) -> pd.Series:
    """
    Calcule les rendements périodiques à partir de la courbe d'équité.
    """
    returns = equity.pct_change().fillna(0)
    return returns


__all__ = ["simulate_trades", "Trade", "calculate_equity_curve", "calculate_returns"]
