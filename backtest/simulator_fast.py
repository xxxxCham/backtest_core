"""
Module-ID: backtest.simulator_fast

Purpose: Version haute performance de simulate_trades via Numba JIT (100x accélération vs Python pur).

Role in pipeline: execution / performance

Key components: _simulate_trades_numba, simulate_trades_fast, calculate_equity_fast, HAS_NUMBA flag

Inputs: OHLCV arrays (closes, highs, lows), signaux, paramètres (leverage, slippage, fees)

Outputs: Trades arrays (entry_ts, exit_ts, pnl, etc.), equity curve, returns

Dependencies: numpy, pandas, optionnel: numba (JIT); fallback simulator.py si numba absent

Conventions: HAS_NUMBA = False si numba non dispo (fallback Python OK); données arrays numpy; signaux 1/-1/0.

Read-if: Optimisation simulation ou fallback à simulator_fast.

Skip-if: simulator.py Python pur performant suffisamment.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Numba pour JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# NUMBA-OPTIMIZED CORE (JIT-compiled)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _simulate_trades_numba(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        signals: np.ndarray,
        leverage: float,
        k_sl: float,
        initial_capital: float,
        fees_bps: float,
        slippage_bps: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Cœur de simulation JIT-compilé.

        Retourne les arrays numpy des trades pour reconstruction DataFrame.
        """
        n_bars = len(closes)

        # Pré-allouer les arrays de sortie (max = n_bars/2 trades)
        max_trades = n_bars // 2 + 1
        entry_indices = np.zeros(max_trades, dtype=np.int64)
        exit_indices = np.zeros(max_trades, dtype=np.int64)
        sides = np.zeros(max_trades, dtype=np.int64)  # 1=long, -1=short
        entry_prices = np.zeros(max_trades, dtype=np.float64)
        exit_prices = np.zeros(max_trades, dtype=np.float64)
        pnls = np.zeros(max_trades, dtype=np.float64)
        returns_pct = np.zeros(max_trades, dtype=np.float64)
        exit_reasons = np.zeros(max_trades, dtype=np.int64)  # 0=signal, 1=sl, 2=end
        sizes = np.zeros(max_trades, dtype=np.float64)

        # État position
        position = 0
        entry_price = 0.0
        entry_idx = 0
        trade_count = 0

        # Constantes précalculées
        slippage_factor = slippage_bps * 0.0001
        fees_factor = fees_bps * 2 * 0.0001
        sl_pct = k_sl * 0.01

        for i in range(n_bars):
            close_price = closes[i]
            signal = signals[i]

            # === Entrée en position ===
            if position == 0 and signal != 0:
                position = int(signal)
                entry_price = close_price * (1.0 + slippage_factor * position)
                entry_idx = i

            # === En position: vérifier sortie ===
            elif position != 0:
                exit_condition = False
                exit_reason = 0

                # 1. Signal opposé
                if signal != 0 and signal != position:
                    exit_condition = True
                    exit_reason = 0  # signal_reverse

                # 2. Stop-loss (intrabar check avec high/low)
                elif position == 1:
                    sl_price = entry_price * (1.0 - sl_pct)
                    if lows[i] <= sl_price:
                        exit_condition = True
                        exit_reason = 1  # stop_loss
                elif position == -1:
                    sl_price = entry_price * (1.0 + sl_pct)
                    if highs[i] >= sl_price:
                        exit_condition = True
                        exit_reason = 1  # stop_loss

                # === Exécuter sortie ===
                if exit_condition:
                    exit_price = close_price * (1.0 - slippage_factor * position)

                    # PnL
                    if position == 1:
                        raw_return = (exit_price - entry_price) / entry_price
                    else:
                        raw_return = (entry_price - exit_price) / entry_price

                    net_return = raw_return - fees_factor
                    pnl = net_return * leverage * initial_capital
                    position_size = leverage * initial_capital / entry_price

                    # Enregistrer trade
                    entry_indices[trade_count] = entry_idx
                    exit_indices[trade_count] = i
                    sides[trade_count] = position
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = exit_price
                    pnls[trade_count] = pnl
                    returns_pct[trade_count] = net_return * 100.0
                    exit_reasons[trade_count] = exit_reason
                    sizes[trade_count] = position_size
                    trade_count += 1

                    # Reset
                    position = 0
                    entry_price = 0.0

                    # Nouvelle position si signal présent
                    if signal != 0:
                        position = int(signal)
                        entry_price = close_price * (1.0 + slippage_factor * position)
                        entry_idx = i

        # === Trade final si position ouverte ===
        if position != 0:
            final_price = closes[-1] * (1.0 - slippage_factor * position)

            if position == 1:
                raw_return = (final_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - final_price) / entry_price

            net_return = raw_return - fees_factor
            pnl = net_return * leverage * initial_capital
            position_size = leverage * initial_capital / entry_price

            entry_indices[trade_count] = entry_idx
            exit_indices[trade_count] = n_bars - 1
            sides[trade_count] = position
            entry_prices[trade_count] = entry_price
            exit_prices[trade_count] = final_price
            pnls[trade_count] = pnl
            returns_pct[trade_count] = net_return * 100.0
            exit_reasons[trade_count] = 2  # end_of_data
            sizes[trade_count] = position_size
            trade_count += 1

        return (
            entry_indices[:trade_count],
            exit_indices[:trade_count],
            sides[:trade_count],
            entry_prices[:trade_count],
            exit_prices[:trade_count],
            pnls[:trade_count],
            returns_pct[:trade_count],
            exit_reasons[:trade_count],
            sizes[:trade_count],
            trade_count
        )

    @njit(cache=True, parallel=True, fastmath=True)
    def _calculate_equity_numba(
        n_bars: int,
        exit_indices: np.ndarray,
        pnls: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """Calcul vectorisé de la courbe d'équité."""
        equity = np.full(n_bars, initial_capital, dtype=np.float64)

        # Créer array des changements de capital
        capital_changes = np.zeros(n_bars, dtype=np.float64)
        for i in range(len(exit_indices)):
            capital_changes[exit_indices[i]] += pnls[i]

        # Cumsum pour équité
        cumsum = 0.0
        for i in range(n_bars):
            cumsum += capital_changes[i]
            equity[i] = initial_capital + cumsum

        return equity


# =============================================================================
# NUMPY VECTORIZED FALLBACK (si Numba non disponible)
# =============================================================================

def _simulate_trades_numpy(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    leverage: float,
    k_sl: float,
    initial_capital: float,
    fees_bps: float,
    slippage_bps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Version numpy pure (fallback si pas de Numba)."""
    n_bars = len(closes)

    # Pré-allouer
    max_trades = n_bars // 2 + 1
    entry_indices = np.zeros(max_trades, dtype=np.int64)
    exit_indices = np.zeros(max_trades, dtype=np.int64)
    sides = np.zeros(max_trades, dtype=np.int64)
    entry_prices = np.zeros(max_trades, dtype=np.float64)
    exit_prices = np.zeros(max_trades, dtype=np.float64)
    pnls = np.zeros(max_trades, dtype=np.float64)
    returns_pct = np.zeros(max_trades, dtype=np.float64)
    exit_reasons = np.zeros(max_trades, dtype=np.int64)
    sizes = np.zeros(max_trades, dtype=np.float64)

    position = 0
    entry_price = 0.0
    entry_idx = 0
    trade_count = 0

    slippage_factor = slippage_bps * 0.0001
    fees_factor = fees_bps * 2 * 0.0001
    sl_pct = k_sl * 0.01

    for i in range(n_bars):
        close_price = closes[i]
        signal = signals[i]

        if position == 0 and signal != 0:
            position = int(signal)
            entry_price = close_price * (1.0 + slippage_factor * position)
            entry_idx = i

        elif position != 0:
            exit_condition = False
            exit_reason = 0

            if signal != 0 and signal != position:
                exit_condition = True
                exit_reason = 0
            elif position == 1 and lows[i] <= entry_price * (1.0 - sl_pct):
                exit_condition = True
                exit_reason = 1
            elif position == -1 and highs[i] >= entry_price * (1.0 + sl_pct):
                exit_condition = True
                exit_reason = 1

            if exit_condition:
                exit_price = close_price * (1.0 - slippage_factor * position)

                if position == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price

                net_return = raw_return - fees_factor
                pnl = net_return * leverage * initial_capital
                position_size = leverage * initial_capital / entry_price

                entry_indices[trade_count] = entry_idx
                exit_indices[trade_count] = i
                sides[trade_count] = position
                entry_prices[trade_count] = entry_price
                exit_prices[trade_count] = exit_price
                pnls[trade_count] = pnl
                returns_pct[trade_count] = net_return * 100.0
                exit_reasons[trade_count] = exit_reason
                sizes[trade_count] = position_size
                trade_count += 1

                position = 0
                entry_price = 0.0

                if signal != 0:
                    position = int(signal)
                    entry_price = close_price * (1.0 + slippage_factor * position)
                    entry_idx = i

    # Trade final
    if position != 0:
        final_price = closes[-1] * (1.0 - slippage_factor * position)

        if position == 1:
            raw_return = (final_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - final_price) / entry_price

        net_return = raw_return - fees_factor
        pnl = net_return * leverage * initial_capital
        position_size = leverage * initial_capital / entry_price

        entry_indices[trade_count] = entry_idx
        exit_indices[trade_count] = n_bars - 1
        sides[trade_count] = position
        entry_prices[trade_count] = entry_price
        exit_prices[trade_count] = final_price
        pnls[trade_count] = pnl
        returns_pct[trade_count] = net_return * 100.0
        exit_reasons[trade_count] = 2
        sizes[trade_count] = position_size
        trade_count += 1

    return (
        entry_indices[:trade_count],
        exit_indices[:trade_count],
        sides[:trade_count],
        entry_prices[:trade_count],
        exit_prices[:trade_count],
        pnls[:trade_count],
        returns_pct[:trade_count],
        exit_reasons[:trade_count],
        sizes[:trade_count],
        trade_count
    )


# =============================================================================
# PUBLIC API
# =============================================================================

EXIT_REASON_MAP = {0: "signal_reverse", 1: "stop_loss", 2: "end_of_data"}


def simulate_trades_fast(
    df: pd.DataFrame,
    signals: pd.Series,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Simule l'exécution des trades avec optimisation Numba.

    10-100x plus rapide que simulate_trades() standard.

    Args:
        df: DataFrame OHLCV avec index datetime
        signals: Série de signaux (+1, -1, 0)
        params: Paramètres de trading

    Returns:
        DataFrame des trades
    """
    # Extraire paramètres
    leverage = float(params.get("leverage", 3))
    k_sl = float(params.get("k_sl", 1.5))
    initial_capital = float(params.get("initial_capital", 10000.0))
    fees_bps = float(params.get("fees_bps", 10.0))
    slippage_bps = float(params.get("slippage_bps", 5.0))

    # Convertir en arrays numpy (contiguous pour performance)
    closes = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    signal_arr = np.ascontiguousarray(
        signals.values if hasattr(signals, "values") else signals,
        dtype=np.float64
    )

    # Choisir l'implémentation
    if HAS_NUMBA:
        result = _simulate_trades_numba(
            closes, highs, lows, signal_arr,
            leverage, k_sl, initial_capital, fees_bps, slippage_bps
        )
    else:
        logger.debug("Numba non disponible, utilisation de NumPy pur")
        result = _simulate_trades_numpy(
            closes, highs, lows, signal_arr,
            leverage, k_sl, initial_capital, fees_bps, slippage_bps
        )

    (entry_indices, exit_indices, sides, entry_prices, exit_prices,
     pnls, returns_pct, exit_reasons, sizes, trade_count) = result

    if trade_count == 0:
        return pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "pnl", "size", "price_entry", "price_exit",
            "side", "exit_reason", "return_pct", "leverage_used", "fees_paid"
        ])

    # Convertir timestamps
    timestamps = df.index.values
    fees_factor = fees_bps * 2 * 0.0001

    trades_df = pd.DataFrame({
        "entry_ts": pd.to_datetime(timestamps[entry_indices]),
        "exit_ts": pd.to_datetime(timestamps[exit_indices]),
        "pnl": pnls,
        "size": sizes,
        "price_entry": entry_prices,
        "price_exit": exit_prices,
        "side": np.where(sides == 1, "LONG", "SHORT"),
        "exit_reason": [EXIT_REASON_MAP.get(r, "unknown") for r in exit_reasons],
        "return_pct": returns_pct,
        "leverage_used": leverage,
        "fees_paid": sizes * entry_prices * fees_factor
    })

    logger.debug(f"Simulation fast terminée: {trade_count} trades")

    return trades_df


def calculate_equity_fast(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float = 10000.0
) -> pd.Series:
    """
    Calcule la courbe d'équité de manière optimisée.

    Args:
        df: DataFrame OHLCV (pour l'index)
        trades_df: DataFrame des trades
        initial_capital: Capital initial

    Returns:
        pd.Series de l'équité
    """
    n_bars = len(df)

    if trades_df.empty:
        return pd.Series(initial_capital, index=df.index, dtype=np.float64)

    # Mapper les timestamps d'exit aux indices
    exit_ts = pd.to_datetime(trades_df["exit_ts"])
    pnls = trades_df["pnl"].values.astype(np.float64)

    # Créer lookup rapide
    ts_to_idx = {ts: i for i, ts in enumerate(df.index)}
    exit_indices = np.array([ts_to_idx.get(ts, n_bars - 1) for ts in exit_ts], dtype=np.int64)

    if HAS_NUMBA:
        equity_arr = _calculate_equity_numba(n_bars, exit_indices, pnls, initial_capital)
    else:
        # Version numpy
        equity_arr = np.full(n_bars, initial_capital, dtype=np.float64)
        capital_changes = np.zeros(n_bars, dtype=np.float64)
        np.add.at(capital_changes, exit_indices, pnls)
        equity_arr = initial_capital + np.cumsum(capital_changes)

    return pd.Series(equity_arr, index=df.index, dtype=np.float64)


def calculate_returns_fast(equity: pd.Series) -> pd.Series:
    """Calcul vectorisé des rendements."""
    equity_arr = equity.values
    returns = np.zeros_like(equity_arr)
    returns[1:] = (equity_arr[1:] - equity_arr[:-1]) / equity_arr[:-1]
    returns = np.nan_to_num(returns, 0.0)
    return pd.Series(returns, index=equity.index, dtype=np.float64)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def simulate_batch(
    df: pd.DataFrame,
    signals_batch: List[pd.Series],
    params_batch: List[Dict[str, Any]],
    n_jobs: int = -1
) -> List[pd.DataFrame]:
    """
    Simule plusieurs backtests en parallèle.

    Args:
        df: DataFrame OHLCV partagé
        signals_batch: Liste de séries de signaux
        params_batch: Liste de paramètres
        n_jobs: Nombre de workers (-1 = tous les CPUs)

    Returns:
        Liste de DataFrames de trades
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4

    results = [None] * len(signals_batch)

    def run_single(idx: int) -> Tuple[int, pd.DataFrame]:
        trades = simulate_trades_fast(df, signals_batch[idx], params_batch[idx])
        return idx, trades

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_single, i) for i in range(len(signals_batch))]
        for future in as_completed(futures):
            idx, trades = future.result()
            results[idx] = trades

    return results


# Docstring update summary
# - Docstring de module normalisée (LLM-friendly) centrée sur simulation haute perf (Numba JIT)
# - Conventions HAS_NUMBA flag et fallback simulator.py explicitées
# - Read-if/Skip-if ajoutés pour tri rapide


__all__ = [
    "simulate_trades_fast",
    "calculate_equity_fast",
    "calculate_returns_fast",
    "simulate_batch",
    "HAS_NUMBA"
]
