"""
PATCH D'OPTIMISATION: Pré-allocation buffer pour kernels Numba

Ce patch élimine les allocations mémoire dans les boucles prange.

AVANT:
- 1.7M combos × 116k bars × 8 bytes = 1.5TB d'allocations cumulées
- Cache thrashing
- GC pressure

APRÈS:
- 1.7M × 116k × 8 bytes = 15.8GB pré-alloués (1x au début)
- Zéro allocations pendant sweep
- Meilleure localité cache

GAIN ATTENDU: +5-15% throughput

Usage:
    1. Backup actuel: cp backtest/sweep_numba.py backtest/sweep_numba.py.backup
    2. Appliquer patch: python .tmp_patch_numba_prealloc.py
    3. Tester: python tools/validate_numba_compilation.py
    4. Benchmark: python -m pytest tests/test_sweep_numba.py -v
"""

OPTIMIZED_BOLLINGER_KERNEL = '''
@njit(cache=True, nogil=True, fastmath=True, boundscheck=False, parallel=True)
def _sweep_bollinger_full(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    bb_periods: np.ndarray,
    bb_stds: np.ndarray,
    entry_zs: np.ndarray,
    leverages: np.ndarray,
    k_sls: np.ndarray,
    initial_capital: float,
    fees_bps: float,
    slippage_bps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep COMPLET en un seul kernel Numba parallèle.
    Calcule signaux + backtest pour chaque combo en parallèle.

    OPTIMISÉ: Pré-allocation buffer pour éliminer allocations dans prange.
    """
    n_combos = len(bb_periods)
    n_bars = len(closes)

    # ✅ PRÉ-ALLOCATION UNIQUE pour tous les threads
    # Mémoire: n_combos × n_bars × 8 bytes
    # Ex: 1.7M × 116k × 8 = 15.8 GB (OK pour DDR5 60GB)
    signals_buffer = np.zeros((n_combos, n_bars), dtype=np.float64)

    # Résultats
    total_pnls = np.zeros(n_combos, dtype=np.float64)
    sharpes = np.zeros(n_combos, dtype=np.float64)
    max_dds = np.zeros(n_combos, dtype=np.float64)
    win_rates = np.zeros(n_combos, dtype=np.float64)
    n_trades_out = np.zeros(n_combos, dtype=np.int64)

    slippage_factor = slippage_bps * 0.0001
    fees_factor = fees_bps * 2 * 0.0001

    # ⚡ PARALLÉLISATION sur les combinaisons
    for combo_idx in prange(n_combos):
        # ✅ Réutiliser buffer pré-alloué (vue sur slice, pas d'allocation)
        signals = signals_buffer[combo_idx]

        bb_period = int(bb_periods[combo_idx])
        entry_z = entry_zs[combo_idx]
        leverage = leverages[combo_idx]
        k_sl = k_sls[combo_idx]
        sl_pct = k_sl * 0.01

        # Réinitialiser signaux (buffer déjà alloué)
        for j in range(n_bars):
            signals[j] = 0.0

        # === Calcul signaux Bollinger inline ===
        for i in range(bb_period, n_bars):
            sma = 0.0
            for j in range(bb_period):
                sma += closes[i - bb_period + 1 + j]
            sma /= bb_period

            var = 0.0
            for j in range(bb_period):
                diff = closes[i - bb_period + 1 + j] - sma
                var += diff * diff
            std = np.sqrt(var / bb_period)

            if std > 1e-10:
                z_score = (closes[i] - sma) / std
                if z_score < -entry_z:
                    signals[i] = 1.0
                elif z_score > entry_z:
                    signals[i] = -1.0

        # === Simulation backtest (IDENTIQUE) ===
        position = 0
        entry_price = 0.0
        equity = initial_capital
        peak_equity = initial_capital
        max_dd = 0.0
        trade_count = 0
        winning_trades = 0
        returns_sum = 0.0
        returns_sq_sum = 0.0

        for i in range(n_bars):
            close_price = closes[i]
            signal = signals[i]

            if position == 0 and signal != 0:
                position = int(signal)
                entry_price = close_price * (1.0 + slippage_factor * position)

            elif position != 0:
                exit_now = False

                if signal != 0 and signal != position:
                    exit_now = True
                elif position == 1 and lows[i] <= entry_price * (1.0 - sl_pct):
                    exit_now = True
                elif position == -1 and highs[i] >= entry_price * (1.0 + sl_pct):
                    exit_now = True

                if exit_now:
                    exit_price = close_price * (1.0 - slippage_factor * position)

                    if position == 1:
                        raw_return = (exit_price - entry_price) / entry_price
                    else:
                        raw_return = (entry_price - exit_price) / entry_price

                    net_return = raw_return - fees_factor
                    pnl = net_return * leverage * initial_capital

                    equity += pnl
                    trade_count += 1
                    if pnl > 0:
                        winning_trades += 1

                    returns_sum += net_return
                    returns_sq_sum += net_return * net_return

                    if equity > peak_equity:
                        peak_equity = equity
                    dd = (peak_equity - equity) / peak_equity * 100.0
                    if dd > max_dd:
                        max_dd = dd

                    position = 0
                    entry_price = 0.0
                    if signal != 0:
                        position = int(signal)
                        entry_price = close_price * (1.0 + slippage_factor * position)

        # Clôturer position ouverte
        if position != 0:
            exit_price = closes[-1] * (1.0 - slippage_factor * position)
            if position == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            net_return = raw_return - fees_factor
            pnl = net_return * leverage * initial_capital
            equity += pnl
            trade_count += 1
            if pnl > 0:
                winning_trades += 1
            returns_sum += net_return
            returns_sq_sum += net_return * net_return

        # Métriques finales
        total_pnls[combo_idx] = equity - initial_capital
        n_trades_out[combo_idx] = trade_count

        if trade_count > 0:
            win_rates[combo_idx] = (winning_trades / trade_count) * 100.0
            mean_ret = returns_sum / trade_count
            if trade_count > 1:
                variance = (returns_sq_sum / trade_count) - (mean_ret * mean_ret)
                if variance > 0:
                    std_ret = np.sqrt(variance)
                    sharpes[combo_idx] = mean_ret / std_ret * np.sqrt(252)

        max_dds[combo_idx] = max_dd

    return total_pnls, sharpes, max_dds, win_rates, n_trades_out
'''


def apply_patch():
    """Applique le patch d'optimisation."""
    import shutil
    from pathlib import Path

    sweep_file = Path("backtest/sweep_numba.py")
    backup_file = Path("backtest/sweep_numba.py.prealloc_backup")

    if not sweep_file.exists():
        print("❌ Fichier backtest/sweep_numba.py non trouvé")
        return False

    # Backup
    shutil.copy(sweep_file, backup_file)
    print(f"✅ Backup créé: {backup_file}")

    # Lire fichier actuel
    content = sweep_file.read_text(encoding="utf-8")

    # Remplacer kernel Bollinger
    # Chercher la fonction @njit _sweep_bollinger_full
    import re

    pattern = r'(@njit\(cache=True.*?\)\s+def _sweep_bollinger_full\(.*?\n.*?return total_pnls, sharpes, max_dds, win_rates, n_trades_out)'

    if re.search(pattern, content, re.DOTALL):
        print("⚠️ ATTENTION: Patch nécessite édition manuelle")
        print("   Raison: Regex complexe pour remplacer fonction complète")
        print("\n📝 INSTRUCTIONS MANUELLES:")
        print("   1. Ouvrir backtest/sweep_numba.py")
        print("   2. Localiser la fonction _sweep_bollinger_full (ligne ~83)")
        print("   3. Remplacer par le code dans .tmp_patch_numba_prealloc.py")
        print("   4. Ajouter avant la boucle prange:")
        print("      signals_buffer = np.zeros((n_combos, n_bars), dtype=np.float64)")
        print("   5. Dans la boucle, remplacer:")
        print("      signals = np.zeros(n_bars, dtype=np.float64)")
        print("      par:")
        print("      signals = signals_buffer[combo_idx]")
        print("      for j in range(n_bars): signals[j] = 0.0")
        return False
    else:
        print("❌ Pattern _sweep_bollinger_full non trouvé")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("PATCH NUMBA PRÉ-ALLOCATION")
    print("=" * 70)
    print()
    print("Ce patch optimise les kernels Numba pour éliminer les allocations")
    print("mémoire dans les boucles prange.")
    print()
    print("GAIN ATTENDU: +5-15% throughput (10k → 10.5-11.5k bt/s)")
    print("COÛT RAM: +15.8GB pour 1.7M combos × 116k bars")
    print()
    print("=" * 70)
    print()

    # Vérifier RAM disponible
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024**3)
        print(f"RAM disponible: {ram_gb:.1f} GB")

        if ram_gb < 20:
            print("⚠️ ATTENTION: RAM <20GB, patch non recommandé")
            print("   Utilisez plutôt un chunking manuel")
            exit(1)
        else:
            print(f"✅ RAM suffisante ({ram_gb:.1f} GB > 20GB)")
    except:
        print("⚠️ Impossible de vérifier RAM (psutil)")

    print()
    print("=" * 70)
    print()
    print("📝 PATCH MANUEL REQUIS")
    print()
    print("Voir code optimisé dans cette variable:")
    print("  OPTIMIZED_BOLLINGER_KERNEL")
    print()
    print("Ou consulter le diagnostic complet:")
    print("  .tmp_diagnostic_numba_performance.md")
    print()
    print("=" * 70)
