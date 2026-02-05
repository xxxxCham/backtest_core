#!/usr/bin/env python
"""
Test minimaliste Streamlit + Numba sweep.
Permet d'isoler si le crash vient de l'UI ou du code Numba.
"""
import os
# Configuration CPU AVANT imports
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil

st.set_page_config(page_title="Test Numba Sweep", page_icon="⚡", layout="wide")

st.title("⚡ Test Numba Sweep Minimaliste")

# Afficher info système
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CPU Cores", f"{psutil.cpu_count(logical=False)} physiques / {psutil.cpu_count(logical=True)} logiques")
with col2:
    mem = psutil.virtual_memory()
    st.metric("RAM Disponible", f"{mem.available / (1024**3):.1f} GB")
with col3:
    st.metric("NUMBA_NUM_THREADS", os.environ.get("NUMBA_NUM_THREADS", "N/A"))

st.divider()

# Paramètres de test
n_combos = st.slider("Nombre de combinaisons", min_value=10, max_value=10000, value=100, step=10)
n_bars = st.slider("Nombre de barres", min_value=1000, max_value=50000, value=10000, step=1000)
strategy = st.selectbox("Stratégie", ["bollinger_atr", "ema_cross", "rsi_reversal"])

if st.button("🚀 Lancer Test Numba"):
    status = st.empty()
    progress_bar = st.progress(0)
    result_area = st.empty()

    try:
        # Étape 1: Générer données
        status.info("📊 Génération données synthétiques...")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n_bars) * 0.001),
            'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            'close': close,
            'volume': np.random.randint(1000, 10000, n_bars),
        })
        progress_bar.progress(20)

        # Étape 2: Import Numba (peut déclencher JIT)
        status.info("⚙️ Import module Numba...")
        t0 = time.perf_counter()
        from backtest.sweep_numba import run_numba_sweep, is_numba_supported
        t1 = time.perf_counter()
        st.write(f"Import: {t1-t0:.2f}s")
        progress_bar.progress(40)

        # Étape 3: Générer grille
        status.info(f"📋 Génération grille ({n_combos} combos)...")
        if strategy == "bollinger_atr":
            param_grid = [
                {'bb_period': float(20 + i % 20), 'bb_std': float(1.5 + (i % 10) * 0.3),
                 'entry_z': 2.0, 'leverage': 1.0, 'k_sl': 1.5}
                for i in range(n_combos)
            ]
        elif strategy == "ema_cross":
            param_grid = [
                {'fast_period': float(5 + i % 15), 'slow_period': float(20 + (i % 20)),
                 'leverage': 1.0, 'k_sl': 1.5}
                for i in range(n_combos)
            ]
        else:  # rsi_reversal
            param_grid = [
                {'rsi_period': float(7 + i % 20), 'overbought': float(70 + (i % 10)),
                 'oversold': float(30 - (i % 10)), 'leverage': 1.0, 'k_sl': 1.5}
                for i in range(n_combos)
            ]
        progress_bar.progress(60)

        # Étape 4: Exécuter sweep
        status.info(f"⚡ Exécution sweep Numba ({n_combos} × {n_bars})...")
        t2 = time.perf_counter()
        results = run_numba_sweep(
            df=df,
            strategy_key=strategy,
            param_grid=param_grid,
            initial_capital=10000.0,
            fees_bps=10.0,
            slippage_bps=5.0,
        )
        t3 = time.perf_counter()
        progress_bar.progress(100)

        # Résultats
        elapsed = t3 - t2
        throughput = n_combos / elapsed if elapsed > 0 else 0
        best = max(results, key=lambda r: r.get("total_pnl", float("-inf")))

        status.success(f"✅ Terminé en {elapsed:.2f}s ({throughput:,.0f} bt/s)")

        result_area.markdown(f"""
        ### Résultats
        - **Combinaisons testées**: {n_combos:,}
        - **Temps d'exécution**: {elapsed:.3f}s
        - **Throughput**: {throughput:,.0f} backtests/seconde
        - **Meilleur PnL**: ${best['total_pnl']:+,.2f}
        - **Sharpe (meilleur)**: {best['sharpe_ratio']:.2f}
        """)

    except Exception as e:
        import traceback
        status.error(f"❌ Erreur: {e}")
        st.code(traceback.format_exc())
