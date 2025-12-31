"""
Validation de l'intégrité des résultats de backtest.

Vérifie :
- Cohérence mathématique (P&L, equity)
- Trades complets vs incomplets
- Timestamps valides
- Pas de valeurs aberrantes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402


def validate_backtest_result(result, df, initial_capital=10000.0, verbose=True):
    """
    Valide l'intégrité d'un résultat de backtest.

    Returns:
        dict: {'valid': bool, 'errors': list, 'warnings': list}
    """
    errors = []
    warnings = []

    if verbose:
        print("=" * 80)
        print("🔍 VALIDATION DE L'INTÉGRITÉ DU BACKTEST")
        print("=" * 80)
        print()

    # 1. Vérifier que trades_df existe et n'est pas vide
    if result.trades.empty:
        warnings.append("Aucun trade généré")
        if verbose:
            print("⚠️  Aucun trade généré (peut être normal selon paramètres)")
        return {'valid': True, 'errors': errors, 'warnings': warnings}

    trades_df = result.trades
    metrics = result.metrics
    equity = result.equity

    if verbose:
        print("📊 Résumé:")
        print(f"   Trades: {len(trades_df)}")
        print(f"   Période: {df.index[0]} → {df.index[-1]}")
        print(f"   Durée: {(df.index[-1] - df.index[0]).days} jours")
        print()

    # 2. Vérifier que total_trades est un entier
    total_trades = metrics.get('total_trades', len(trades_df))
    if not isinstance(total_trades, (int, np.integer)):
        errors.append(f"total_trades n'est pas un entier : {total_trades} (type: {type(total_trades)})")

    if total_trades != len(trades_df):
        errors.append(f"total_trades ({total_trades}) ≠ len(trades_df) ({len(trades_df)})")

    if verbose:
        print("✓ Test 1: Nombre de trades")
        print(f"   total_trades = {total_trades} (type: {type(total_trades).__name__})")
        print(f"   len(trades_df) = {len(trades_df)}")
        if errors:
            print(f"   ❌ {errors[-1]}")
        else:
            print("   ✅ OK")
        print()

    # 3. Vérifier timestamps valides
    if verbose:
        print("✓ Test 2: Validité des timestamps")

    df_start = df.index[0]
    df_end = df.index[-1]

    # Harmoniser les timezones pour comparaison
    entry_ts = pd.to_datetime(trades_df['entry_ts'])
    exit_ts = pd.to_datetime(trades_df['exit_ts'])

    if hasattr(df.index, 'tz') and df.index.tz is not None:
        if entry_ts.dt.tz is None:
            entry_ts = entry_ts.dt.tz_localize(df.index.tz)
        elif entry_ts.dt.tz != df.index.tz:
            entry_ts = entry_ts.dt.tz_convert(df.index.tz)

        if exit_ts.dt.tz is None:
            exit_ts = exit_ts.dt.tz_localize(df.index.tz)
        elif exit_ts.dt.tz != df.index.tz:
            exit_ts = exit_ts.dt.tz_convert(df.index.tz)

    invalid_entries = trades_df[
        (entry_ts < df_start) | (entry_ts > df_end)
    ]
    if len(invalid_entries) > 0:
        errors.append(f"{len(invalid_entries)} trades avec entry_ts hors période")
        if verbose:
            print(f"   ❌ {len(invalid_entries)} entry_ts hors période:")
            for idx, row in invalid_entries.head(3).iterrows():
                print(f"      {row['entry_ts']}")

    invalid_exits = trades_df[
        (exit_ts < df_start) | (exit_ts > df_end)
    ]
    if len(invalid_exits) > 0:
        errors.append(f"{len(invalid_exits)} trades avec exit_ts hors période")
        if verbose:
            print(f"   ❌ {len(invalid_exits)} exit_ts hors période:")
            for idx, row in invalid_exits.head(3).iterrows():
                print(f"      {row['exit_ts']}")

    # Trades incomplets (exit après la fin des données)
    incomplete_trades = trades_df[exit_ts == df_end]
    if len(incomplete_trades) > 0:
        warnings.append(f"{len(incomplete_trades)} trades incomplets (fermés à la fin)")
        if verbose:
            print(f"   ⚠️  {len(incomplete_trades)} trades fermés à la dernière barre")
            print("      (Position ouverte en fin de backtest)")

    if not errors and not warnings:
        if verbose:
            print("   ✅ Tous les timestamps valides")

    print()

    # 4. Cohérence P&L
    if verbose:
        print("✓ Test 3: Cohérence P&L")

    # P&L individuel
    sum_pnl_trades = trades_df['pnl'].sum()

    # P&L dans metrics (peut être total_return ou total_pnl)
    metrics_pnl = metrics.get('total_pnl', None)
    if metrics_pnl is None:
        # Essayer total_return
        metrics_pnl = metrics.get('total_return', None)
        if metrics_pnl is not None:
            if verbose:
                print("   ℹ️  'total_pnl' absent, utilise 'total_return'")

    if metrics_pnl is not None:
        diff_pnl = abs(sum_pnl_trades - metrics_pnl)
        if diff_pnl > 0.01:  # Tolérance 1 cent
            errors.append(f"Incohérence P&L: sum(trades)={sum_pnl_trades:.2f} ≠ metrics={metrics_pnl:.2f} (diff={diff_pnl:.2f})")
            if verbose:
                print(f"   ❌ sum(trades.pnl) = {sum_pnl_trades:.2f}")
                print(f"      metrics P&L = {metrics_pnl:.2f}")
                print(f"      Différence = {diff_pnl:.2f}")
        else:
            if verbose:
                print(f"   ✅ sum(trades.pnl) = {sum_pnl_trades:.2f}")
                print(f"      metrics P&L = {metrics_pnl:.2f}")
    else:
        warnings.append("Impossible de vérifier P&L (metric absent)")

    print()

    # 5. Cohérence Equity
    if verbose:
        print("✓ Test 4: Cohérence Equity")

    equity_start = equity.iloc[0]
    equity_end = equity.iloc[-1]

    # Vérifier que equity démarre à initial_capital
    if abs(equity_start - initial_capital) > 0.01:
        errors.append(f"Equity start={equity_start:.2f} ≠ initial_capital={initial_capital:.2f}")
        if verbose:
            print(f"   ❌ equity[0] = {equity_start:.2f} (attendu: {initial_capital:.2f})")
    else:
        if verbose:
            print(f"   ✅ equity[0] = {equity_start:.2f} (= initial_capital)")

    # Vérifier que equity finale = capital + P&L
    expected_final = initial_capital + sum_pnl_trades
    diff_equity = abs(equity_end - expected_final)

    if diff_equity > 0.01:
        errors.append(f"Equity finale={equity_end:.2f} ≠ capital+PnL={expected_final:.2f} (diff={diff_equity:.2f})")
        if verbose:
            print(f"   ❌ equity[-1] = {equity_end:.2f}")
            print(f"      capital + sum(pnl) = {expected_final:.2f}")
            print(f"      Différence = {diff_equity:.2f}")
    else:
        if verbose:
            print(f"   ✅ equity[-1] = {equity_end:.2f}")
            print(f"      capital + sum(pnl) = {expected_final:.2f}")

    print()

    # 6. Vérifier que l'equity varie (mark-to-market fonctionne)
    if verbose:
        print("✓ Test 5: Mark-to-Market")

    # Compter combien de barres ont une equity différente
    equity_changes = (equity.diff().fillna(0) != 0).sum()
    equity_unique_values = equity.nunique()

    if equity_unique_values < 10:
        warnings.append(f"Equity a seulement {equity_unique_values} valeurs uniques (MTM suspect)")
        if verbose:
            print(f"   ⚠️  Equity a {equity_unique_values} valeurs uniques sur {len(equity)} barres")
            print("      (Mark-to-market pourrait ne pas fonctionner)")
    else:
        if verbose:
            print(f"   ✅ Equity varie correctement : {equity_changes}/{len(equity)} changements")

    print()

    # 7. Vérifier entry < exit pour chaque trade
    if verbose:
        print("✓ Test 6: Ordre des timestamps")

    invalid_order = trades_df[entry_ts >= exit_ts]
    if len(invalid_order) > 0:
        errors.append(f"{len(invalid_order)} trades avec entry_ts >= exit_ts")
        if verbose:
            print(f"   ❌ {len(invalid_order)} trades avec entry >= exit:")
            for idx, row in invalid_order.head(3).iterrows():
                print(f"      Entry: {row['entry_ts']}, Exit: {row['exit_ts']}")
    else:
        if verbose:
            print("   ✅ Tous les trades ont entry < exit")

    print()

    # Résumé
    if verbose:
        print("=" * 80)
        print("📋 RÉSUMÉ")
        print("=" * 80)

        if errors:
            print(f"❌ {len(errors)} ERREUR(S) DÉTECTÉE(S):")
            for i, err in enumerate(errors, 1):
                print(f"   {i}. {err}")
        else:
            print("✅ Aucune erreur détectée")

        print()

        if warnings:
            print(f"⚠️  {len(warnings)} WARNING(S):")
            for i, warn in enumerate(warnings, 1):
                print(f"   {i}. {warn}")
        else:
            print("✅ Aucun warning")

        print()

    valid = len(errors) == 0
    return {'valid': valid, 'errors': errors, 'warnings': warnings}


if __name__ == "__main__":
    print("🧪 TEST DE VALIDATION D'INTÉGRITÉ")
    print()

    # Charger données de test
    try:
        df = pd.read_csv('data/sample_data/BTCUSDT_1h_6months.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("❌ Fichier BTCUSDT_1h_6months.csv introuvable")
        print("   Utilisez tools/generate_6month_data.py pour le créer")
        sys.exit(1)

    print(f"📊 Données: {len(df)} barres, {df.index[0]} → {df.index[-1]}")
    print()

    # Tester avec 3 configurations différentes
    test_configs = [
        {'atr_period': 14, 'atr_mult': 2.0, 'leverage': 1},
        {'atr_period': 10, 'atr_mult': 1.5, 'leverage': 1},
        {'atr_period': 30, 'atr_mult': 4.0, 'leverage': 1},
    ]

    all_valid = True

    for i, params in enumerate(test_configs, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/3: {params}")
        print(f"{'=' * 80}\n")

        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(df=df, strategy='atr_channel', params=params, timeframe='1h')

        validation = validate_backtest_result(result, df, initial_capital=10000, verbose=True)

        if not validation['valid']:
            all_valid = False
            print(f"\n❌ TEST {i} ÉCHOUÉ")
        else:
            print(f"\n✅ TEST {i} RÉUSSI")

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ TOUS LES TESTS RÉUSSIS - Intégrité vérifiée")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ - Vérifier les erreurs ci-dessus")
    print("=" * 80)
