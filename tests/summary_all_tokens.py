#!/usr/bin/env python3
"""
Module-ID: summary_all_tokens

Purpose: Résumé global couverture données pour tous les tokens - période backtest commune.

Role in pipeline: data summary

Key components: summarize_all_tokens(), affichage table glob

Inputs: Dossier parquet multi-tokens

Outputs: Résumé par timeframe (min/max dates, tokens dispos)

Dependencies: pandas, pathlib, collections

Conventions: Timeframes: 1m, 5m, 15m, 1h, 4h, 1d

Read-if: Vue d'ensemble couverture globale todos tokens.

Skip-if: Analyse par token suffisante.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd


def summarize_all_tokens(data_folder: str):
    """Résumé global de tous les tokens."""

    folder = Path(data_folder)
    parquet_files = list(folder.glob("*.parquet"))

    print(f"\n{'='*100}")
    print("RÉSUMÉ GLOBAL - TOUS LES TOKENS")
    print(f"{'='*100}\n")

    # Structure: timeframe -> {min_start, max_start, min_end, max_end}
    timeframe_stats = defaultdict(lambda: {
        'starts': [],
        'ends': [],
        'tokens': set()
    })

    print(f"Analyse de {len(parquet_files)} fichiers...\n")

    for file_path in parquet_files:
        try:
            filename = file_path.stem
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                continue

            token = parts[0]
            timeframe = parts[1]

            df = pd.read_parquet(file_path)

            if 'timestamp' in df.columns:
                sample_ts = float(df['timestamp'].iloc[0])
                if sample_ts > 1e12:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                elif sample_ts > 1e9:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                start_date = df['timestamp'].min()
                end_date = df['timestamp'].max()

                timeframe_stats[timeframe]['starts'].append(start_date)
                timeframe_stats[timeframe]['ends'].append(end_date)
                timeframe_stats[timeframe]['tokens'].add(token)

        except Exception:
            pass

    # Afficher le résumé par timeframe
    print(f"{'Timeframe':<10} {'Tokens':>7} {'Date début MIN':<20} {'Date début MAX':<20} "
          f"{'Date fin MIN':<20} {'Date fin MAX':<20} {'Écart début':>12} {'Écart fin':>10}")
    print(f"{'-'*10} {'-'*7} {'-'*20} {'-'*20} {'-'*20} {'-'*20} {'-'*12} {'-'*10}")

    timeframe_order = ['3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

    for tf in timeframe_order:
        if tf in timeframe_stats:
            stats = timeframe_stats[tf]
            starts = stats['starts']
            ends = stats['ends']
            tokens = stats['tokens']

            min_start = min(starts)
            max_start = max(starts)
            min_end = min(ends)
            max_end = max(ends)

            start_diff = (max_start - min_start).days
            end_diff = (max_end - min_end).days

            # Indicateurs visuels
            start_icon = "✅" if start_diff == 0 else "⚠️ "
            end_icon = "✅" if end_diff == 0 else "⚠️ "

            print(f"{tf:<10} {len(tokens):>7} {str(min_start.date()):<20} {str(max_start.date()):<20} "
                  f"{str(min_end.date()):<20} {str(max_end.date()):<20} "
                  f"{start_icon}{start_diff:>9}j {end_icon}{end_diff:>8}j")

    print(f"\n{'='*100}")
    print("PROBLÈMES DÉTECTÉS")
    print(f"{'='*100}\n")

    problems = []
    for tf in timeframe_order:
        if tf in timeframe_stats:
            stats = timeframe_stats[tf]
            starts = stats['starts']
            ends = stats['ends']

            min_start = min(starts)
            max_start = max(starts)
            min_end = min(ends)
            max_end = max(ends)

            start_diff = (max_start - min_start).days
            end_diff = (max_end - min_end).days

            if start_diff > 0:
                problems.append(f"⚠️  {tf}: Dates de DÉBUT incohérentes (écart de {start_diff} jours)")
                problems.append(f"   → Plus ancienne: {min_start.date()}")
                problems.append(f"   → Plus récente:  {max_start.date()}")

            if end_diff > 7:  # Tolérance de 7 jours
                problems.append(f"⚠️  {tf}: Dates de FIN incohérentes (écart de {end_diff} jours)")
                problems.append(f"   → Plus ancienne: {min_end.date()}")
                problems.append(f"   → Plus récente:  {max_end.date()}")

    if problems:
        for p in problems:
            print(p)
    else:
        print("✅ Aucun problème détecté - toutes les périodes sont cohérentes!")

    # Recommandations
    print(f"\n{'='*100}")
    print("RECOMMANDATIONS")
    print(f"{'='*100}\n")

    # Vérifier si les timeframes longs ont moins de profondeur
    if '1h' in timeframe_stats and '4h' in timeframe_stats:
        tf_1h_min = min(timeframe_stats['1h']['starts']).date()
        tf_4h_min = min(timeframe_stats['4h']['starts']).date()

        if tf_4h_min > tf_1h_min:
            gap_days = (tf_4h_min - tf_1h_min).days
            print(f"⚠️  Les timeframes longs (4h, 1d, 1w, 1M) manquent {gap_days} jours d'historique")
            print("   par rapport aux timeframes courts (3m, 5m, 15m, 30m, 1h)")
            print("\n💡 SOLUTION:")
            print("   1. Le paramètre HISTORY_DAYS a été modifié à 2200 jours")
            print("   2. Relancer le téléchargement pour étendre les données jusqu'en 2020")
            print("   3. Commande: cd D:\\my_soft\\gestionnaire_telechargement_multi-timeframe")
            print("                python unified_data_historique_with_indicators.py")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    data_folder = r"D:\ThreadX_big\data\crypto\processed\parquet"
    summarize_all_tokens(data_folder)
