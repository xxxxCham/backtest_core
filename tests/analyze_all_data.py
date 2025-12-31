#!/usr/bin/env python3
"""
Module-ID: analyze_all_data

Purpose: Analyser tous fichiers OHLCV parquet d'un dossier - déterminer période backtest commune globale.

Role in pipeline: data quality assurance

Key components: analyze_all_parquet_files(), résumé global par timeframe

Inputs: Dossier avec fichiers SYMBOL_TIMEFRAME.parquet

Outputs: Coverage par token/timeframe, période globale min/max, rapport erreurs

Dependencies: pandas, pathlib, collections

Conventions: Fichiers nommés SYMBOL_TIMEFRAME.parquet; résumé par timeframe

Read-if: Planifier backtest multi-token, vérifier couverture commune.

Skip-if: Token unique ou données pré-analysées.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd


def analyze_all_parquet_files(folder_path: str):
    """Analyse tous les fichiers parquet d'un dossier."""

    folder = Path(folder_path)
    parquet_files = list(folder.glob("*.parquet"))

    print(f"\n{'='*80}")
    print(f"ANALYSE DE {len(parquet_files)} FICHIERS PARQUET")
    print(f"Dossier: {folder_path}")
    print(f"{'='*80}\n")

    # Structures pour stocker les résultats
    tokens_data = defaultdict(lambda: defaultdict(dict))  # token -> timeframe -> {start, end, bars}
    timeframe_ranges = defaultdict(list)  # timeframe -> [(start, end), ...]
    global_min = None
    global_max = None

    errors = []

    # Analyser chaque fichier
    for i, file_path in enumerate(parquet_files, 1):
        try:
            # Extraire token et timeframe du nom de fichier
            filename = file_path.stem
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                continue

            token = parts[0]
            timeframe = parts[1]

            # Charger et analyser
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
                bars = len(df)

                # Stocker les données
                tokens_data[token][timeframe] = {
                    'start': start_date,
                    'end': end_date,
                    'bars': bars,
                    'file': file_path.name
                }

                timeframe_ranges[timeframe].append((start_date, end_date))

                # Mettre à jour min/max global
                if global_min is None or start_date < global_min:
                    global_min = start_date
                if global_max is None or end_date > global_max:
                    global_max = end_date

                if i % 20 == 0:
                    print(f"   Analysé {i}/{len(parquet_files)} fichiers...")

        except Exception as e:
            errors.append(f"{file_path.name}: {str(e)}")

    print(f"\n✅ Analyse terminée: {len(parquet_files)} fichiers\n")

    # Afficher résumé par timeframe
    print(f"{'='*80}")
    print("RÉSUMÉ PAR TIMEFRAME")
    print(f"{'='*80}\n")

    timeframe_order = ['3m', '5m', '15m', '30m', '1h', '4h', '1d']

    for tf in timeframe_order:
        if tf in timeframe_ranges:
            ranges = timeframe_ranges[tf]
            tf_min = min(r[0] for r in ranges)
            tf_max = max(r[1] for r in ranges)
            num_tokens = len(ranges)

            print(f"⏱️  {tf:4s} ({num_tokens:3d} tokens)")
            print(f"   📅 {tf_min.date()} → {tf_max.date()}")
            print(f"   📊 {(tf_max - tf_min).days} jours\n")

    # Période commune (intersection)
    print(f"{'='*80}")
    print("PÉRIODE GLOBALE DE BACKTEST")
    print(f"{'='*80}\n")

    if global_min and global_max:
        duration = global_max - global_min
        print("📅 PÉRIODE TOTALE DISPONIBLE")
        print(f"   Du   : {global_min.date()} {global_min.time()}")
        print(f"   Au   : {global_max.date()} {global_max.time()}")
        print(f"   Durée: {duration.days} jours ({duration.days/365.25:.1f} années)")
        print()

        # Calculer la période commune pour chaque timeframe
        print("📊 PÉRIODE COMMUNE PAR TIMEFRAME (pour backtests multi-tokens)")
        for tf in timeframe_order:
            if tf in timeframe_ranges:
                ranges = timeframe_ranges[tf]
                common_start = max(r[0] for r in ranges)
                common_end = min(r[1] for r in ranges)
                common_days = (common_end - common_start).days

                print(f"   {tf:4s}: {common_start.date()} → {common_end.date()} ({common_days} jours)")

    # Top tokens par nombre de timeframes
    print(f"\n{'='*80}")
    print("TOP 20 TOKENS (par nombre de timeframes)")
    print(f"{'='*80}\n")

    token_counts = [(token, len(timeframes)) for token, timeframes in tokens_data.items()]
    token_counts.sort(key=lambda x: x[1], reverse=True)

    for i, (token, count) in enumerate(token_counts[:20], 1):
        timeframes = sorted(tokens_data[token].keys())
        print(f"{i:2d}. {token:20s} ({count} timeframes): {', '.join(timeframes)}")

    print(f"\n{'='*80}")
    print("STATISTIQUES GÉNÉRALES")
    print(f"{'='*80}\n")
    print(f"   Tokens uniques      : {len(tokens_data)}")
    print(f"   Timeframes          : {', '.join(sorted(timeframe_ranges.keys()))}")
    print(f"   Fichiers totaux     : {len(parquet_files)}")
    print(f"   Erreurs             : {len(errors)}")

    if errors:
        print("\n⚠️  ERREURS DÉTECTÉES:")
        for err in errors[:10]:
            print(f"   • {err}")

    print(f"\n{'='*80}\n")

    return tokens_data, timeframe_ranges


if __name__ == "__main__":
    folder_path = r"D:\ThreadX_big\data\crypto\processed\parquet"
    analyze_all_parquet_files(folder_path)
