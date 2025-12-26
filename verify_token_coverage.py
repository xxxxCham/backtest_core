#!/usr/bin/env python3
"""
Vérifie la couverture des données par token et timeframe.
Affiche les périodes de téléchargement pour chaque token.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys


def analyze_token_coverage(data_folder: str, token_filter: str = None):
    """
    Analyse la couverture des données par token.

    Args:
        data_folder: Dossier contenant les fichiers parquet
        token_filter: Filtre optionnel pour un token spécifique (ex: "AAVE")
    """

    folder = Path(data_folder)
    parquet_files = list(folder.glob("*.parquet"))

    print(f"\n{'='*90}")
    print(f"VÉRIFICATION DE LA COUVERTURE DES DONNÉES")
    print(f"Dossier: {data_folder}")
    if token_filter:
        print(f"Filtre token: {token_filter}")
    print(f"{'='*90}\n")

    # Structure: token -> timeframe -> {start, end, bars, file}
    tokens_data = defaultdict(lambda: defaultdict(dict))

    # Analyser chaque fichier
    print(f"Analyse de {len(parquet_files)} fichiers...\n")

    for file_path in parquet_files:
        try:
            # Extraire token et timeframe du nom
            filename = file_path.stem
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                continue

            token = parts[0]
            timeframe = parts[1]

            # Appliquer le filtre si spécifié
            if token_filter and not token.upper().startswith(token_filter.upper()):
                continue

            # Charger le fichier
            df = pd.read_parquet(file_path)

            if 'timestamp' in df.columns:
                # Convertir les timestamps
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

                tokens_data[token][timeframe] = {
                    'start': start_date,
                    'end': end_date,
                    'bars': bars,
                    'file': file_path.name,
                    'duration_days': (end_date - start_date).days
                }

        except Exception as e:
            print(f"⚠️  Erreur sur {file_path.name}: {e}")

    # Afficher les résultats par token
    timeframe_order = ['3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

    for token in sorted(tokens_data.keys()):
        timeframes = tokens_data[token]

        print(f"\n{'='*90}")
        print(f"TOKEN: {token}")
        print(f"{'='*90}")

        # Tableau des timeframes
        print(f"\n{'Timeframe':<10} {'Début':<20} {'Fin':<20} {'Barres':>8} {'Jours':>6}")
        print(f"{'-'*10} {'-'*20} {'-'*20} {'-'*8} {'-'*6}")

        dates_start = []
        dates_end = []

        for tf in timeframe_order:
            if tf in timeframes:
                data = timeframes[tf]
                start = data['start']
                end = data['end']
                bars = data['bars']
                days = data['duration_days']

                dates_start.append(start)
                dates_end.append(end)

                print(f"{tf:<10} {start.date()} {start.time()} {end.date()} {end.time()} "
                      f"{bars:>8,} {days:>6}")

        # Vérifier la cohérence des périodes
        if len(dates_start) > 1:
            min_start = min(dates_start)
            max_start = max(dates_start)
            min_end = min(dates_end)
            max_end = max(dates_end)

            print(f"\n{'='*90}")
            print(f"ANALYSE DE COHÉRENCE")
            print(f"{'='*90}")

            # Vérifier si toutes les dates de début sont identiques
            start_diff = (max_start - min_start).days
            end_diff = (max_end - min_end).days

            if start_diff == 0:
                print(f"✅ Dates de DÉBUT cohérentes: {min_start.date()}")
            else:
                print(f"⚠️  Dates de DÉBUT différentes:")
                print(f"   Plus ancienne : {min_start.date()} {min_start.time()}")
                print(f"   Plus récente  : {max_start.date()} {max_start.time()}")
                print(f"   Écart         : {start_diff} jours")

            print()

            if end_diff == 0:
                print(f"✅ Dates de FIN cohérentes: {max_end.date()}")
            else:
                print(f"⚠️  Dates de FIN différentes:")
                print(f"   Plus ancienne : {min_end.date()} {min_end.time()}")
                print(f"   Plus récente  : {max_end.date()} {max_end.time()}")
                print(f"   Écart         : {end_diff} jours")

            # Période commune (intersection)
            common_start = max_start
            common_end = min_end

            if common_end > common_start:
                common_days = (common_end - common_start).days
                print(f"\n📊 PÉRIODE COMMUNE (intersection de tous les timeframes):")
                print(f"   {common_start.date()} → {common_end.date()} ({common_days} jours)")
            else:
                print(f"\n❌ Pas de période commune entre les timeframes!")

    # Résumé global
    print(f"\n{'='*90}")
    print(f"RÉSUMÉ GLOBAL")
    print(f"{'='*90}\n")
    print(f"Tokens analysés: {len(tokens_data)}")

    # Compter les timeframes par token
    tf_counts = {}
    for token, timeframes in tokens_data.items():
        count = len(timeframes)
        tf_counts[count] = tf_counts.get(count, 0) + 1

    print(f"\nRépartition par nombre de timeframes:")
    for count in sorted(tf_counts.keys(), reverse=True):
        print(f"   {count} timeframes: {tf_counts[count]} tokens")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    data_folder = r"D:\ThreadX_big\data\crypto\processed\parquet"

    # Vérifier si un token spécifique est demandé en argument
    token_filter = None
    if len(sys.argv) > 1:
        token_filter = sys.argv[1]
        print(f"\nFiltre appliqué: {token_filter}")

    analyze_token_coverage(data_folder, token_filter)
