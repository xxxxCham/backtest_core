#!/usr/bin/env python3
"""
Module-ID: analyze_data

Purpose: Analyser détail fichier OHLCV unique - dates, continuité, gaps, complétude de barres.

Role in pipeline: data quality assurance

Key components: analyze_parquet_file(), affichage stats

Inputs: Chemin fichier parquet OHLCV

Outputs: Statistiques: nombre barres, plage dates, gaps, continuité, colonnes

Dependencies: pandas, pathlib, collections

Conventions: Index pandas.DatetimeIndex; colonnes [open, high, low, close, volume]

Read-if: Vérifier qualité/couverture fichier avant backtest.

Skip-if: Données pré-validées.
"""

import sys
from pathlib import Path

import pandas as pd


def analyze_parquet_file(file_path: str):
    """Analyse un fichier Parquet OHLCV."""

    print(f"\n{'='*70}")
    print(f"ANALYSE: {Path(file_path).name}")
    print(f"{'='*70}\n")

    try:
        # Charger le fichier
        df = pd.read_parquet(file_path)

        # Informations de base
        print("📊 STATISTIQUES GÉNÉRALES")
        print(f"   Nombre total de barres : {len(df):,}")
        print(f"   Taille mémoire         : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print(f"   Colonnes               : {list(df.columns)}")

        # Vérifier l'index et convertir les timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            index_col = df.index
        elif 'timestamp' in df.columns:
            # Essayer de détecter le format du timestamp
            sample_ts = float(df['timestamp'].iloc[0])  # Convertir en float pour éviter problème numpy
            if sample_ts > 1e12:
                # Timestamp en millisecondes
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif sample_ts > 1e9:
                # Timestamp en secondes
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # Format datetime normal
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            index_col = df['timestamp']
        else:
            print("❌ Pas d'index datetime trouvé!")
            return

        # Plage de dates
        start_date = index_col.min()
        end_date = index_col.max()
        duration = end_date - start_date

        print("\n📅 PLAGE DE DATES")
        print(f"   Début  : {start_date}")
        print(f"   Fin    : {end_date}")
        print(f"   Durée  : {duration.days} jours ({duration.days / 365.25:.1f} années)")

        # Vérifier la continuité (gaps)
        print("\n🔍 VÉRIFICATION CONTINUITÉ (gaps)")

        # Calculer les différences entre timestamps consécutifs
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
        else:
            time_diffs = index_col.diff()

        # Mode attendu (1h = 3600s)
        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)  # Tolérance de 5 min

        # Identifier les gaps
        gaps = time_diffs[(time_diffs > expected_diff + tolerance) | (time_diffs < expected_diff - tolerance)]
        gaps = gaps.dropna()

        if len(gaps) == 0:
            print("   ✅ Aucun gap détecté - données continues!")
        else:
            print(f"   ⚠️  {len(gaps)} gaps détectés:")
            for i, (idx, gap) in enumerate(gaps.items()):
                if i < 10:  # Montrer max 10 gaps
                    print(f"      • {idx}: écart de {gap}")
                elif i == 10:
                    print(f"      ... et {len(gaps) - 10} autres gaps")
                    break

        # Statistiques OHLCV
        print("\n📈 STATISTIQUES OHLCV")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                values = df[col]
                print(
                    f"   {col.upper():8s}: min={values.min():.2f}  max={values.max():.2f}  "
                    f"mean={values.mean():.2f}  null={values.isna().sum()}"
                )

        # Vérifier les valeurs manquantes
        print("\n🔎 VALEURS MANQUANTES")
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            print("   ✅ Aucune valeur manquante")
        else:
            print("   ⚠️  Valeurs manquantes détectées:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"      • {col}: {count} ({count/len(df)*100:.2f}%)")

        # Vérifier cohérence OHLC
        print("\n✓ COHÉRENCE OHLC")
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = (
                (df['high'] < df['open'])
                | (df['high'] < df['close'])
                | (df['high'] < df['low'])
            )
            invalid_low = (
                (df['low'] > df['open'])
                | (df['low'] > df['close'])
                | (df['low'] > df['high'])
            )

            if invalid_high.sum() == 0 and invalid_low.sum() == 0:
                print("   ✅ Toutes les barres OHLC sont cohérentes")
            else:
                print(f"   ⚠️  Barres incohérentes: high={invalid_high.sum()}, low={invalid_low.sum()}")

        # Résumé final
        print(f"\n{'='*70}")
        print("RÉSUMÉ")
        print(f"{'='*70}")
        print(f"✅ Données chargées    : {len(df):,} barres")
        print(f"✅ Période couverte    : {start_date.date()} → {end_date.date()}")
        print(f"{'✅' if len(gaps) == 0 else '⚠️ '} Continuité         : {'OK' if len(gaps) == 0 else f'{len(gaps)} gaps'}")
        print(f"{'✅' if null_counts.sum() == 0 else '⚠️ '} Valeurs manquantes : {'Aucune' if null_counts.sum() == 0 else 'Présentes'}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = r"D:\backtest_core\docs\AAVEUSDC_1h.parquet"

    analyze_parquet_file(file_path)
