"""
Module-ID: data.config

Purpose: Configuration et logique métier pour la gestion des données OHLCV.
         Extraction de la logique depuis ui/sidebar.py (DDD refactoring).

Role in pipeline: domain / configuration

Key components:
- scan_data_availability: Scan multi-token des données disponibles
- get_intelligent_timeframe_defaults: Sélection optimisée des TF
- validate_period_for_tokens: Validation multi-token d'une période
- generate_random_token_suggestions: Suggestions aléatoires

Dependencies: data.loader, pandas

Conventions: Fonctions pures (pas de Streamlit), retournent des dicts/dataclasses

Read-if: Configuration des données pour UI ou CLI
Skip-if: Logique de trading
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from utils.log import get_logger

logger = get_logger(__name__)

import pandas as pd
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

# Nombre de suggestions aléatoires à afficher
RANDOM_SLOTS_COUNT = 6

# Priorité des timeframes pour sélection intelligente
TIMEFRAME_PRIORITY_MAP = {
    "1m": 1, "3m": 2, "5m": 3, "15m": 4, "30m": 5,
    "1h": 6, "2h": 7, "4h": 8, "6h": 9, "8h": 10, "12h": 11,
    "1d": 12, "3d": 13, "1w": 14, "1M": 15
}

# Facteur de fréquence de trading par timeframe (opportunités relatives)
TIMEFRAME_FREQUENCY_FACTOR = {
    "1m": 1440,   # 1440 barres/jour
    "3m": 480,    # 480 barres/jour
    "5m": 288,    # 288 barres/jour
    "15m": 96,    # 96 barres/jour
    "30m": 48,    # 48 barres/jour
    "1h": 24,     # 24 barres/jour
    "2h": 12,     # 12 barres/jour
    "4h": 6,      # 6 barres/jour
    "6h": 4,      # 4 barres/jour
    "8h": 3,      # 3 barres/jour
    "12h": 2,     # 2 barres/jour
    "1d": 1,      # 1 barre/jour
    "3d": 0.33,   # 0.33 barre/jour
    "1w": 0.14,   # 0.14 barre/jour
    "1M": 0.03    # 0.03 barre/jour
}

# Catégorisation des timeframes pour analyse indépendante
TIMEFRAME_CATEGORIES = {
    "scalping": ["1m", "3m", "5m"],           # Trading haute fréquence
    "intraday": ["15m", "30m", "1h", "2h"],   # Trading intraday
    "swing": ["4h", "6h", "8h", "12h"],        # Swing trading
    "position": ["1d", "3d", "1w", "1M"]       # Position trading
}

# Facteurs de tolérance aux gaps par catégorie
CATEGORY_GAP_TOLERANCE = {
    "scalping": 0.02,    # 2% de gaps max (très sensible)
    "intraday": 0.05,    # 5% de gaps max (sensible)
    "swing": 0.10,       # 10% de gaps max (modéré)
    "position": 0.20     # 20% de gaps max (tolérant)
}


class OptimalPeriod(NamedTuple):
    """Période optimale avec métadonnées de qualité."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    completeness_score: float  # 0-100%
    tokens_complete: int
    tokens_total: int
    avg_data_density: float
    description: str
    category: str = "mixed"  # Catégorie de timeframe
    timeframes: List[str] = None  # Timeframes concernés


class CategoryAnalysis(NamedTuple):
    """Analyse par catégorie de timeframes."""
    category: str
    timeframes: List[str]
    symbols: List[str]
    optimal_periods: List[OptimalPeriod]
    best_period: Optional[OptimalPeriod]
    data_quality_score: float
    trading_opportunities_score: float


class DataGap(NamedTuple):
    """Représente un gap dans les données."""
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float
    token: str
    timeframe: str


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

RANDOM_SLOTS_COUNT = 6

TIMEFRAME_PRIORITY_MAP = {
    "1m": 1, "3m": 2, "5m": 3, "15m": 4, "30m": 5,
    "1h": 6, "2h": 7, "4h": 8, "6h": 9, "12h": 10,
    "1d": 11, "1w": 12, "1M": 13
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DataAvailabilityResult:
    """Résultat du scan de disponibilité des données."""
    availability: Dict[Tuple[str, str], Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)
    missing_data: List[str] = field(default_factory=list)
    common_start: Optional[pd.Timestamp] = None
    common_end: Optional[pd.Timestamp] = None
    has_common_range: bool = False
    rows: List[Dict[str, Any]] = field(default_factory=list)
    optimal_periods: List[OptimalPeriod] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les lignes en DataFrame pour affichage."""
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame(self.rows)

    def get_best_period(self) -> Optional[OptimalPeriod]:
        """Retourne la meilleure période optimale si disponible."""
        return self.optimal_periods[0] if self.optimal_periods else None


@dataclass
class PeriodValidationResult:
    """Résultat de la validation d'une période."""
    tokens_ok: List[str] = field(default_factory=list)
    tokens_partial: List[str] = field(default_factory=list)
    tokens_missing: List[str] = field(default_factory=list)
    all_ok: bool = False


# ============================================================================
# DATA DISCOVERY FUNCTIONS
# ============================================================================

def _load_ohlcv_silent(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Charge OHLCV sans logs (utilisé pour analyse des gaps/metadata)."""
    from data.loader import _find_data_file, _read_file, _normalize_ohlcv

    file_path = _find_data_file(symbol, timeframe)
    if file_path is None:
        return None

    df = _read_file(file_path)
    df = _normalize_ohlcv(df)
    return df


@lru_cache(maxsize=256)
def _get_range_and_gaps(
    symbol: str,
    timeframe: str,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Tuple["DataGap", ...]]:
    """
    Retourne plage + gaps avec un seul chargement (cache en mémoire).

    Objectif: éviter les multiples load_ohlcv() et leurs logs.
    """
    try:
        df = _load_ohlcv_silent(symbol, timeframe)
        if df is None or df.empty:
            return None, None, tuple()

        # Calculer l'intervalle attendu entre les barres
        if "m" in timeframe:
            expected_interval = pd.Timedelta(minutes=int(timeframe.replace("m", "")))
        elif "h" in timeframe:
            expected_interval = pd.Timedelta(hours=int(timeframe.replace("h", "")))
        elif "d" in timeframe:
            expected_interval = pd.Timedelta(days=int(timeframe.replace("d", "")))
        else:
            expected_interval = pd.Timedelta(hours=1)  # Fallback

        gap_threshold = expected_interval * 2
        time_diffs = df.index[1:] - df.index[:-1]
        gap_indices = time_diffs > gap_threshold

        gaps: List[DataGap] = []
        if gap_indices.any():
            gap_positions = np.where(gap_indices)[0]

            for gap_idx in gap_positions:
                gap_start = df.index[gap_idx]
                gap_end = df.index[gap_idx + 1]
                duration_days = (gap_end - gap_start).total_seconds() / (24 * 3600)

                if duration_days > 0.1:  # Ignorer gaps < 2.4h
                    gaps.append(DataGap(
                        start=gap_start,
                        end=gap_end,
                        duration_days=duration_days,
                        token=symbol,
                        timeframe=timeframe
                    ))

        gaps_sorted = tuple(sorted(gaps, key=lambda g: g.duration_days, reverse=True))
        return df.index[0], df.index[-1], gaps_sorted
    except (OSError, ValueError, KeyError, IndexError, AttributeError):
        return None, None, tuple()


def get_data_date_range(symbol: str, timeframe: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Récupère la plage de dates disponibles pour un symbole/timeframe.

    Args:
        symbol: Le symbole (ex: "BTCUSDC")
        timeframe: Le timeframe (ex: "1h")

    Returns:
        Tuple (start, end) ou None si données indisponibles
    """
    start_ts, end_ts, _ = _get_range_and_gaps(symbol, timeframe)
    if start_ts is None or end_ts is None:
        return None
    return (start_ts, end_ts)


def check_data_completeness(
    symbol: str,
    timeframe: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Tuple[bool, str, int]:
    """
    Vérifie la complétude des données dans une plage spécifique.

    Cette fonction charge effectivement les données avec filtrage pour détecter
    les trous ou données manquantes que la simple plage début/fin ne révèle pas.

    Args:
        symbol: Le symbole (ex: "BTCUSDC")
        timeframe: Le timeframe (ex: "1h")
        start_date: Date début du filtrage (optionnel)
        end_date: Date fin du filtrage (optionnel)

    Returns:
        Tuple (is_complete, message, actual_rows)
        - is_complete: True si données complètes dans la plage
        - message: Description de l'état des données
        - actual_rows: Nombre de barres effectivement disponibles
    """
    try:
        from data.loader import load_ohlcv

        # Charger avec filtrage de dates si spécifié
        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None

        df = load_ohlcv(symbol, timeframe, start=start_str, end=end_str)

        if df is None or df.empty:
            return False, f"Aucune donnée disponible pour {symbol}/{timeframe}", 0

        actual_rows = len(df)
        period_start = df.index[0]
        period_end = df.index[-1]

        # Calculer la période effective chargée
        period_days = (period_end - period_start).days

        # Estimation approximative du nombre de barres attendues
        # (pas parfait car dépend des week-ends/jours fériés, mais donne une idée)
        if "m" in timeframe:
            minutes_per_bar = int(timeframe.replace("m", ""))
            expected_bars_per_day = 1440 / minutes_per_bar  # 24h * 60min
        elif "h" in timeframe:
            hours_per_bar = int(timeframe.replace("h", ""))
            expected_bars_per_day = 24 / hours_per_bar
        elif "d" in timeframe:
            expected_bars_per_day = 1
        else:
            expected_bars_per_day = 1  # Fallback

        expected_bars = int(period_days * expected_bars_per_day * 0.7)  # 70% pour week-ends

        # Vérifier si les données semblent complètes (seuil 50% des barres attendues)
        completeness_ratio = actual_rows / max(expected_bars, 1)

        if completeness_ratio >= 0.5:
            message = f"✅ Données complètes: {actual_rows} barres sur {period_days} jours"
            return True, message, actual_rows
        else:
            message = f"⚠️ Données incomplètes: {actual_rows} barres (attendu ~{expected_bars})"
            return False, message, actual_rows

    except Exception as e:
        return False, f"❌ Erreur chargement: {str(e)}", 0


def analyze_data_gaps(symbol: str, timeframe: str) -> List[DataGap]:
    """
    Analyse les gaps/trous dans les données d'un token/timeframe.

    Args:
        symbol: Le symbole (ex: "BTCUSDC")
        timeframe: Le timeframe (ex: "1h")

    Returns:
        Liste des gaps détectés, triés par durée décroissante
    """
    _, _, gaps = _get_range_and_gaps(symbol, timeframe)
    return list(gaps)


def find_optimal_periods(
    symbols: List[str],
    timeframes: List[str],
    min_period_days: int = 30,
    max_periods: int = 3
) -> List[OptimalPeriod]:
    """
    Trouve les périodes optimales avec tolérance aux gaps selon les timeframes.

    Analyse intelligente qui :
    1. Scan les gaps de tous les tokens/timeframes
    2. Utilise les tolérances CATEGORY_GAP_TOLERANCE selon les timeframes
    3. Identifie les segments viables entre les gros gaps
    4. Score les périodes selon la complétude des données
    5. Retourne les meilleures périodes classées

    Args:
        symbols: Liste des symboles à analyser
        timeframes: Liste des timeframes à analyser
        min_period_days: Période minimale acceptable (défaut: 30 jours)
        max_periods: Nombre maximum de périodes à retourner

    Returns:
        Liste des périodes optimales, triées par score décroissant
    """
    try:
        # 0. Déterminer la tolérance aux gaps selon les timeframes
        categories = categorize_timeframes(timeframes)

        # Prendre la tolérance la plus stricte parmi les catégories présentes
        relevant_tolerances = []
        for category, category_tfs in categories.items():
            if category_tfs:  # Catégorie non vide
                tolerance = CATEGORY_GAP_TOLERANCE.get(category, 0.10)
                relevant_tolerances.append(tolerance)

        gap_tolerance = min(relevant_tolerances) if relevant_tolerances else 0.10
        logger.info(f"🎯 Tolérance gaps utilisée: {gap_tolerance:.1%} (timeframes: {timeframes})")

        # 1. Collecter toutes les données disponibles et leurs gaps
        all_data_ranges = {}
        all_gaps = {}

        for symbol in symbols:
            for tf in timeframes:
                combo = (symbol, tf)

                # Récupérer la plage globale
                date_range = get_data_date_range(symbol, tf)
                if date_range:
                    all_data_ranges[combo] = date_range

                    # Analyser les gaps
                    gaps = analyze_data_gaps(symbol, tf)
                    all_gaps[combo] = gaps

        if not all_data_ranges:
            return []

        # 2. Trouver la plage commune globale
        all_starts = [range_[0] for range_ in all_data_ranges.values()]
        all_ends = [range_[1] for range_ in all_data_ranges.values()]

        global_start = max(all_starts)
        global_end = min(all_ends)

        if global_start >= global_end:
            return []

        # 3. NOUVEAU: Identifier les gros gaps selon la tolérance
        total_days = (global_end - global_start).days
        if total_days < min_period_days:
            return []

        # Collecter tous les gaps et filtrer selon la tolérance
        major_gaps = []  # Gaps qui cassent vraiment la continuité
        for combo, gaps in all_gaps.items():
            for gap in gaps:
                gap_days = (gap.end - gap.start).days

                # Un gap est "majeur" s'il dépasse la tolérance de la catégorie
                # Ex: pour intraday (5% tolérance), sur 100 jours, gap > 5 jours = majeur
                period_for_tolerance = max(min_period_days, 30)  # Au moins 30 jours de référence
                gap_threshold_days = period_for_tolerance * gap_tolerance

                if gap_days > gap_threshold_days:
                    major_gaps.append((gap.start, gap.end, gap_days))
                    logger.debug(f"   Gap majeur: {gap.start.strftime('%Y-%m-%d')} → {gap.end.strftime('%Y-%m-%d')} ({gap_days:.1f}j)")

        # Trier par date de début
        major_gaps.sort(key=lambda x: x[0])
        logger.info(f"   Total gaps majeurs: {len(major_gaps)}")

        # 4. Créer des segments entre les gros gaps
        segments = []
        current_start = global_start

        for gap_start, gap_end, gap_days in major_gaps:
            # Si le gap commence après current_start, on a un segment potentiel
            if gap_start > current_start:
                segment_end = gap_start
                segment_days = (segment_end - current_start).days

                if segment_days >= min_period_days:
                    segments.append((current_start, segment_end, segment_days))
                    logger.debug(f"   Segment viable: {current_start.strftime('%Y-%m-%d')} → {segment_end.strftime('%Y-%m-%d')} ({segment_days}j)")

            # Nouveau début après ce gap
            current_start = max(current_start, gap_end)

        # Dernier segment après le dernier gap
        if current_start < global_end:
            segment_days = (global_end - current_start).days
            if segment_days >= min_period_days:
                segments.append((current_start, global_end, segment_days))
                logger.debug(f"   Segment final: {current_start.strftime('%Y-%m-%d')} → {global_end.strftime('%Y-%m-%d')} ({segment_days}j)")

        # 5. Si aucun segment sans gros gaps, prendre toute la plage avec tolérance
        if not segments:
            logger.info(f"   Aucun segment sans gaps majeurs, utilisation plage complète avec tolérance")
            segments = [(global_start, global_end, total_days)]

        # 6. Évaluer la qualité de chaque segment
        scored_periods = []

        for segment_start, segment_end, segment_days in segments:
            # Compter les données disponibles dans ce segment
            total_data_points = 0
            total_possible_points = 0
            tolerated_gaps = 0
            tokens_complete = 0

            for symbol in symbols:
                for tf in timeframes:
                    combo = (symbol, tf)
                    if combo not in all_data_ranges:
                        continue

                    # Vérifier si cette combo a des gaps tolérables dans le segment
                    has_major_gaps = False
                    gaps = all_gaps.get(combo, [])

                    for gap in gaps:
                        # Gap chevauche-t-il le segment ?
                        if (gap.start < segment_end and gap.end > segment_start):
                            gap_days = (gap.end - gap.start).days
                            gap_threshold_days = segment_days * gap_tolerance

                            if gap_days > gap_threshold_days:
                                has_major_gaps = True
                                break
                            else:
                                tolerated_gaps += 1

                    if not has_major_gaps:
                        tokens_complete += 1

                    # Calculer densité théorique
                    freq_factor = TIMEFRAME_FREQUENCY_FACTOR.get(tf, 1440)  # défaut = 1d
                    possible_points = segment_days * freq_factor
                    total_possible_points += possible_points

                    # Estimer points réels (approximation)
                    total_gap_time = sum(
                        (min(gap.end, segment_end) - max(gap.start, segment_start)).total_seconds() / 3600
                        for gap in gaps
                        if gap.start < segment_end and gap.end > segment_start
                    )

                    gap_ratio = min(1.0, total_gap_time / (segment_days * 24))
                    estimated_points = possible_points * (1 - gap_ratio)
                    total_data_points += max(0, estimated_points)

            # Calculer les métriques de qualité
            tokens_total = len(symbols) * len(timeframes)

            if tokens_total > 0 and total_possible_points > 0:
                completeness_score = (tokens_complete / tokens_total) * 100
                data_density = min(100.0, total_data_points / total_possible_points * 100)

                # Bonus pour périodes plus longues
                period_bonus = min(1.2, 1.0 + (segment_days - min_period_days) / (min_period_days * 2))

                # Score global : pondérer complétude et densité
                quality_score = (completeness_score * 0.6) + (data_density * 0.4)
                quality_score *= period_bonus

                description = f"{segment_days}j, {tokens_complete}/{tokens_total} combos complètes"
                if data_density < 90:
                    description += f", densité {data_density:.1f}%"
                if tolerated_gaps > 0:
                    description += f", {tolerated_gaps} gaps tolérés"

                scored_periods.append(OptimalPeriod(
                    start_date=segment_start,
                    end_date=segment_end,
                    completeness_score=completeness_score,
                    tokens_complete=tokens_complete,
                    tokens_total=tokens_total,
                    avg_data_density=data_density / 100.0,
                    description=description
                ))

        # 7. Trier par score et déduplicquer les périodes similaires
        scored_periods.sort(key=lambda p: p.completeness_score * p.avg_data_density, reverse=True)

        # Déduplication : éviter les périodes qui se chevauchent trop
        unique_periods = []
        for period in scored_periods:
            is_duplicate = False
            for existing in unique_periods:
                overlap_days = min(period.end_date, existing.end_date) - max(period.start_date, existing.start_date)
                overlap_ratio = overlap_days.days / (period.end_date - period.start_date).days

                if overlap_ratio > 0.7:  # 70% de chevauchement = doublon
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_periods.append(period)

                if len(unique_periods) >= max_periods:
                    break

        return unique_periods

    except (ValueError, KeyError, TypeError, AttributeError):
        return []


def _build_segments_from_gaps(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    gaps: List[DataGap],
    gap_threshold_days: float,
    min_period_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Découpe une plage de données en segments en excluant les gaps majeurs."""
    if data_start >= data_end:
        return []

    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = data_start
    for gap in sorted(gaps, key=lambda g: g.start):
        gap_days = (gap.end - gap.start).days
        if gap_days > gap_threshold_days:
            segment_end = gap.start
            if (segment_end - current_start).days >= min_period_days:
                segments.append((current_start, segment_end))
            current_start = max(current_start, gap.end)

    if (data_end - current_start).days >= min_period_days:
        segments.append((current_start, data_end))

    return segments


def _intersect_segments(
    left: List[Tuple[pd.Timestamp, pd.Timestamp]],
    right: List[Tuple[pd.Timestamp, pd.Timestamp]],
    min_period_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Intersecte deux listes de segments et filtre par durée minimale."""
    intersections: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start < end and (end - start).days >= min_period_days:
                intersections.append((start, end))
    return intersections


def _estimate_segment_density(
    symbols: List[str],
    timeframe: str,
    segment_start: pd.Timestamp,
    segment_end: pd.Timestamp,
    gaps_by_symbol: Dict[str, List[DataGap]],
) -> float:
    """Estime la densité des données dans un segment via les gaps."""
    segment_hours = max(1.0, (segment_end - segment_start).total_seconds() / 3600.0)
    total_hours = segment_hours * max(1, len(symbols))
    gap_hours = 0.0

    for symbol in symbols:
        for gap in gaps_by_symbol.get(symbol, []):
            overlap_start = max(gap.start, segment_start)
            overlap_end = min(gap.end, segment_end)
            if overlap_start < overlap_end:
                gap_hours += (overlap_end - overlap_start).total_seconds() / 3600.0

    gap_ratio = min(1.0, gap_hours / total_hours) if total_hours > 0 else 1.0
    return max(0.0, 1.0 - gap_ratio)


def find_longest_common_period_for_timeframe(
    symbols: List[str],
    timeframe: str,
    min_period_days: int,
) -> List[OptimalPeriod]:
    """
    Trouve la plus longue période commune consécutive pour un timeframe.

    Procédure:
    - Pour chaque token, découpe sa plage en segments sans gaps majeurs.
    - Intersecte tous les segments pour obtenir la plage commune.
    - Sélectionne le segment commun le plus long.
    """
    if not symbols:
        return []

    category = get_timeframe_category(timeframe)
    gap_tolerance = CATEGORY_GAP_TOLERANCE.get(category, 0.10)

    segments_by_symbol: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    gaps_by_symbol: Dict[str, List[DataGap]] = {}

    for symbol in symbols:
        date_range = get_data_date_range(symbol, timeframe)
        if not date_range:
            return []

        data_start, data_end = date_range
        total_days = max(1, (data_end - data_start).days)
        gap_threshold_days = max(1.0, total_days * gap_tolerance)

        gaps = analyze_data_gaps(symbol, timeframe)
        gaps_by_symbol[symbol] = gaps

        segments = _build_segments_from_gaps(
            data_start=data_start,
            data_end=data_end,
            gaps=gaps,
            gap_threshold_days=gap_threshold_days,
            min_period_days=min_period_days,
        )
        if not segments:
            return []
        segments_by_symbol[symbol] = segments

    common_segments = segments_by_symbol[symbols[0]]
    for symbol in symbols[1:]:
        common_segments = _intersect_segments(
            common_segments,
            segments_by_symbol[symbol],
            min_period_days,
        )
        if not common_segments:
            return []

    best_start, best_end = max(
        common_segments,
        key=lambda s: (s[1] - s[0]).days,
    )
    duration_days = (best_end - best_start).days
    avg_density = _estimate_segment_density(
        symbols,
        timeframe,
        best_start,
        best_end,
        gaps_by_symbol,
    )

    description = f"{duration_days}j, commun a {len(symbols)}/{len(symbols)} tokens"
    return [
        OptimalPeriod(
            start_date=best_start,
            end_date=best_end,
            completeness_score=100.0,
            tokens_complete=len(symbols),
            tokens_total=len(symbols),
            avg_data_density=avg_density,
            description=description,
            category=category,
            timeframes=[timeframe],
        )
    ]


def categorize_timeframes(timeframes: List[str]) -> Dict[str, List[str]]:
    """
    Catégorise les timeframes selon leur durée.

    Args:
        timeframes: Liste des timeframes à catégoriser

    Returns:
        Dict avec clés 'scalping', 'intraday', 'swing', 'position'
    """
    categories = {
        'scalping': [],
        'intraday': [],
        'swing': [],
        'position': []
    }

    for tf in timeframes:
        for category, tf_list in TIMEFRAME_CATEGORIES.items():
            if tf in tf_list:
                categories[category].append(tf)
                break

    return categories


def get_min_period_days_for_timeframes(timeframes: List[str]) -> int:
    """
    Détermine la durée minimale recommandée selon les timeframes sélectionnés.

    Args:
        timeframes: Liste des timeframes à évaluer

    Returns:
        Nombre de jours minimum recommandé
    """
    if not timeframes:
        return 30

    frequency_factor = sum(
        TIMEFRAME_FREQUENCY_FACTOR.get(tf, 1) for tf in timeframes
    )

    if frequency_factor > 100:  # Timeframes très courts (1m, 5m)
        return 7
    if frequency_factor > 20:  # Timeframes courts (15m, 30m, 1h)
        return 30
    return 90  # Timeframes longs (4h, 1d, 1w)


def get_timeframe_category(timeframe: str) -> str:
    """Retourne la catégorie d'un timeframe ou 'mixed' si inconnue."""
    for category, tf_list in TIMEFRAME_CATEGORIES.items():
        if timeframe in tf_list:
            return category
    return "mixed"


def analyze_by_category(
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Analyse les données par catégorie de timeframe.

    Args:
        symbols: Liste des symboles
        timeframes: Liste des timeframes
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        Dict avec analyse par catégorie
    """
    categories = categorize_timeframes(timeframes)
    analysis = {}

    for category, category_timeframes in categories.items():
        if not category_timeframes:
            continue

        # Analyse standard pour cette catégorie
        availability_result = scan_data_availability(symbols, category_timeframes)

        # Périodes optimales spécialisées pour cette catégorie
        optimal_periods = []
        if availability_result.has_common_range:
            min_period_days = get_min_period_days_for_timeframes(category_timeframes)

            optimal_periods = find_optimal_periods(
                symbols=symbols,
                timeframes=category_timeframes,
                min_period_days=min_period_days,
                max_periods=3
            )

        analysis[category] = {
            'timeframes': category_timeframes,
            'availability': availability_result,
            'optimal_periods': optimal_periods,
            'gap_tolerance': CATEGORY_GAP_TOLERANCE.get(category, 10.0),
            'frequency_factor': sum(TIMEFRAME_FREQUENCY_FACTOR.get(tf, 1) for tf in category_timeframes),
            'recommendations': _get_category_recommendations(category, optimal_periods)
        }

    return analysis


def analyze_by_timeframe(
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Analyse les données par timeframe (plage commune par TF).

    Args:
        symbols: Liste des symboles
        timeframes: Liste des timeframes
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        Dict avec analyse par timeframe
    """
    analysis: Dict[str, Any] = {}

    for timeframe in timeframes:
        availability_result = scan_data_availability(symbols, [timeframe])

        optimal_periods: List[OptimalPeriod] = []
        if availability_result.has_common_range:
            min_period_days = get_min_period_days_for_timeframes([timeframe])
            optimal_periods = find_longest_common_period_for_timeframe(
                symbols=symbols,
                timeframe=timeframe,
                min_period_days=min_period_days,
            )

        category = get_timeframe_category(timeframe)

        analysis[timeframe] = {
            "timeframes": [timeframe],
            "availability": availability_result,
            "optimal_periods": optimal_periods,
            "gap_tolerance": CATEGORY_GAP_TOLERANCE.get(category, 0.10),
            "frequency_factor": TIMEFRAME_FREQUENCY_FACTOR.get(timeframe, 1),
            "recommendations": _get_category_recommendations(category, optimal_periods),
        }

    return analysis


def find_harmonized_period(category_analysis: Dict[str, Any]) -> Optional[OptimalPeriod]:
    """
    Trouve une période harmonisée entre toutes les catégories.

    Args:
        category_analysis: Résultat d'analyze_by_category()

    Returns:
        Période optimale commune ou None
    """
    all_periods = []
    all_timeframes = []

    # Collecter toutes les périodes et timeframes
    for category, data in category_analysis.items():
        all_periods.extend(data['optimal_periods'])
        all_timeframes.extend(data['timeframes'])

    if not all_periods:
        return None

    # Trouver la période avec le meilleur score global
    best_period = max(all_periods, key=lambda p: p.completeness_score * p.avg_data_density)

    # Créer une période harmonisée
    return OptimalPeriod(
        start_date=best_period.start_date,
        end_date=best_period.end_date,
        completeness_score=best_period.completeness_score,
        tokens_complete=best_period.tokens_complete,
        tokens_total=best_period.tokens_total,
        avg_data_density=best_period.avg_data_density,
        description=f"Période harmonisée (basée sur {best_period.description})",
        category="harmonized",
        timeframes=all_timeframes
    )


def _get_category_recommendations(category: str, optimal_periods: List[OptimalPeriod]) -> List[str]:
    """
    Génère des recommandations spécifiques à une catégorie.

    Args:
        category: Nom de la catégorie
        optimal_periods: Périodes optimales trouvées

    Returns:
        Liste de recommandations
    """
    recommendations = []

    if not optimal_periods:
        recommendations.append(f"❌ Aucune période optimale trouvée pour {category}")
        return recommendations

    best_period = optimal_periods[0]

    if category == 'scalping':
        if best_period.avg_data_density < 0.95:
            recommendations.append("⚠️ Scalping nécessite des données très denses")
        if (best_period.end_date - best_period.start_date).days < 30:
            recommendations.append("⚠️ Période courte pour backtests scalping fiables")

    elif category == 'intraday':
        if best_period.completeness_score < 0.9:
            recommendations.append("⚠️ Gaps de données problématiques pour intraday")
        duration_days = (best_period.end_date - best_period.start_date).days
        if duration_days > 365:
            recommendations.append("✅ Période longue idéale pour intraday")

    elif category == 'swing':
        tolerance = CATEGORY_GAP_TOLERANCE.get(category, 10.0)
        if best_period.completeness_score < (100 - tolerance) / 100:
            recommendations.append("⚠️ Gaps importants même pour swing trading")
        else:
            recommendations.append("✅ Gaps acceptables pour swing trading")

    elif category == 'position':
        if best_period.completeness_score > 0.8:  # Position trading tolère plus de gaps
            recommendations.append("✅ Qualité suffisante pour position trading")
        duration_days = (best_period.end_date - best_period.start_date).days
        if duration_days > 1000:
            recommendations.append("✅ Période très longue excellente pour position")

    return recommendations


def discover_available_data() -> Tuple[List[str], List[str]]:
    """
    Découvre les données disponibles dans le système.

    Returns:
        Tuple (liste de symboles, liste de timeframes)
    """
    try:
        from data.loader import discover_available_data as _discover
        return _discover()
    except (ImportError, AttributeError):
        return (["BTCUSDC", "ETHUSDC"], ["1h", "4h", "1d"])


# ============================================================================
# AVAILABILITY SCANNING
# ============================================================================

def scan_data_availability(
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    find_optimal: bool = True
) -> DataAvailabilityResult:
    """
    Scanne la disponibilité des données pour toutes les combinaisons symbole/timeframe.

    AMÉLIORÉ:
    - Vérifie la complétude réelle des données avec filtrage de dates
    - Trouve automatiquement les périodes optimales si find_optimal=True

    Cette fonction fait le calcul métier SANS aucun appel Streamlit.

    Args:
        symbols: Liste des symboles à scanner
        timeframes: Liste des timeframes à scanner
        start_date: Date de début pour vérification de complétude (optionnel)
        end_date: Date de fin pour vérification de complétude (optionnel)
        find_optimal: Si True, calcule aussi les périodes optimales

    Returns:
        DataAvailabilityResult avec toutes les infos de disponibilité + périodes optimales
    """
    result = DataAvailabilityResult()

    all_starts: List[pd.Timestamp] = []
    all_ends: List[pd.Timestamp] = []

    for symbol in symbols:
        for tf in timeframes:
            # D'abord vérifier la plage globale disponible
            date_range = get_data_date_range(symbol, tf)

            if date_range:
                data_start, data_end = date_range
                result.availability[(symbol, tf)] = date_range
                all_starts.append(data_start)
                all_ends.append(data_end)
                coverage_pct = None
                missing_pct = None
                missing_days = None

                # Si des dates de filtrage sont spécifiées, vérifier la complétude réelle
                if start_date and end_date:
                    is_complete, message, actual_rows = check_data_completeness(
                        symbol, tf, start_date, end_date
                    )

                    if is_complete:
                        status = "✅"
                        status_msg = f"Complet ({actual_rows} barres)"
                    else:
                        status = "⚠️"
                        status_msg = message.replace("⚠️ ", "").replace("❌ ", "")
                        result.missing_data.append(f"{symbol}/{tf} (données incomplètes)")
                else:
                    # Pas de vérification de complétude, analyser les gaps
                    gaps = analyze_data_gaps(symbol, tf)
                    if gaps:
                        total_gap_days = sum(gap.duration_days for gap in gaps)
                        total_days = (data_end - data_start).days
                        gap_ratio = total_gap_days / max(total_days, 1)
                        coverage_pct = max(0.0, 1.0 - gap_ratio) * 100.0
                        missing_pct = gap_ratio * 100.0
                        missing_days = total_gap_days

                        if gap_ratio > 0.1:  # Plus de 10% de gaps
                            status = "⚠️"
                            status_msg = f"{len(gaps)} gaps ({total_gap_days:.1f}j manquants)"
                        else:
                            status = "✅"
                            status_msg = f"Quasi-complet ({len(gaps)} petits gaps)"
                    else:
                        status = "✅"
                        days = (data_end - data_start).days
                        status_msg = f"{days} jours complets"
                        coverage_pct = 100.0
                        missing_pct = 0.0
                        missing_days = 0.0

                result.rows.append({
                    "Token": symbol,
                    "TF": tf,
                    "Début": data_start.strftime("%Y-%m-%d"),
                    "Fin": data_end.strftime("%Y-%m-%d"),
                    "Jours": (data_end - data_start).days,
                    "Couverture %": coverage_pct,
                    "Manquant %": missing_pct,
                    "Jours manquants": missing_days,
                    "Plage commune %": None,
                    "Status": status,
                    "Détails": status_msg
                })
            else:
                result.missing_data.append(f"{symbol}/{tf}")
                result.rows.append({
                    "Token": symbol,
                    "TF": tf,
                    "Début": "-",
                    "Fin": "-",
                    "Jours": 0,
                    "Couverture %": None,
                    "Manquant %": None,
                    "Jours manquants": None,
                    "Plage commune %": None,
                    "Status": "❌",
                    "Détails": "Fichier non trouvé"
                })

    # Calculer la plage commune (intersection)
    if all_starts and all_ends:
        result.common_start = max(all_starts)
        result.common_end = min(all_ends)
        result.has_common_range = result.common_start < result.common_end
        if result.has_common_range:
            common_days = (result.common_end - result.common_start).days
            for row in result.rows:
                days = row.get("Jours", 0) or 0
                if days > 0:
                    row["Plage commune %"] = round((common_days / days) * 100.0, 1)
                else:
                    row["Plage commune %"] = None

    # Trouver les périodes optimales si demandé
    if find_optimal and len(symbols) > 0 and len(timeframes) > 0:
        optimal_periods = find_optimal_periods(symbols, timeframes)
        # Stocker dans le result pour usage ultérieur
        if hasattr(result, 'optimal_periods'):
            result.optimal_periods = optimal_periods
        else:
            # Ajouter dynamiquement l'attribut
            setattr(result, 'optimal_periods', optimal_periods)

    return result


# ============================================================================
# INTELLIGENT DEFAULTS
# ============================================================================

def get_intelligent_timeframe_defaults(
    available_timeframes: List[str],
    common_days: Optional[int] = None
) -> List[str]:
    """
    Sélectionne des timeframes par défaut basés sur la plage commune.

    Règles:
    - Plage < 30 jours → timeframes courts (15m, 30m, 1h)
    - Plage 30-180 jours → timeframes moyens (30m, 1h, 4h)
    - Plage > 180 jours → timeframes longs (1h, 4h, 1d)

    Args:
        available_timeframes: Timeframes disponibles
        common_days: Nombre de jours dans la plage commune

    Returns:
        Liste de 1-2 timeframes recommandés
    """
    if not available_timeframes:
        return []

    sorted_tfs = sorted(
        available_timeframes,
        key=lambda tf: TIMEFRAME_PRIORITY_MAP.get(tf, 999)
    )

    if common_days is None or common_days >= 180:
        preferred = ["1h", "4h", "1d"]
    elif common_days >= 30:
        preferred = ["30m", "1h", "4h"]
    else:
        preferred = ["15m", "30m", "1h"]

    defaults = []
    for pref in preferred:
        if pref in available_timeframes and len(defaults) < 2:
            defaults.append(pref)

    if not defaults and sorted_tfs:
        defaults = [sorted_tfs[0]]

    return defaults


def generate_random_token_suggestions(
    available_tokens: List[str],
    selected_tokens: List[str],
    count: int = RANDOM_SLOTS_COUNT
) -> List[str]:
    """
    Génère des suggestions de tokens aléatoires avec anti-doublons.

    Args:
        available_tokens: Tous les tokens disponibles
        selected_tokens: Tokens déjà sélectionnés (exclus)
        count: Nombre de suggestions à générer

    Returns:
        Liste de tokens uniques non sélectionnés
    """
    pool = [t for t in available_tokens if t not in selected_tokens]

    if not pool:
        return []

    return random.sample(pool, min(count, len(pool)))


# ============================================================================
# PERIOD VALIDATION
# ============================================================================

def validate_period_for_tokens(
    start_date: date,
    end_date: date,
    data_availability: Dict[Tuple[str, str], Tuple[pd.Timestamp, pd.Timestamp]]
) -> PeriodValidationResult:
    """
    Valide une période sélectionnée contre les données disponibles.

    Compare au niveau du jour pour éviter les faux positifs dus aux heures.

    Args:
        start_date: Date de début sélectionnée
        end_date: Date de fin sélectionnée
        data_availability: Dict {(symbol, tf): (start_ts, end_ts)}

    Returns:
        PeriodValidationResult avec tokens ok/partiels/manquants
    """
    result = PeriodValidationResult()

    # Normaliser en dates pures (pas de timestamps)
    start_day = start_date if isinstance(start_date, date) else start_date.date()
    end_day = end_date if isinstance(end_date, date) else end_date.date()

    for (symbol, tf), (data_start, data_end) in data_availability.items():
        token_key = f"{symbol}/{tf}"

        # Comparer au niveau du jour
        data_start_day = data_start.date() if hasattr(data_start, 'date') else data_start
        data_end_day = data_end.date() if hasattr(data_end, 'date') else data_end

        if end_day < data_start_day or start_day > data_end_day:
            result.tokens_missing.append(token_key)
        elif start_day < data_start_day or end_day > data_end_day:
            result.tokens_partial.append(token_key)
        else:
            result.tokens_ok.append(token_key)

    result.all_ok = (
        len(result.tokens_ok) > 0
        and len(result.tokens_missing) == 0
        and len(result.tokens_partial) == 0
    )

    return result


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_date_fr(date_obj) -> str:
    """
    Formate une date au format français JJ/MM/AAAA.

    Args:
        date_obj: datetime.date, pandas.Timestamp ou string

    Returns:
        String au format "01/12/2025"
    """
    if isinstance(date_obj, str):
        try:
            parsed = pd.to_datetime(date_obj)
            return parsed.strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            return date_obj
    elif hasattr(date_obj, 'strftime'):
        return date_obj.strftime("%d/%m/%Y")
    else:
        return str(date_obj)


def compute_period_days(start_date: date, end_date: date) -> int:
    """
    Calcule le nombre de jours entre deux dates.

    Args:
        start_date: Date de début
        end_date: Date de fin

    Returns:
        Nombre de jours (inclusif)
    """
    return (end_date - start_date).days