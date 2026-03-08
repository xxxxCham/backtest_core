"""
Module-ID: data.loader

Purpose: Chargement OHLCV multi-formats (CSV, Parquet, JSON, Feather) + découverte auto.

Role in pipeline: data input

Key components: load_ohlcv(), discover_available_data(), _get_data_dir()

Inputs: CSV/Parquet/JSON/Feather files, env vars BACKTEST_DATA_DIR/TRADX_DATA_ROOT

Outputs: Normalized pandas DataFrame {timestamp, open, high, low, close, volume}

Dependencies: pandas, pathlib, numpy, functools

Conventions: DatetimeIndex; OHLCV colonnes lowercase; env var priority; @lru_cache.

Read-if: Modification formats supports ou paths par défaut.

Skip-if: Vous appelez load_ohlcv(filename).
"""

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Trim post-listing : supprime les premières barres erratiques d'un token
#   max(TRIM_MIN_HOURS, TRIM_PCT% du total)
# Désactivé si TRIM_PCT = 0.  Défaut : 2% avec plancher 24h.
# ---------------------------------------------------------------------------
TRIM_LAUNCH_PCT = float(os.environ.get("BACKTEST_TRIM_LAUNCH_PCT", "2"))
TRIM_LAUNCH_MIN_HOURS = int(os.environ.get("BACKTEST_TRIM_LAUNCH_MIN_HOURS", "24"))
TRIM_LAUNCH_MAX_DAYS = int(os.environ.get("BACKTEST_TRIM_LAUNCH_MAX_DAYS", "30"))
DATASET_QUALITY_MODE = str(os.environ.get("BACKTEST_DATASET_QUALITY_MODE", "error")).strip().lower()
DATASET_MIN_COVERAGE_RATIO = float(os.environ.get("BACKTEST_DATASET_MIN_COVERAGE_RATIO", "0.85"))
DATASET_MAX_LARGEST_GAP_RATIO = float(os.environ.get("BACKTEST_DATASET_MAX_LARGEST_GAP_RATIO", "0.10"))
DATASET_MAX_ZERO_VOLUME_RATIO = float(os.environ.get("BACKTEST_DATASET_MAX_ZERO_VOLUME_RATIO", "0.05"))

MIN_BARS_BY_TIMEFRAME = {
    "1M": 24,
    "1w": 52,
    "1d": 180,
    "4h": 300,
    "1h": 500,
    "30m": 750,
    "15m": 1000,
    "5m": 1500,
    "3m": 1800,
    "1m": 2000,
}


@dataclass(frozen=True)
class DatasetQualityReport:
    """Rapport synthétique de qualité d'un dataset prêt à l'emploi."""

    symbol: str
    timeframe: str
    source_path: Optional[str]
    bars_before_trim: int
    launch_trim_bars: int
    bars_after_filter: int
    start_ts: Optional[str]
    end_ts: Optional[str]
    gap_count: int
    missing_bars: int
    largest_gap_bars: int
    coverage_ratio: float
    zero_volume_bars: int
    zero_volume_ratio: float
    invalid_ohlc_bars: int
    min_required_bars: int
    is_valid: bool
    blocking_reasons: Tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        reasons = ", ".join(self.blocking_reasons) if self.blocking_reasons else "ok"
        return (
            f"{self.symbol}/{self.timeframe}: {self.bars_after_filter} barres utiles, "
            f"trim lancement={self.launch_trim_bars}, couverture={self.coverage_ratio * 100:.1f}%, "
            f"gaps={self.gap_count} ({self.missing_bars} barres), "
            f"plus gros gap={self.largest_gap_bars}, volume=0={self.zero_volume_ratio * 100:.1f}% "
            f"[{reasons}]"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source_path": self.source_path,
            "bars_before_trim": self.bars_before_trim,
            "launch_trim_bars": self.launch_trim_bars,
            "bars_after_filter": self.bars_after_filter,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "gap_count": self.gap_count,
            "missing_bars": self.missing_bars,
            "largest_gap_bars": self.largest_gap_bars,
            "coverage_ratio": self.coverage_ratio,
            "coverage_pct": self.coverage_ratio * 100.0,
            "zero_volume_bars": self.zero_volume_bars,
            "zero_volume_ratio": self.zero_volume_ratio,
            "zero_volume_pct": self.zero_volume_ratio * 100.0,
            "invalid_ohlc_bars": self.invalid_ohlc_bars,
            "min_required_bars": self.min_required_bars,
            "is_valid": self.is_valid,
            "blocking_reasons": list(self.blocking_reasons),
            "summary": self.summary(),
        }

def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convertit un timeframe string ('1h', '15m', '1d') en pd.Timedelta."""
    if not timeframe or len(timeframe) < 2:
        raise ValueError(f"Timeframe invalide: '{timeframe}'")

    unit = timeframe[-1]
    try:
        amount = int(timeframe[:-1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Timeframe invalide: '{timeframe}'") from exc

    if amount <= 0:
        raise ValueError(f"Timeframe invalide (<=0): '{timeframe}'")

    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    if unit == "w":
        return pd.Timedelta(weeks=amount)
    if unit == "M":
        # Timedelta ne supporte pas "ME" (MonthEnd). Approximation cohérente avec le reste du code.
        return pd.Timedelta(days=30 * amount)

    raise ValueError(f"Unité de timeframe non supportée: '{timeframe}'")


def _minimum_bars_for_timeframe(timeframe: str) -> int:
    """Retourne un seuil minimum de barres utiles selon le timeframe."""
    return int(MIN_BARS_BY_TIMEFRAME.get(timeframe, 200))


def _apply_date_window(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Applique une fenêtre temporelle avec inclusion des dates pures en fin de journée."""
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_exclusive = end_ts + pd.Timedelta(days=1)
            df = df[df.index < end_exclusive]
        else:
            df = df[df.index <= end_ts]

    return df


def _count_invalid_ohlc_bars(df: pd.DataFrame) -> int:
    """Compte les barres OHLC incohérentes."""
    if df.empty:
        return 0
    invalid = (
        (df["high"] < df["low"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
    )
    return int(invalid.sum())


def _resolve_effective_quality_mode(enforce_quality: Optional[bool]) -> str:
    """Résout le mode effectif de contrôle qualité."""
    if enforce_quality is True:
        return "error"
    if enforce_quality is False:
        return "warn"
    if DATASET_QUALITY_MODE in {"off", "warn", "error"}:
        return DATASET_QUALITY_MODE
    return "error"


def _trim_launch_period(
    df: pd.DataFrame,
    timeframe: str,
    trim_pct: Optional[float] = None,
    min_hours: Optional[int] = None,
) -> pd.DataFrame:
    """
    Supprime les premières barres post-listing d'un DataFrame OHLCV.

    Nombre de barres supprimées = max(floor_24h, pct% du total).

    Args:
        df: DataFrame avec DatetimeIndex trié
        timeframe: Timeframe des barres ('1h', '15m', …)
        trim_pct: % de barres à couper (None → env BACKTEST_TRIM_LAUNCH_PCT)
        min_hours: Plancher en heures (None → env BACKTEST_TRIM_LAUNCH_MIN_HOURS)

    Returns:
        DataFrame tronqué (inchangé si désactivé ou données insuffisantes)
    """
    pct = trim_pct if trim_pct is not None else TRIM_LAUNCH_PCT
    if pct <= 0 or df.empty:
        return df

    floor_h = min_hours if min_hours is not None else TRIM_LAUNCH_MIN_HOURS

    # Nombre de barres correspondant au plancher horaire
    bar_delta = _timeframe_to_timedelta(timeframe)
    floor_bars = max(1, int(pd.Timedelta(hours=floor_h) / bar_delta))

    # Nombre de barres correspondant au pourcentage
    pct_bars = int(len(df) * pct / 100)

    trim_count = max(floor_bars, pct_bars)

    max_days = max(0, TRIM_LAUNCH_MAX_DAYS)
    if max_days > 0:
        cap_bars = max(1, int(pd.Timedelta(days=max_days) / bar_delta))
        trim_count = min(trim_count, cap_bars)

    # Sécurité : ne jamais trimmer plus de 50 % des données
    if trim_count >= len(df) * 0.5:
        logger.warning(
            f"⚠️  Trim launch: {trim_count} barres ({pct}% / floor {floor_h}h) "
            f"≥ 50 % du dataset ({len(df)} barres). Trim ignoré."
        )
        return df

    trimmed = df.iloc[trim_count:]
    trim_end_ts = trimmed.index[0]
    logger.info(
        f"✂️  Trim post-listing: {trim_count} barres supprimées "
        f"({df.index[0]} → {trim_end_ts}, "
        f"max({floor_bars} barres/{floor_h}h, {pct_bars} barres/{pct}%), "
        f"cap {max_days}j)"
    )
    return trimmed


def _mark_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute la colonne ``_tradable`` au DataFrame OHLCV.

    _tradable = False pour les bougies à volume nul (synthétiques ou suspectes).
    Le moteur de backtest masque les signaux d'entrée sur ces barres.
    Les barres restent dans le DataFrame pour la continuité des indicateurs.
    """
    if df.empty:
        return df

    df["_tradable"] = df["volume"] > 0

    n_untradable = int((~df["_tradable"]).sum())
    if n_untradable > 0:
        ratio = n_untradable / len(df)
        logger.info(
            f"📊 Data quality: {n_untradable}/{len(df)} barres non-tradables "
            f"({ratio:.1%}) — volume=0"
        )

    return df


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_multiplier: float = 2.0
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Détecte les gaps (discontinuités) dans un DataFrame OHLCV.

    Args:
        df: DataFrame avec DatetimeIndex
        timeframe: Timeframe attendu (ex: "1h", "30m")
        max_gap_multiplier: Gap = trou > expected_delta * multiplier

    Returns:
        Liste de (gap_start, gap_end, nb_barres_manquantes)
    """
    if len(df) < 2:
        return []

    expected_delta = _timeframe_to_timedelta(timeframe)

    gaps = []
    timestamps = df.index

    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i-1]

        if delta > expected_delta * max_gap_multiplier:
            nb_missing = int(delta / expected_delta) - 1
            gaps.append((timestamps[i-1], timestamps[i], nb_missing))

    return gaps


def apply_dataset_quality_stage(
    df: pd.DataFrame,
    timeframe: str,
    *,
    symbol: str = "UNKNOWN",
    source_path: Optional[Path] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    trim_launch_pct: Optional[float] = None,
    enforce_quality: Optional[bool] = None,
) -> Tuple[pd.DataFrame, DatasetQualityReport]:
    """
    Étape centrale de préparation/qualification d'un dataset avant usage.

    Pipeline :
    1. Trim post-listing
    2. Filtrage date éventuel
    3. Marquage des barres non-tradables
    4. Contrôle de couverture/gaps/volume nul/cohérence OHLC
    """
    if not is_valid_timeframe(timeframe):
        raise ValueError(f"Timeframe invalide pour contrôle qualité: '{timeframe}'")

    mode = _resolve_effective_quality_mode(enforce_quality)
    raw_df = df.copy()
    raw_bars = len(raw_df)

    trimmed_df = _trim_launch_period(raw_df, timeframe, trim_pct=trim_launch_pct)
    launch_trim_bars = max(0, raw_bars - len(trimmed_df))

    filtered_df = _apply_date_window(trimmed_df, start=start, end=end)
    if filtered_df.empty:
        raise ValueError(
            f"Aucune donnée exploitable après contrôle qualité pour {symbol}/{timeframe} "
            f"(trim={launch_trim_bars} barres, fenêtre={start} -> {end})"
        )

    invalid_ohlc_bars = _count_invalid_ohlc_bars(filtered_df)
    filtered_df = _mark_data_quality(filtered_df.copy())

    gaps = detect_gaps(filtered_df, timeframe, max_gap_multiplier=2.0)
    gap_count = len(gaps)
    missing_bars = int(sum(nb_missing for _, _, nb_missing in gaps))
    largest_gap_bars = max((nb_missing for _, _, nb_missing in gaps), default=0)
    expected_bars = len(filtered_df) + missing_bars
    coverage_ratio = (len(filtered_df) / expected_bars) if expected_bars > 0 else 1.0
    zero_volume_bars = int((filtered_df["volume"] <= 0).sum())
    zero_volume_ratio = (zero_volume_bars / len(filtered_df)) if len(filtered_df) > 0 else 0.0
    largest_gap_ratio = (largest_gap_bars / expected_bars) if expected_bars > 0 else 0.0
    min_required_bars = _minimum_bars_for_timeframe(timeframe)

    blocking_reasons: List[str] = []
    if len(filtered_df) < min_required_bars:
        blocking_reasons.append(f"not_enough_bars<{min_required_bars}")
    if invalid_ohlc_bars > 0:
        blocking_reasons.append(f"invalid_ohlc={invalid_ohlc_bars}")
    if coverage_ratio < DATASET_MIN_COVERAGE_RATIO:
        blocking_reasons.append(f"coverage<{DATASET_MIN_COVERAGE_RATIO:.2f}")
    if largest_gap_ratio > DATASET_MAX_LARGEST_GAP_RATIO:
        blocking_reasons.append(f"largest_gap>{DATASET_MAX_LARGEST_GAP_RATIO:.2f}")
    if zero_volume_ratio > DATASET_MAX_ZERO_VOLUME_RATIO:
        blocking_reasons.append(f"zero_volume>{DATASET_MAX_ZERO_VOLUME_RATIO:.2f}")

    report = DatasetQualityReport(
        symbol=str(symbol or "UNKNOWN").upper(),
        timeframe=timeframe,
        source_path=str(source_path) if source_path else None,
        bars_before_trim=raw_bars,
        launch_trim_bars=launch_trim_bars,
        bars_after_filter=len(filtered_df),
        start_ts=str(filtered_df.index[0]) if len(filtered_df) else None,
        end_ts=str(filtered_df.index[-1]) if len(filtered_df) else None,
        gap_count=gap_count,
        missing_bars=missing_bars,
        largest_gap_bars=largest_gap_bars,
        coverage_ratio=coverage_ratio,
        zero_volume_bars=zero_volume_bars,
        zero_volume_ratio=zero_volume_ratio,
        invalid_ohlc_bars=invalid_ohlc_bars,
        min_required_bars=min_required_bars,
        is_valid=not blocking_reasons,
        blocking_reasons=tuple(blocking_reasons),
    )

    filtered_df.attrs["dataset_quality"] = report.to_dict()
    filtered_df.attrs["dataset_quality_report"] = report
    filtered_df.attrs["symbol"] = report.symbol
    filtered_df.attrs["timeframe"] = timeframe

    if not report.is_valid:
        message = f"Dataset rejeté: {report.summary()}"
        if mode == "error":
            raise ValueError(message)
        if mode == "warn":
            logger.warning("⚠️ %s", message)
    else:
        logger.info("✅ Dataset qualifié: %s", report.summary())

    return filtered_df, report

# Extensions supportées
SUPPORTED_EXTENSIONS = (".parquet", ".feather", ".csv", ".json")
SUPPORTED_EXTENSION_SET = {ext.lower() for ext in SUPPORTED_EXTENSIONS}

# Dossiers à exclure du scan de données (cache/artefacts locaux)
IGNORED_SCAN_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tmp",
    ".tmp_pytest_codex",
    ".tmp_pytest_codex_run",
    "pytest_temp",
    ".git",
    ".venv",
    ".venv_old",
}

# Nom de fichier accepté:
#   SYMBOL_TIMEFRAME.ext
#   SYMBOL_TIMEFRAME_<suffix>.ext
#   SYMBOL-TIMEFRAME-suffix.ext (support partiel via séparateur après timeframe)
DATA_FILE_STEM_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9][A-Za-z0-9.\-]*)_(?P<timeframe>\d+[mhdwM])(?:$|[_-].*)"
)

# Répertoire de données par défaut
DEFAULT_DATA_DIR = Path(__file__).parent / "sample_data"

def _optional_env_path(key: str) -> Optional[Path]:
    value = os.environ.get(key)
    if not value:
        return None
    return Path(value)


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    """Supprime les doublons de chemins en conservant l'ordre."""
    seen = set()
    out: List[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _extract_symbol_timeframe_from_stem(stem: str) -> Optional[Tuple[str, str]]:
    """Extrait (SYMBOL, timeframe) depuis un nom de fichier sans extension."""
    match = DATA_FILE_STEM_RE.match(stem)
    if not match:
        return None

    symbol = match.group("symbol").upper()
    timeframe = match.group("timeframe")
    if not is_valid_timeframe(timeframe):
        return None
    return symbol, timeframe


def _iter_supported_files(root: Path):
    """Parcourt récursivement les fichiers supportés en excluant les dossiers de cache."""
    if not root.exists() or not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_SCAN_DIRS]
        current = Path(dirpath)
        for name in filenames:
            suffix = Path(name).suffix.lower()
            if suffix in SUPPORTED_EXTENSION_SET:
                yield current / name


def _build_legacy_windows_data_dirs() -> List[Path]:
    """Construit des candidats Windows connus pour auto-détection data."""
    if os.name != "nt":
        return []

    drive_root = Path(Path.cwd().anchor) if Path.cwd().anchor else Path("D:/")
    home_root = Path.home()

    candidates = [
        # Banque principale du gestionnaire multi-timeframe
        drive_root / ".my_soft" / "gestionnaire_telechargement_multi-timeframe" / "processed" / "parquet",
        drive_root / ".my_soft" / "gestionnaire_telechargement_multi-timeframe" / "raw",
        # Variante sous profil utilisateur
        home_root / ".my_soft" / "gestionnaire_telechargement_multi-timeframe" / "processed" / "parquet",
        home_root / ".my_soft" / "gestionnaire_telechargement_multi-timeframe" / "raw",
        # Compat historique ThreadX
        drive_root / "ThreadX_big" / "data",
        drive_root / "ThreadX_big" / "processed_data",
    ]
    return _dedupe_paths(candidates)


def _path_has_supported_data(path: Path) -> bool:
    """Vérifie si un dossier contient au moins un fichier OHLCV détectable."""
    if not path.exists() or not path.is_dir():
        return False

    for file_path in _iter_supported_files(path):
        if _extract_symbol_timeframe_from_stem(file_path.stem) is not None:
            return True

    return False


# Chemins de données via variables d'environnement (prioritaires)
THREADX_DATA_DIR = _optional_env_path("BACKTEST_CORE_DATA_DIR")
GESTIONNAIRE_DATA_DIR = _optional_env_path("BACKTEST_CORE_GESTIONNAIRE_DIR")
GESTIONNAIRE_RAW_DIR = _optional_env_path("BACKTEST_CORE_RAW_DIR")

# Compatibilité Windows (auto-détection banque externe sans env obligatoire).
LEGACY_WINDOWS_DATA_DIRS: List[Path] = _build_legacy_windows_data_dirs()


def _get_data_dir() -> Path:
    """Détermine le répertoire de données à utiliser."""
    # Variable d'environnement prioritaire
    env_dir = os.environ.get("BACKTEST_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if _path_has_supported_data(env_path):
            return env_path
        if env_path.exists():
            logger.warning(
                f"BACKTEST_DATA_DIR existe mais ne contient pas de fichiers OHLCV valides: {env_path}"
            )

    # Variable TRADX_DATA_ROOT (compatibilité)
    tradx_dir = os.environ.get("TRADX_DATA_ROOT")
    if tradx_dir:
        tradx_path = Path(tradx_dir)
        if _path_has_supported_data(tradx_path):
            return tradx_path
        if tradx_path.exists():
            logger.warning(
                f"TRADX_DATA_ROOT existe mais ne contient pas de fichiers OHLCV valides: {tradx_path}"
            )

    # Gestionnaire multi-timeframe (données processed)
    if GESTIONNAIRE_DATA_DIR and _path_has_supported_data(GESTIONNAIRE_DATA_DIR):
        return GESTIONNAIRE_DATA_DIR

    # Gestionnaire multi-timeframe (données raw en fallback)
    if GESTIONNAIRE_RAW_DIR and _path_has_supported_data(GESTIONNAIRE_RAW_DIR):
        return GESTIONNAIRE_RAW_DIR

    # Chemin ThreadX_big (données principales)
    if THREADX_DATA_DIR and _path_has_supported_data(THREADX_DATA_DIR):
        return THREADX_DATA_DIR

    # Auto-détection: prioriser les banques externes, puis fallback projet.
    project_data_dir = Path(__file__).parent.parent / "data"
    candidates = _dedupe_paths(
        LEGACY_WINDOWS_DATA_DIRS
        + [
            Path.cwd() / "data",
            project_data_dir,
            Path.cwd() / "data" / "sample_data",
            project_data_dir / "sample_data",
            DEFAULT_DATA_DIR,
        ]
    )

    for candidate in candidates:
        if _path_has_supported_data(candidate):
            return candidate

    # Créer le répertoire par défaut
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


@lru_cache(maxsize=4)
def _scan_data_files_for_dir(data_dir: str) -> Tuple[Path, ...]:
    """Scanne les fichiers de données disponibles."""
    root = Path(data_dir)
    files = list(_iter_supported_files(root))

    return tuple(sorted(set(files)))


def _scan_data_files() -> Tuple[Path, ...]:
    """Scanne les fichiers de données depuis le répertoire actuellement résolu."""
    data_dir = str(_get_data_dir().resolve())
    return _scan_data_files_for_dir(data_dir)


def discover_available_data() -> Tuple[List[str], List[str]]:
    """
    Découvre les tokens et timeframes disponibles.

    Returns:
        Tuple (liste de tokens, liste de timeframes)
    """
    tokens = set()
    timeframes = set()

    for file_path in _scan_data_files():
        parsed = _extract_symbol_timeframe_from_stem(file_path.stem)
        if parsed is None:
            continue
        symbol, tf = parsed
        tokens.add(symbol)
        timeframes.add(tf)

    # Tri des timeframes par ordre logique
    def tf_sort_key(tf: str) -> Tuple[int, int]:
        if not tf:
            return (99, 0)
        unit = tf[-1]
        try:
            amount = int(tf[:-1])
        except ValueError:
            amount = 0
        order = {"m": 0, "h": 1, "d": 2, "w": 3, "M": 4}.get(unit, 5)
        return (order, amount)

    return sorted(tokens), sorted(timeframes, key=tf_sort_key)


def is_valid_timeframe(tf: str) -> bool:
    """
    Valide qu'un timeframe est dans un format correct.

    Args:
        tf: Timeframe à valider (ex: "1m", "5m", "1h", "4h", "1d")

    Returns:
        True si le timeframe est valide, False sinon
    """
    if not tf or len(tf) < 2:
        return False

    # Validation supplémentaire : rejeter patterns problématiques
    problematic_patterns = [
        r".*\.meta$",    # Fichiers .meta
        r".*\.data$",    # Fichiers .data
        r".*\.cache$",   # Fichiers cache
        r".*_backup$",   # Fichiers backup
        r".*\.tmp$",     # Fichiers temporaires
    ]

    for pattern in problematic_patterns:
        if re.match(pattern, tf, re.IGNORECASE):
            return False

    # Doit se terminer par m, h, d, w ou M
    unit = tf[-1]
    if unit not in ('m', 'h', 'd', 'w', 'M'):
        return False

    # La partie numérique doit être un entier positif
    try:
        amount = int(tf[:-1])
        return amount > 0
    except ValueError:
        return False


def _read_file(path: Path) -> pd.DataFrame:
    """Lit un fichier de données selon son extension."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".feather":
        return pd.read_feather(path)
    elif suffix == ".csv":
        return pd.read_csv(path, parse_dates=True)
    elif suffix == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Extension non supportée: {suffix}")


def _find_data_file(symbol: str, timeframe: str) -> Optional[Path]:
    """Cherche le fichier de données correspondant.

    IMPORTANT: Le symbole est comparé en case-insensitive (BTCUSDC = btcusdc),
    mais le timeframe est comparé en case-SENSITIVE pour distinguer
    '1m' (1 minute) de '1M' (1 mois).
    """
    symbol = symbol.upper()

    for file_path in _scan_data_files():
        parsed = _extract_symbol_timeframe_from_stem(file_path.stem)
        if parsed is None:
            continue
        parsed_symbol, parsed_tf = parsed
        if parsed_symbol == symbol and parsed_tf == timeframe:
            return file_path

    return None


def load_ohlcv_file(
    file_path: Path,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    trim_launch_pct: Optional[float] = None,
    enforce_quality: Optional[bool] = None,
) -> Tuple[pd.DataFrame, DatasetQualityReport]:
    """
    Charge un fichier OHLCV arbitraire puis applique l'étape qualité centrale.
    """
    resolved_path = Path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {resolved_path}")

    parsed = _extract_symbol_timeframe_from_stem(resolved_path.stem)
    parsed_symbol = parsed[0] if parsed else None
    parsed_timeframe = parsed[1] if parsed else None
    effective_symbol = str(symbol or parsed_symbol or "UNKNOWN").upper()
    effective_timeframe = str(timeframe or parsed_timeframe or "").strip()

    if not effective_timeframe:
        raise ValueError(
            f"Impossible de déduire le timeframe depuis {resolved_path.name}; "
            "fournissez-le explicitement."
        )

    df = _read_file(resolved_path)
    df = _normalize_ohlcv(df)
    return apply_dataset_quality_stage(
        df,
        effective_timeframe,
        symbol=effective_symbol,
        source_path=resolved_path,
        start=start,
        end=end,
        trim_launch_pct=trim_launch_pct,
        enforce_quality=enforce_quality,
    )


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un DataFrame OHLCV au format standard.

    Format de sortie:
    - Index: DatetimeIndex (UTC)
    - Colonnes: open, high, low, close, volume (minuscules)
    - Types: float64 pour OHLCV
    """
    df = df.copy()

    # Normaliser les noms de colonnes (minuscules)
    df.columns = df.columns.str.lower()

    # Mapper les variantes de noms courantes
    column_map = {
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "prix_ouverture": "open", "prix_haut": "high", "prix_bas": "low",
        "prix_cloture": "close", "vol": "volume"
    }
    df = df.rename(columns=column_map)

    # Vérifier colonnes requises
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Configurer l'index datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        # Chercher colonne de temps
        time_cols = ["timestamp", "time", "datetime", "date", "ts"]
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            # Détecter le format du timestamp (millisecondes vs secondes vs datetime)
            sample_ts = df[time_col].iloc[0]
            if isinstance(sample_ts, (int, float, np.integer, np.floating)):
                # Convertir en float pour éviter problème numpy
                sample_val = float(sample_ts)
                if sample_val > 1e12:
                    # Timestamp en millisecondes
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms')
                elif sample_val > 1e9:
                    # Timestamp en secondes
                    df[time_col] = pd.to_datetime(df[time_col], unit='s')
                else:
                    # Format datetime normal
                    df[time_col] = pd.to_datetime(df[time_col])
            else:
                # String datetime
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            # Essayer de parser l'index
            df.index = pd.to_datetime(df.index)

    # Convertir en UTC si nécessaire
    if hasattr(df.index, 'tz') and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")  # type: ignore[union-attr]
    elif hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")  # type: ignore[union-attr]

    # Sélectionner et trier
    df = df[required].copy()
    df = df.sort_index()

    # Convertir types
    for col in required:
        df[col] = df[col].astype(np.float64)

    # Nettoyer NaN
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        logger.warning(f"Supprimé {original_len - len(df)} lignes avec NaN")

    return df


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    trim_launch_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Charge les données OHLCV pour un symbole et timeframe.

    Args:
        symbol: Symbole de l'actif (ex: "BTCUSDT")
        timeframe: Intervalle de temps (ex: "1m", "1h", "1d")
        start: Date de début (optionnel, format ISO)
        end: Date de fin (optionnel, format ISO)
            - Si date pure (ex: "2025-02-28"), inclut toute la journée
            - Si date+heure (ex: "2025-02-28 12:00:00"), précision stricte
        trim_launch_pct: % de barres post-listing à couper (plancher 24h).
            None = utilise BACKTEST_TRIM_LAUNCH_PCT (env var, défaut 2).

    Returns:
        DataFrame OHLCV normalisé avec index datetime UTC

    Raises:
        FileNotFoundError: Si aucun fichier correspondant n'est trouvé
        ValueError: Si le fichier est invalide

    Note:
        Dates end "pures" (sans heure) incluent toutes les barres du jour.
        Ex: end="2025-02-28" → inclut barres jusqu'à 23:59:59
    """
    if not is_valid_timeframe(timeframe):
        raise ValueError(
            f"Timeframe invalide: '{timeframe}'. "
            "Format attendu: <nombre><unité> (ex: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)."
        )

    logger.info(f"Chargement données: {symbol}/{timeframe}")

    # Chercher le fichier
    file_path = _find_data_file(symbol, timeframe)
    if file_path is None:
        data_dir = _get_data_dir()
        raise FileNotFoundError(
            f"Fichier OHLCV introuvable pour {symbol}/{timeframe} dans {data_dir}"
        )

    df, report = load_ohlcv_file(
        file_path,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        trim_launch_pct=trim_launch_pct,
    )

    logger.info(
        "  Après contrôle qualité: %s → %s (%s barres utiles, couverture %.1f%%)",
        df.index[0],
        df.index[-1],
        len(df),
        report.coverage_ratio * 100.0,
    )

    return df


def get_available_timeframes(symbol: str) -> List[str]:
    """Retourne les timeframes disponibles pour un symbole."""
    symbol = symbol.upper()
    timeframes = set()

    for file_path in _scan_data_files():
        parsed = _extract_symbol_timeframe_from_stem(file_path.stem)
        if parsed is None:
            continue
        parsed_symbol, parsed_tf = parsed
        if parsed_symbol == symbol:
            timeframes.add(parsed_tf)

    return sorted(timeframes)


def get_data_date_range(
    symbol: str,
    timeframe: str
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Retourne la plage de dates disponible pour un symbole/timeframe.

    Args:
        symbol: Symbole de l'actif (ex: "BTCUSDC")
        timeframe: Intervalle de temps (ex: "1h")

    Returns:
        Tuple (date_debut, date_fin) ou None si fichier non trouvé
    """
    file_path = _find_data_file(symbol, timeframe)
    if file_path is None:
        return None

    try:
        df, _ = load_ohlcv_file(
            file_path,
            symbol=symbol,
            timeframe=timeframe,
            enforce_quality=False,
        )
        if df.empty:
            return None
        return (df.index[0], df.index[-1])
    except (OSError, ValueError, KeyError, IndexError):
        return None


@lru_cache(maxsize=1024)
def is_usable_dataset(symbol: str, timeframe: str) -> bool:
    """Retourne True si le dataset passe l'étape qualité centrale."""
    file_path = _find_data_file(symbol, timeframe)
    if file_path is None:
        return False
    try:
        _, report = load_ohlcv_file(
            file_path,
            symbol=symbol,
            timeframe=timeframe,
            enforce_quality=False,
        )
        return bool(report.is_valid)
    except (FileNotFoundError, OSError, ValueError, KeyError, IndexError):
        return False


__all__ = [
    "DatasetQualityReport",
    "apply_dataset_quality_stage",
    "discover_available_data",
    "get_available_timeframes",
    "get_data_date_range",
    "is_usable_dataset",
    "load_ohlcv",
    "load_ohlcv_file",
]
