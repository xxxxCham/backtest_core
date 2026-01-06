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

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.log import get_logger

logger = get_logger(__name__)

# Extensions supportées
SUPPORTED_EXTENSIONS = (".parquet", ".feather", ".csv", ".json")

# Répertoire de données par défaut
DEFAULT_DATA_DIR = Path(__file__).parent / "sample_data"

# Chemin ThreadX_big (données principales)
THREADX_DATA_DIR = Path("D:/ThreadX_big/data/crypto/processed/parquet")


def _get_data_dir() -> Path:
    """Détermine le répertoire de données à utiliser."""
    # Variable d'environnement prioritaire
    env_dir = os.environ.get("BACKTEST_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.exists():
            return env_path

    # Variable TRADX_DATA_ROOT (compatibilité)
    tradx_dir = os.environ.get("TRADX_DATA_ROOT")
    if tradx_dir:
        tradx_path = Path(tradx_dir)
        if tradx_path.exists():
            return tradx_path

    # Chemin ThreadX_big (données principales)
    if THREADX_DATA_DIR.exists():
        return THREADX_DATA_DIR

    # Répertoire par défaut
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR

    # Chercher dans des emplacements courants
    candidates = [
        Path("D:/ThreadX_big/data/crypto/processed/parquet"),
        Path.cwd() / "data" / "sample_data",
        Path.cwd() / "data",
        Path(__file__).parent.parent.parent / "data",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Créer le répertoire par défaut
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


@lru_cache(maxsize=1)
def _scan_data_files() -> Tuple[Path, ...]:
    """Scanne les fichiers de données disponibles."""
    data_dir = _get_data_dir()
    files: List[Path] = []

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(data_dir.glob(f"*{ext}"))
        files.extend(data_dir.glob(f"**/*{ext}"))  # Récursif

    return tuple(sorted(set(files)))


def discover_available_data() -> Tuple[List[str], List[str]]:
    """
    Découvre les tokens et timeframes disponibles.

    Returns:
        Tuple (liste de tokens, liste de timeframes)
    """
    tokens = set()
    timeframes = set()

    for file_path in _scan_data_files():
        # Format attendu: SYMBOL_TIMEFRAME.ext (ex: BTCUSDT_1m.parquet)
        parts = file_path.stem.split("_", 1)
        if len(parts) == 2:
            symbol, tf = parts
            tokens.add(symbol.upper())
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
        order = {"m": 0, "h": 1, "d": 2, "w": 3}.get(unit, 4)
        return (order, amount)

    return sorted(tokens), sorted(timeframes, key=tf_sort_key)


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
    """Cherche le fichier de données correspondant."""
    symbol = symbol.upper()
    target = f"{symbol}_{timeframe}"

    for file_path in _scan_data_files():
        if file_path.stem.upper() == target.upper():
            return file_path

    return None


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
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Charge les données OHLCV pour un symbole et timeframe.

    Args:
        symbol: Symbole de l'actif (ex: "BTCUSDT")
        timeframe: Intervalle de temps (ex: "1m", "1h", "1d")
        start: Date de début (optionnel, format ISO)
        end: Date de fin (optionnel, format ISO)

    Returns:
        DataFrame OHLCV normalisé avec index datetime UTC

    Raises:
        FileNotFoundError: Si aucun fichier correspondant n'est trouvé
        ValueError: Si le fichier est invalide
    """
    logger.info(f"Chargement données: {symbol}/{timeframe}")

    # Chercher le fichier
    file_path = _find_data_file(symbol, timeframe)
    if file_path is None:
        data_dir = _get_data_dir()
        raise FileNotFoundError(
            f"Fichier OHLCV introuvable pour {symbol}/{timeframe} dans {data_dir}"
        )

    # Lire et normaliser
    df = _read_file(file_path)
    df = _normalize_ohlcv(df)

    logger.info(f"  Période: {df.index[0]} → {df.index[-1]} ({len(df)} barres)")

    # Stocker les bornes disponibles pour message d'erreur
    data_start = df.index[0]
    data_end = df.index[-1]

    # Filtrer par dates si spécifié
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[df.index <= end_ts]

    if df.empty:
        raise ValueError(
            f"Aucune donnée dans la période {start} - {end}. "
            f"Données disponibles: {data_start.strftime('%Y-%m-%d')} → "
            f"{data_end.strftime('%Y-%m-%d')}"
        )

    logger.info(f"  Après filtrage: {len(df)} barres")

    return df


def get_available_timeframes(symbol: str) -> List[str]:
    """Retourne les timeframes disponibles pour un symbole."""
    symbol = symbol.upper()
    timeframes = set()

    for file_path in _scan_data_files():
        parts = file_path.stem.split("_", 1)
        if len(parts) == 2 and parts[0].upper() == symbol:
            timeframes.add(parts[1])

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
        df = _read_file(file_path)
        df = _normalize_ohlcv(df)
        if df.empty:
            return None
        return (df.index[0], df.index[-1])
    except Exception:
        return None


__all__ = ["load_ohlcv", "discover_available_data", "get_available_timeframes", "get_data_date_range"]
