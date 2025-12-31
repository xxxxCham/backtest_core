"""
Module-ID: backtest.storage

Purpose: Persister et indexer les résultats de backtests pour rechargement/recherche rapide.

Role in pipeline: persistence / reporting

Key components: ResultStorage, StoredResultMetadata, get_storage

Inputs: RunResult, run_id, auto_cleanup flag

Outputs: Fichiers JSON/Parquet dans backtest_results/{run_id}/, index.json

Dependencies: pandas, pathlib, json, optionnel: pyarrow (Parquet)

Conventions: Structure run_id/metadata.json + equity.parquet + trades.parquet; index.json catalogue; auto_cleanup garde N derniers runs.

Read-if: Persistance résultats, recherche historique, ou gestion stockage.

Skip-if: Backtests ponctuels sans sauvegarde.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backtest.engine import RunResult
from backtest.sweep import SweepResults
from metrics_types import PerformanceMetricsPct, normalize_metrics
from utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_STORAGE_DIR = Path("backtest_results")
MAX_RESULTS_TO_KEEP = 1000  # Nombre maximum de résultats à garder


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StoredResultMetadata:
    """Métadonnées d'un résultat sauvegardé."""
    run_id: str
    timestamp: str
    strategy: str
    symbol: str
    timeframe: str
    params: Dict[str, Any]
    metrics: PerformanceMetricsPct
    n_bars: int
    n_trades: int
    period_start: str
    period_end: str
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dict pour sérialisation."""
        payload = asdict(self)
        payload["metrics"] = normalize_metrics(self.metrics, "pct")
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredResultMetadata":
        """Crée depuis un dict."""
        metrics = normalize_metrics(data.get("metrics", {}), "pct")
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            strategy=data["strategy"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            params=data.get("params", {}),
            metrics=metrics,
            n_bars=data["n_bars"],
            n_trades=data["n_trades"],
            period_start=data.get("period_start", ""),
            period_end=data.get("period_end", ""),
            duration_sec=data.get("duration_sec", 0.0),
        )


# =============================================================================
# STORAGE ENGINE
# =============================================================================

class ResultStorage:
    """
    Gestionnaire de stockage des résultats de backtests.

    Features:
    - Sauvegarde automatique avec structure organisée
    - Index pour recherche rapide
    - Compression optionnelle
    - Nettoyage automatique des anciens résultats

    Example:
        >>> storage = ResultStorage()
        >>>
        >>> # Sauvegarder un résultat
        >>> storage.save_result(run_result)
        >>>
        >>> # Lister tous les résultats
        >>> all_results = storage.list_results()
        >>>
        >>> # Rechercher
        >>> best_runs = storage.search_results(min_sharpe=2.0)
        >>>
        >>> # Charger un résultat spécifique
        >>> result = storage.load_result(run_id)
    """

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
        compress: bool = False,
    ):
        """
        Initialise le gestionnaire de stockage.

        Args:
            storage_dir: Répertoire de stockage (défaut: backtest_results/)
            auto_save: Activer la sauvegarde automatique
            compress: Compresser les fichiers Parquet
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.auto_save = auto_save
        self.compress = compress

        # Créer le répertoire si nécessaire
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Chemin de l'index
        self.index_path = self.storage_dir / "index.json"

        # Charger ou créer l'index
        self._index: Dict[str, StoredResultMetadata] = self._load_index()

        logger.info(f"ResultStorage initialisé: {self.storage_dir} ({len(self._index)} résultats)")

    # =========================================================================
    # SAUVEGARDE
    # =========================================================================

    def save_result(
        self,
        result: RunResult,
        run_id: Optional[str] = None,
        auto_cleanup: bool = False,
    ) -> str:
        """
        Sauvegarde un résultat de backtest.

        Args:
            result: RunResult à sauvegarder
            run_id: ID personnalisé (sinon utilise result.meta['run_id'])
            auto_cleanup: Nettoyer les anciens résultats si trop nombreux

        Returns:
            run_id du résultat sauvegardé
        """
        # Récupérer ou générer le run_id
        if run_id is None:
            run_id = result.meta.get("run_id", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Créer le répertoire du run
        run_dir = self.storage_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Sauvegarder les métadonnées
            metrics_pct = normalize_metrics(result.metrics, "pct")
            metadata = StoredResultMetadata(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                strategy=result.meta.get("strategy", "unknown"),
                symbol=result.meta.get("symbol", "unknown"),
                timeframe=result.meta.get("timeframe", "unknown"),
                params=result.meta.get("params", {}),
                metrics=metrics_pct,
                n_bars=result.meta.get("n_bars", len(result.equity)),
                n_trades=len(result.trades),
                period_start=result.meta.get("period_start", ""),
                period_end=result.meta.get("period_end", ""),
                duration_sec=result.meta.get("duration_sec", 0.0),
            )

            metadata_path = run_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

            # 2. Sauvegarder la courbe d'équité
            equity_path = run_dir / "equity.parquet"
            equity_df = result.equity.to_frame(name="equity")
            equity_df.to_parquet(
                equity_path,
                compression="snappy" if self.compress else None,
                index=True,
            )

            # 3. Sauvegarder les trades
            trades_path = run_dir / "trades.parquet"
            result.trades.to_parquet(
                trades_path,
                compression="snappy" if self.compress else None,
                index=False,
            )

            # 4. Sauvegarder les returns
            returns_path = run_dir / "returns.parquet"
            returns_df = result.returns.to_frame(name="returns")
            returns_df.to_parquet(
                returns_path,
                compression="snappy" if self.compress else None,
                index=True,
            )

            # 5. Mettre à jour l'index
            self._index[run_id] = metadata
            self._save_index()

            logger.info(f"✅ Résultat sauvegardé: {run_id} ({metadata.strategy})")

            # Nettoyage optionnel
            if auto_cleanup:
                self._cleanup_old_results()

            return run_id

        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
            # Nettoyer en cas d'erreur
            if run_dir.exists():
                shutil.rmtree(run_dir)
            raise

    def save_sweep_results(
        self,
        sweep_results: SweepResults,
        sweep_id: Optional[str] = None,
    ) -> str:
        """
        Sauvegarde les résultats d'un sweep.

        Args:
            sweep_results: SweepResults à sauvegarder
            sweep_id: ID personnalisé du sweep

        Returns:
            sweep_id
        """
        if sweep_id is None:
            sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        sweep_dir = self.storage_dir / sweep_id
        sweep_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Sauvegarder le résumé
            summary = {
                "sweep_id": sweep_id,
                "timestamp": datetime.now().isoformat(),
                "n_completed": sweep_results.n_completed,
                "n_failed": sweep_results.n_failed,
                "total_time": sweep_results.total_time,
                "best_params": sweep_results.best_params,
                "best_metrics": normalize_metrics(sweep_results.best_metrics, "pct"),
                "resource_stats": sweep_results.resource_stats,
            }

            summary_path = sweep_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Sauvegarder tous les résultats en DataFrame
            results_df = sweep_results.to_dataframe()
            results_path = sweep_dir / "all_results.parquet"
            results_df.to_parquet(
                results_path,
                compression="snappy" if self.compress else None,
                index=False,
            )

            logger.info(f"✅ Sweep sauvegardé: {sweep_id} ({sweep_results.n_completed} résultats)")

            return sweep_id

        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du sweep: {e}")
            if sweep_dir.exists():
                shutil.rmtree(sweep_dir)
            raise

    # =========================================================================
    # CHARGEMENT
    # =========================================================================

    def load_result(self, run_id: str) -> RunResult:
        """
        Charge un résultat de backtest.

        Args:
            run_id: ID du run à charger

        Returns:
            RunResult reconstruit

        Raises:
            FileNotFoundError: Si le run_id n'existe pas
        """
        run_dir = self.storage_dir / run_id

        if not run_dir.exists():
            raise FileNotFoundError(f"Run inexistant: {run_id}")

        try:
            # 1. Charger les métadonnées
            metadata_path = run_dir / "metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            metadata = StoredResultMetadata.from_dict(metadata_dict)

            # 2. Charger l'équité
            equity_path = run_dir / "equity.parquet"
            equity_df = pd.read_parquet(equity_path)
            equity = equity_df["equity"]

            # 3. Charger les trades
            trades_path = run_dir / "trades.parquet"
            trades = pd.read_parquet(trades_path)

            # 4. Charger les returns
            returns_path = run_dir / "returns.parquet"
            returns_df = pd.read_parquet(returns_path)
            returns = returns_df["returns"]

            # 5. Reconstruire le RunResult
            result = RunResult(
                equity=equity,
                returns=returns,
                trades=trades,
                metrics=metadata.metrics,
                meta={
                    "run_id": metadata.run_id,
                    "strategy": metadata.strategy,
                    "symbol": metadata.symbol,
                    "timeframe": metadata.timeframe,
                    "params": metadata.params,
                    "n_bars": metadata.n_bars,
                    "period_start": metadata.period_start,
                    "period_end": metadata.period_end,
                    "duration_sec": metadata.duration_sec,
                    "loaded_from_storage": True,
                    "loaded_at": datetime.now().isoformat(),
                }
            )

            logger.info(f"✅ Résultat chargé: {run_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de {run_id}: {e}")
            raise

    def load_sweep_results(self, sweep_id: str) -> Dict[str, Any]:
        """
        Charge les résultats d'un sweep.

        Args:
            sweep_id: ID du sweep

        Returns:
            Dict avec summary et results_df
        """
        sweep_dir = self.storage_dir / sweep_id

        if not sweep_dir.exists():
            raise FileNotFoundError(f"Sweep inexistant: {sweep_id}")

        try:
            # Charger le résumé
            summary_path = sweep_dir / "summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["best_metrics"] = normalize_metrics(
                summary.get("best_metrics", {}), "pct"
            )

            # Charger les résultats
            results_path = sweep_dir / "all_results.parquet"
            results_df = pd.read_parquet(results_path)

            logger.info(f"✅ Sweep chargé: {sweep_id}")

            return {
                "summary": summary,
                "results_df": results_df,
                "sweep_id": sweep_id,
            }

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du sweep {sweep_id}: {e}")
            raise

    # =========================================================================
    # RECHERCHE & LISTAGE
    # =========================================================================

    def list_results(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        reverse: bool = True,
    ) -> List[StoredResultMetadata]:
        """
        Liste tous les résultats disponibles.

        Args:
            limit: Limiter le nombre de résultats
            sort_by: Champ de tri (timestamp, sharpe_ratio, etc.)
            reverse: Tri descendant

        Returns:
            Liste de métadonnées
        """
        results = list(self._index.values())

        # Tri
        if sort_by == "timestamp":
            results.sort(key=lambda x: x.timestamp, reverse=reverse)
        elif sort_by == "sharpe_ratio":
            results.sort(
                key=lambda x: x.metrics.get("sharpe_ratio", 0),
                reverse=reverse
            )
        elif sort_by == "total_return":
            results.sort(
                key=lambda x: x.metrics.get("total_return_pct", 0),
                reverse=reverse
            )

        # Limite
        if limit:
            results = results[:limit]

        return results

    def search_results(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[StoredResultMetadata]:
        """
        Recherche des résultats avec filtres.

        Args:
            strategy: Nom de la stratégie
            symbol: Symbole
            timeframe: Timeframe
            min_sharpe: Sharpe ratio minimum
            max_drawdown: Drawdown maximum (%)
            min_trades: Nombre minimum de trades
            date_from: Date minimum (ISO format)
            date_to: Date maximum (ISO format)

        Returns:
            Liste de métadonnées filtrées
        """
        results = list(self._index.values())

        # Filtres
        if strategy:
            results = [r for r in results if r.strategy == strategy]

        if symbol:
            results = [r for r in results if r.symbol == symbol]

        if timeframe:
            results = [r for r in results if r.timeframe == timeframe]

        if min_sharpe is not None:
            results = [
                r for r in results
                if r.metrics.get("sharpe_ratio", 0) >= min_sharpe
            ]

        if max_drawdown is not None:
            results = [
                r for r in results
                if r.metrics.get("max_drawdown_pct", 100) <= max_drawdown
            ]

        if min_trades is not None:
            results = [r for r in results if r.n_trades >= min_trades]

        if date_from:
            results = [r for r in results if r.timestamp >= date_from]

        if date_to:
            results = [r for r in results if r.timestamp <= date_to]

        return results

    def get_best_results(
        self,
        n: int = 10,
        metric: str = "sharpe_ratio",
    ) -> List[StoredResultMetadata]:
        """
        Retourne les N meilleurs résultats selon une métrique.

        Args:
            n: Nombre de résultats
            metric: Métrique de tri

        Returns:
            Liste des meilleurs résultats
        """
        results = list(self._index.values())
        results.sort(
            key=lambda x: x.metrics.get(metric, float("-inf")),
            reverse=True
        )
        return results[:n]

    # =========================================================================
    # GESTION
    # =========================================================================

    def delete_result(self, run_id: str) -> bool:
        """
        Supprime un résultat.

        Args:
            run_id: ID du run à supprimer

        Returns:
            True si supprimé, False sinon
        """
        run_dir = self.storage_dir / run_id

        if not run_dir.exists():
            logger.warning(f"⚠️ Run inexistant: {run_id}")
            return False

        try:
            shutil.rmtree(run_dir)

            if run_id in self._index:
                del self._index[run_id]
                self._save_index()

            logger.info(f"🗑️ Résultat supprimé: {run_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression: {e}")
            return False

    def _cleanup_old_results(self, keep_last: int = MAX_RESULTS_TO_KEEP) -> int:
        """
        Nettoie les anciens résultats pour éviter l'accumulation.

        Args:
            keep_last: Nombre de résultats à garder

        Returns:
            Nombre de résultats supprimés
        """
        results = list(self._index.values())
        results.sort(key=lambda x: x.timestamp, reverse=True)

        to_delete = results[keep_last:]
        deleted_count = 0

        for result in to_delete:
            if self.delete_result(result.run_id):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"🧹 Nettoyage: {deleted_count} anciens résultats supprimés")

        return deleted_count

    def clear_all(self) -> bool:
        """
        Supprime TOUS les résultats (attention!).

        Returns:
            True si succès
        """
        try:
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(parents=True, exist_ok=True)

            self._index = {}
            self._save_index()

            logger.warning("🧹 TOUS les résultats ont été supprimés")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage: {e}")
            return False

    # =========================================================================
    # INDEX
    # =========================================================================

    def _load_index(self) -> Dict[str, StoredResultMetadata]:
        """Charge l'index depuis le disque."""
        if not self.index_path.exists():
            return {}

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            # Reconstruire les objets StoredResultMetadata
            index = {}
            for run_id, meta_dict in index_data.items():
                try:
                    index[run_id] = StoredResultMetadata.from_dict(meta_dict)
                except Exception as e:
                    logger.warning(f"⚠️ Métadonnée corrompue pour {run_id}: {e}")

            return index

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de l'index: {e}")
            return {}

    def _save_index(self) -> None:
        """Sauvegarde l'index sur le disque."""
        try:
            index_data = {
                run_id: meta.to_dict()
                for run_id, meta in self._index.items()
            }

            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde de l'index: {e}")

    def rebuild_index(self) -> int:
        """
        Reconstruit l'index en scannant tous les répertoires.

        Utile en cas de corruption ou si des fichiers ont été ajoutés manuellement.

        Returns:
            Nombre de résultats indexés
        """
        logger.info("🔄 Reconstruction de l'index...")

        self._index = {}
        count = 0

        for run_dir in self.storage_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)

                metadata = StoredResultMetadata.from_dict(meta_dict)
                self._index[metadata.run_id] = metadata
                count += 1

            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger {run_dir.name}: {e}")

        self._save_index()
        logger.info(f"✅ Index reconstruit: {count} résultats")

        return count


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

_storage_instance: Optional[ResultStorage] = None


def get_storage(
    storage_dir: Optional[Union[str, Path]] = None,
    auto_save: bool = True,
    compress: bool = False,
) -> ResultStorage:
    """
    Retourne l'instance globale de ResultStorage (singleton).

    Args:
        storage_dir: Répertoire de stockage
        auto_save: Activer la sauvegarde automatique
        compress: Compresser les fichiers

    Returns:
        ResultStorage instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = ResultStorage(
            storage_dir=storage_dir,
            auto_save=auto_save,
            compress=compress,
        )
    return _storage_instance


__all__ = [
    "ResultStorage",
    "StoredResultMetadata",
    "get_storage",
]


# Docstring update summary
# - Docstring de module normalisée (LLM-friendly) centrée sur persistance/indexation
# - Conventions structure répertoires et index.json explicitées
# - Read-if/Skip-if ajoutés pour tri rapide
