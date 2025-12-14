"""
Tests pour le système de stockage des résultats de backtests.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import BacktestEngine, RunResult
from backtest.storage import ResultStorage, StoredResultMetadata, get_storage
from backtest.sweep import SweepEngine
from strategies.bollinger_atr import BollingerATRStrategy


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    return tmp_path / "test_storage"


@pytest.fixture
def storage(temp_storage_dir):
    """Crée une instance de ResultStorage pour les tests."""
    return ResultStorage(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_data():
    """Génère des données OHLCV de test."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    data = {
        "open": 100 + pd.Series(range(100)) * 0.1,
        "high": 101 + pd.Series(range(100)) * 0.1,
        "low": 99 + pd.Series(range(100)) * 0.1,
        "close": 100.5 + pd.Series(range(100)) * 0.1,
        "volume": 1000 + pd.Series(range(100)) * 10,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_result(sample_data):
    """Génère un RunResult de test."""
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(
        df=sample_data,
        strategy=BollingerATRStrategy(),
        params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 1},
        symbol="TESTUSDT",
        timeframe="1h",
    )
    return result


# =============================================================================
# TESTS SAUVEGARDE
# =============================================================================

def test_save_result(storage, sample_result):
    """Test sauvegarde d'un résultat."""
    run_id = storage.save_result(sample_result)

    assert run_id is not None
    assert (storage.storage_dir / run_id).exists()
    assert (storage.storage_dir / run_id / "metadata.json").exists()
    assert (storage.storage_dir / run_id / "equity.parquet").exists()
    assert (storage.storage_dir / run_id / "trades.parquet").exists()
    assert (storage.storage_dir / run_id / "returns.parquet").exists()


def test_save_result_custom_run_id(storage, sample_result):
    """Test sauvegarde avec run_id personnalisé."""
    custom_id = "custom_test_run"
    run_id = storage.save_result(sample_result, run_id=custom_id)

    assert run_id == custom_id
    assert (storage.storage_dir / custom_id).exists()


def test_save_result_updates_index(storage, sample_result):
    """Test que la sauvegarde met à jour l'index."""
    initial_count = len(storage._index)
    storage.save_result(sample_result)

    assert len(storage._index) == initial_count + 1


def test_save_multiple_results(storage, sample_result):
    """Test sauvegarde de plusieurs résultats."""
    run_id_1 = storage.save_result(sample_result, run_id="test_run_1")
    run_id_2 = storage.save_result(sample_result, run_id="test_run_2")

    assert run_id_1 != run_id_2
    assert len(storage._index) == 2


# =============================================================================
# TESTS CHARGEMENT
# =============================================================================

def test_load_result(storage, sample_result):
    """Test chargement d'un résultat."""
    run_id = storage.save_result(sample_result)
    loaded_result = storage.load_result(run_id)

    assert isinstance(loaded_result, RunResult)
    assert len(loaded_result.equity) == len(sample_result.equity)
    assert len(loaded_result.trades) == len(sample_result.trades)
    assert loaded_result.metrics == sample_result.metrics


def test_load_nonexistent_result(storage):
    """Test chargement d'un résultat inexistant."""
    with pytest.raises(FileNotFoundError):
        storage.load_result("nonexistent_run_id")


def test_load_preserves_metadata(storage, sample_result):
    """Test que le chargement préserve les métadonnées."""
    run_id = storage.save_result(sample_result)
    loaded_result = storage.load_result(run_id)

    assert loaded_result.meta["strategy"] == sample_result.meta["strategy"]
    assert loaded_result.meta["symbol"] == sample_result.meta["symbol"]
    assert loaded_result.meta["timeframe"] == sample_result.meta["timeframe"]


# =============================================================================
# TESTS RECHERCHE
# =============================================================================

def test_list_results_empty(storage):
    """Test listage sur stockage vide."""
    results = storage.list_results()
    assert len(results) == 0


def test_list_results(storage, sample_result):
    """Test listage des résultats."""
    storage.save_result(sample_result, run_id="test_list_1")
    storage.save_result(sample_result, run_id="test_list_2")

    results = storage.list_results()
    assert len(results) == 2
    assert all(isinstance(r, StoredResultMetadata) for r in results)


def test_list_results_with_limit(storage, sample_result):
    """Test listage avec limite."""
    for i in range(5):
        storage.save_result(sample_result, run_id=f"test_limit_{i}")

    results = storage.list_results(limit=3)
    assert len(results) == 3


def test_search_by_strategy(storage, sample_result):
    """Test recherche par stratégie."""
    storage.save_result(sample_result)

    # Le nom de la stratégie est "BollingerATR" (pas "bollinger_atr")
    results = storage.search_results(strategy="BollingerATR")
    assert len(results) == 1

    results = storage.search_results(strategy="ema_cross")
    assert len(results) == 0


def test_search_by_symbol(storage, sample_result):
    """Test recherche par symbole."""
    storage.save_result(sample_result)

    results = storage.search_results(symbol="TESTUSDT")
    assert len(results) == 1

    results = storage.search_results(symbol="BTCUSDT")
    assert len(results) == 0


def test_search_by_min_sharpe(storage, sample_result):
    """Test recherche par Sharpe ratio minimum."""
    storage.save_result(sample_result)

    # Obtenir le sharpe du résultat
    sharpe = sample_result.metrics.get("sharpe_ratio", 0)

    results = storage.search_results(min_sharpe=sharpe - 1)
    assert len(results) >= 1

    results = storage.search_results(min_sharpe=sharpe + 10)
    assert len(results) == 0


def test_get_best_results(storage, sample_result):
    """Test récupération des meilleurs résultats."""
    for _ in range(5):
        storage.save_result(sample_result)

    best = storage.get_best_results(n=3, metric="sharpe_ratio")
    assert len(best) <= 3
    assert all(isinstance(r, StoredResultMetadata) for r in best)


# =============================================================================
# TESTS GESTION
# =============================================================================

def test_delete_result(storage, sample_result):
    """Test suppression d'un résultat."""
    run_id = storage.save_result(sample_result)
    assert (storage.storage_dir / run_id).exists()

    success = storage.delete_result(run_id)
    assert success is True
    assert not (storage.storage_dir / run_id).exists()
    assert run_id not in storage._index


def test_delete_nonexistent_result(storage):
    """Test suppression d'un résultat inexistant."""
    success = storage.delete_result("nonexistent")
    assert success is False


def test_clear_all(storage, sample_result):
    """Test nettoyage complet."""
    storage.save_result(sample_result, run_id="test_clear_1")
    storage.save_result(sample_result, run_id="test_clear_2")

    assert len(storage._index) == 2

    success = storage.clear_all()
    assert success is True
    assert len(storage._index) == 0


def test_rebuild_index(storage, sample_result):
    """Test reconstruction de l'index."""
    # Sauvegarder quelques résultats
    storage.save_result(sample_result, run_id="test_rebuild_1")
    storage.save_result(sample_result, run_id="test_rebuild_2")

    # Simuler corruption de l'index
    storage._index = {}

    # Reconstruire
    count = storage.rebuild_index()
    assert count == 2
    assert len(storage._index) == 2


# =============================================================================
# TESTS INDEX
# =============================================================================

def test_index_persistence(temp_storage_dir, sample_result):
    """Test que l'index persiste entre les instances."""
    # Créer et sauvegarder
    storage1 = ResultStorage(storage_dir=temp_storage_dir)
    storage1.save_result(sample_result)
    run_id = list(storage1._index.keys())[0]

    # Créer nouvelle instance
    storage2 = ResultStorage(storage_dir=temp_storage_dir)

    # Vérifier que l'index est chargé
    assert run_id in storage2._index
    assert len(storage2._index) == 1


# =============================================================================
# TESTS SWEEP
# =============================================================================

def test_save_sweep_results(storage, sample_data):
    """Test sauvegarde des résultats de sweep."""
    engine = SweepEngine(max_workers=2)
    sweep_results = engine.run_sweep(
        df=sample_data,
        strategy=BollingerATRStrategy(),
        param_grid={"entry_z": [1.5, 2.0], "k_sl": [1.0, 1.5]},
        show_progress=False,
    )

    sweep_id = storage.save_sweep_results(sweep_results)

    assert sweep_id is not None
    assert (storage.storage_dir / sweep_id).exists()
    assert (storage.storage_dir / sweep_id / "summary.json").exists()
    assert (storage.storage_dir / sweep_id / "all_results.parquet").exists()


def test_load_sweep_results(storage, sample_data):
    """Test chargement des résultats de sweep."""
    engine = SweepEngine(max_workers=2, auto_save=False)
    sweep_results = engine.run_sweep(
        df=sample_data,
        strategy=BollingerATRStrategy(),
        param_grid={"entry_z": [1.5, 2.0]},
        show_progress=False,
    )

    sweep_id = storage.save_sweep_results(sweep_results)
    loaded = storage.load_sweep_results(sweep_id)

    assert loaded is not None
    assert "summary" in loaded
    assert "results_df" in loaded
    assert len(loaded["results_df"]) == sweep_results.n_completed


# =============================================================================
# TESTS INTÉGRATION AUTO_SAVE
# =============================================================================

def test_engine_auto_save_disabled(temp_storage_dir, sample_data):
    """Test que le moteur n'enregistre pas automatiquement sans appel explicite à storage."""
    storage = ResultStorage(storage_dir=temp_storage_dir)
    initial_count = len(storage._index)

    engine = BacktestEngine()
    result = engine.run(
        df=sample_data,
        strategy=BollingerATRStrategy(),
        params={"entry_z": 2.0},
    )

    # Recharger l'index - aucune sauvegarde automatique
    storage2 = ResultStorage(storage_dir=temp_storage_dir)
    assert len(storage2._index) == initial_count


def test_sweep_engine_auto_save_disabled(temp_storage_dir, sample_data):
    """Test que le sweep n'enregistre pas automatiquement sans appel explicite à storage."""
    storage = ResultStorage(storage_dir=temp_storage_dir)

    engine = SweepEngine(max_workers=2)
    results = engine.run_sweep(
        df=sample_data,
        strategy=BollingerATRStrategy(),
        param_grid={"entry_z": [1.5, 2.0]},
        show_progress=False,
    )

    # Vérifier qu'aucun dossier sweep_* n'a été créé automatiquement
    sweep_dirs = list(temp_storage_dir.glob("sweep_*"))
    assert len(sweep_dirs) == 0


# =============================================================================
# TESTS EDGE CASES
# =============================================================================

def test_save_with_missing_metadata(storage):
    """Test sauvegarde avec métadonnées incomplètes."""
    # Créer un RunResult minimal
    equity = pd.Series([10000, 10100, 10200], index=pd.date_range("2023-01-01", periods=3, freq="1h"))
    returns = pd.Series([0, 0.01, 0.01], index=equity.index)
    trades = pd.DataFrame()

    result = RunResult(
        equity=equity,
        returns=returns,
        trades=trades,
        metrics={},
        meta={},
    )

    # Ne devrait pas lever d'exception
    run_id = storage.save_result(result)
    assert run_id is not None


def test_compression_enabled(temp_storage_dir, sample_result):
    """Test sauvegarde avec compression."""
    storage_compressed = ResultStorage(storage_dir=temp_storage_dir, compress=True)
    run_id = storage_compressed.save_result(sample_result)

    # Vérifier que le fichier existe et est lisible
    loaded = storage_compressed.load_result(run_id)
    assert loaded is not None
