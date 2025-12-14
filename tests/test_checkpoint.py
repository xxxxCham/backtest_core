"""
Tests pour Checkpoint Manager.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_checkpoint_dir():
    """Crée un répertoire temporaire pour les checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCheckpointMetadata:
    """Tests pour CheckpointMetadata."""
    
    def test_create_metadata(self):
        """Test création metadata."""
        from utils.checkpoint import CheckpointMetadata
        
        metadata = CheckpointMetadata(
            checkpoint_id="test_123",
            created_at="2024-01-01T00:00:00",
            operation_type="sweep",
            total_items=100,
        )
        
        assert metadata.checkpoint_id == "test_123"
        assert metadata.progress == 0.0
        assert metadata.status == "running"
    
    def test_to_dict(self):
        """Test conversion to_dict."""
        from utils.checkpoint import CheckpointMetadata
        
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            created_at="2024-01-01",
            operation_type="sweep",
        )
        
        d = metadata.to_dict()
        
        assert "checkpoint_id" in d
        assert "operation_type" in d
    
    def test_from_dict(self):
        """Test création depuis dict."""
        from utils.checkpoint import CheckpointMetadata
        
        data = {
            "checkpoint_id": "test",
            "created_at": "2024-01-01",
            "operation_type": "sweep",
            "progress": 0.5,
            "total_items": 100,
            "completed_items": 50,
            "status": "running",
            "last_updated": None,
            "error_message": None,
        }
        
        metadata = CheckpointMetadata.from_dict(data)
        
        assert metadata.checkpoint_id == "test"
        assert metadata.progress == 0.5


class TestCheckpoint:
    """Tests pour Checkpoint."""
    
    def test_create_checkpoint(self):
        """Test création checkpoint."""
        from utils.checkpoint import Checkpoint, CheckpointMetadata
        
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            created_at="2024-01-01",
            operation_type="sweep",
            total_items=10,
        )
        
        checkpoint = Checkpoint(metadata=metadata)
        
        assert checkpoint.metadata.checkpoint_id == "test"
        assert checkpoint.results == []
    
    def test_add_result(self):
        """Test ajout de résultat."""
        from utils.checkpoint import Checkpoint, CheckpointMetadata
        
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            created_at="2024-01-01",
            operation_type="sweep",
            total_items=10,
        )
        
        checkpoint = Checkpoint(metadata=metadata)
        
        checkpoint.add_result({"params": {"x": 1}, "score": 0.5})
        checkpoint.add_result({"params": {"x": 2}, "score": 0.7})
        
        assert len(checkpoint.results) == 2
        assert checkpoint.metadata.completed_items == 2
        assert checkpoint.metadata.progress == 0.2
    
    def test_to_and_from_dict(self):
        """Test serialization."""
        from utils.checkpoint import Checkpoint, CheckpointMetadata
        
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            created_at="2024-01-01",
            operation_type="sweep",
            total_items=10,
        )
        
        checkpoint = Checkpoint(
            metadata=metadata,
            state={"current_index": 5},
            results=[{"score": 1.0}],
        )
        
        # Round trip
        d = checkpoint.to_dict()
        restored = Checkpoint.from_dict(d)
        
        assert restored.metadata.checkpoint_id == "test"
        assert restored.state["current_index"] == 5
        assert len(restored.results) == 1


class TestCheckpointManager:
    """Tests pour CheckpointManager."""
    
    def test_create_checkpoint(self, temp_checkpoint_dir):
        """Test création de checkpoint."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=100)
        
        assert checkpoint.metadata.operation_type == "sweep"
        assert checkpoint.metadata.total_items == 100
        assert checkpoint.metadata.status == "running"
    
    def test_save_and_load(self, temp_checkpoint_dir):
        """Test sauvegarde et chargement."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=50)
        checkpoint.add_result({"score": 1.0})
        checkpoint.add_result({"score": 2.0})
        manager.save(checkpoint)
        
        # Charger
        loaded = manager.load(checkpoint.metadata.checkpoint_id)
        
        assert loaded is not None
        assert len(loaded.results) == 2
        assert loaded.metadata.completed_items == 2
    
    def test_find_latest(self, temp_checkpoint_dir):
        """Test find_latest."""
        from utils.checkpoint import CheckpointManager
        import time
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Créer plusieurs checkpoints
        cp1 = manager.create("sweep", total_items=10)
        time.sleep(0.1)
        cp2 = manager.create("sweep", total_items=20)
        
        latest = manager.find_latest("sweep")
        
        assert latest is not None
        assert latest.metadata.checkpoint_id == cp2.metadata.checkpoint_id
    
    def test_find_incomplete(self, temp_checkpoint_dir):
        """Test find_incomplete."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Créer un checkpoint complété
        cp1 = manager.create("sweep", total_items=10)
        manager.complete(cp1)
        
        # Créer un checkpoint en cours
        cp2 = manager.create("sweep", total_items=20)
        
        incomplete = manager.find_incomplete("sweep")
        
        assert incomplete is not None
        assert incomplete.metadata.checkpoint_id == cp2.metadata.checkpoint_id
    
    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test list_checkpoints."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        manager.create("sweep", total_items=10)
        manager.create("sweep", total_items=20)
        manager.create("validation", total_items=5)
        
        all_checkpoints = manager.list_checkpoints()
        sweep_checkpoints = manager.list_checkpoints("sweep")
        
        assert len(all_checkpoints) == 3
        assert len(sweep_checkpoints) == 2
    
    def test_complete(self, temp_checkpoint_dir):
        """Test marquer comme terminé."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=10)
        manager.complete(checkpoint)
        
        assert checkpoint.metadata.status == "completed"
        assert checkpoint.metadata.progress == 1.0
    
    def test_fail(self, temp_checkpoint_dir):
        """Test marquer comme échoué."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=10)
        manager.fail(checkpoint, "Test error")
        
        assert checkpoint.metadata.status == "failed"
        assert checkpoint.metadata.error_message == "Test error"
    
    def test_pause(self, temp_checkpoint_dir):
        """Test mise en pause."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=10)
        manager.pause(checkpoint)
        
        assert checkpoint.metadata.status == "paused"
    
    def test_delete(self, temp_checkpoint_dir):
        """Test suppression."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint = manager.create("sweep", total_items=10)
        checkpoint_id = checkpoint.metadata.checkpoint_id
        
        manager.delete(checkpoint_id)
        
        loaded = manager.load(checkpoint_id)
        assert loaded is None
    
    def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test nettoyage automatique."""
        from utils.checkpoint import CheckpointManager
        import time
        
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=2)
        
        # Créer plus que le max
        for i in range(4):
            cp = manager.create("sweep", total_items=10)
            manager.complete(cp)
            time.sleep(0.05)
        
        # Devrait avoir gardé seulement 2 + les incomplets
        checkpoints = manager.list_checkpoints("sweep")
        completed = [c for c in checkpoints if c.metadata.status == "completed"]
        
        assert len(completed) <= 2
    
    def test_context_manager_success(self, temp_checkpoint_dir):
        """Test context manager - succès."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        with manager.checkpoint_context("sweep", total_items=5, resume=False) as (cp, start):
            assert start == 0
            for i in range(5):
                cp.add_result({"i": i})
        
        assert cp.metadata.status == "completed"
    
    def test_context_manager_resume(self, temp_checkpoint_dir):
        """Test context manager - reprise."""
        from utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Premier run - simuler interruption
        checkpoint = manager.create("sweep", total_items=10)
        for i in range(5):
            checkpoint.add_result({"i": i})
        manager.pause(checkpoint)
        
        # Deuxième run - reprise
        with manager.checkpoint_context("sweep", total_items=10, resume=True) as (cp, start):
            assert start == 5  # Reprend où on s'est arrêté
            for i in range(start, 10):
                cp.add_result({"i": i})
        
        assert cp.metadata.status == "completed"
        assert len(cp.results) == 10


class TestCheckpointManagerSingleton:
    """Tests pour le singleton."""
    
    def test_get_checkpoint_manager(self, temp_checkpoint_dir):
        """Test get_checkpoint_manager."""
        from utils.checkpoint import get_checkpoint_manager, _default_manager
        import utils.checkpoint
        
        # Reset singleton
        utils.checkpoint._default_manager = None
        
        manager = get_checkpoint_manager(temp_checkpoint_dir)
        
        assert manager is not None
        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
