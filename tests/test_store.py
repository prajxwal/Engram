"""Tests for MemoryStore — dual-backend persistence."""

import json

import pytest

from engram.models import ConflictRecord, ConflictResolution, ConflictVerdict, Memory, MemoryType


class TestMemoryStoreLifecycle:
    """Tests for initialization and cleanup."""

    def test_initialize_creates_dirs(self, store, tmp_config):
        import os
        assert os.path.exists(tmp_config.data_dir)
        assert os.path.exists(tmp_config.chroma_path)

    def test_double_init_is_safe(self, store):
        store.initialize()  # Should not raise


class TestMemoryStoreCRUD:
    """Tests for add, get, update, delete operations."""

    def test_add_and_get(self, store, sample_memory, mock_embeddings):
        memory = sample_memory(content="User's name is Alex")
        embedding = mock_embeddings.embed(memory.content)

        result = store.add(memory, embedding)
        assert result.id == memory.id

        retrieved = store.get(memory.id)
        assert retrieved is not None
        assert retrieved.content == "User's name is Alex"
        assert retrieved.type == MemoryType.FACT

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_update_metadata(self, store, sample_memory, mock_embeddings):
        memory = sample_memory()
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)

        memory.importance = 0.9
        memory.pinned = True
        store.update(memory)

        retrieved = store.get(memory.id)
        assert retrieved.importance == 0.9
        assert retrieved.pinned is True

    def test_update_with_embedding(self, store, sample_memory, mock_embeddings):
        memory = sample_memory(content="old content")
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)

        memory.content = "new content"
        new_embedding = mock_embeddings.embed(memory.content)
        store.update(memory, embedding=new_embedding)

        retrieved = store.get(memory.id)
        assert retrieved.content == "new content"

    def test_delete(self, store, sample_memory, mock_embeddings):
        memory = sample_memory()
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)

        assert store.delete(memory.id) is True
        assert store.get(memory.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False


class TestMemoryStoreSearch:
    """Tests for semantic search."""

    def test_search_returns_results(self, store, sample_memory, mock_embeddings):
        m1 = sample_memory(content="User likes Python programming")
        m2 = sample_memory(content="User prefers dark mode")
        m3 = sample_memory(content="The cat's name is Luna")

        for m in [m1, m2, m3]:
            store.add(m, mock_embeddings.embed(m.content))

        query_embedding = mock_embeddings.embed("Python coding")
        results = store.search(query_embedding, top_k=3)

        assert len(results) > 0
        # Results should be (Memory, similarity_score) tuples
        for memory, score in results:
            assert isinstance(memory, Memory)
            assert isinstance(score, float)

    def test_search_empty_store(self, store, mock_embeddings):
        query_embedding = mock_embeddings.embed("anything")
        results = store.search(query_embedding)
        assert results == []

    def test_search_excludes_archived(self, store, sample_memory, mock_embeddings):
        memory = sample_memory(content="archived fact")
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)
        store.archive(memory.id)

        query_embedding = mock_embeddings.embed("archived fact")
        results = store.search(query_embedding, include_archived=False)
        ids = [m.id for m, _ in results]
        assert memory.id not in ids


class TestMemoryStoreArchive:
    """Tests for archive and restore."""

    def test_archive(self, store, sample_memory, mock_embeddings):
        memory = sample_memory()
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)

        assert store.archive(memory.id) is True
        retrieved = store.get(memory.id)
        assert retrieved.archived is True

    def test_archive_nonexistent(self, store):
        assert store.archive("nonexistent") is False

    def test_restore(self, store, sample_memory, mock_embeddings):
        memory = sample_memory()
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)
        store.archive(memory.id)

        assert store.restore(memory.id, embedding) is True
        retrieved = store.get(memory.id)
        assert retrieved.archived is False


class TestMemoryStoreStats:
    """Tests for stats and counting."""

    def test_count(self, store, sample_memory, mock_embeddings):
        assert store.count() == 0

        m1 = sample_memory(content="fact 1")
        store.add(m1, mock_embeddings.embed(m1.content))
        assert store.count() == 1

    def test_count_excludes_archived(self, store, sample_memory, mock_embeddings):
        memory = sample_memory()
        embedding = mock_embeddings.embed(memory.content)
        store.add(memory, embedding)
        store.archive(memory.id)

        assert store.count(include_archived=False) == 0
        assert store.count(include_archived=True) == 1

    def test_stats(self, store, sample_memory, mock_embeddings):
        m1 = sample_memory(content="fact 1", importance=0.8)
        m2 = sample_memory(content="fact 2", importance=0.6)
        m2.pinned = True
        store.add(m1, mock_embeddings.embed(m1.content))
        store.add(m2, mock_embeddings.embed(m2.content))

        s = store.stats()
        assert s["active_memories"] == 2
        assert s["pinned_memories"] == 1
        assert s["total_memories"] == 2


class TestMemoryStoreListMemories:
    """Tests for list_memories with filters."""

    def test_list_all(self, store, sample_memory, mock_embeddings):
        for i in range(3):
            m = sample_memory(content=f"memory {i}")
            store.add(m, mock_embeddings.embed(m.content))
        assert len(store.list_memories()) == 3

    def test_list_by_type(self, store, sample_memory, mock_embeddings):
        m1 = sample_memory(content="a fact", memory_type=MemoryType.FACT)
        m2 = sample_memory(content="a preference", memory_type=MemoryType.PREFERENCE)
        store.add(m1, mock_embeddings.embed(m1.content))
        store.add(m2, mock_embeddings.embed(m2.content))

        facts = store.list_memories(memory_type=MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].type == MemoryType.FACT

    def test_list_pinned_only(self, store, sample_memory, mock_embeddings):
        m1 = sample_memory(content="pinned", pinned=True)
        m2 = sample_memory(content="not pinned", pinned=False)
        store.add(m1, mock_embeddings.embed(m1.content))
        store.add(m2, mock_embeddings.embed(m2.content))

        pinned = store.list_memories(pinned_only=True)
        assert len(pinned) == 1
        assert pinned[0].pinned is True


class TestMemoryStoreConflictLog:
    """Tests for conflict audit log."""

    def test_log_and_list_conflicts(self, store):
        record = ConflictRecord(
            old_memory_id="old1",
            old_memory_content="Luna is 3",
            new_memory_id="new1",
            new_memory_content="Luna is 4",
            verdict=ConflictVerdict.UPDATES,
            resolution=ConflictResolution.RECENCY_WINS,
            memory_type=MemoryType.FACT,
        )
        store.log_conflict(record)

        conflicts = store.list_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["old_memory_content"] == "Luna is 3"

    def test_list_conflicts_empty(self, store):
        assert store.list_conflicts() == []


class TestMemoryStoreMeta:
    """Tests for key-value metadata store."""

    def test_set_and_get_meta(self, store):
        store.set_meta("embedding_model", "test-model")
        assert store.get_meta("embedding_model") == "test-model"

    def test_get_nonexistent_meta(self, store):
        assert store.get_meta("nonexistent") is None


class TestMemoryStoreExportImport:
    """Tests for export/import."""

    def test_export_and_import(self, store, sample_memory, mock_embeddings, tmp_path):
        m1 = sample_memory(content="exportable fact")
        store.add(m1, mock_embeddings.embed(m1.content))

        export_path = str(tmp_path / "export.json")
        count = store.export_memories(export_path)
        assert count == 1

        # Verify file contents
        with open(export_path) as f:
            data = json.load(f)
        assert len(data["memories"]) == 1

        # Import into fresh store
        from engram.config import EngramConfig
        from engram.store import MemoryStore

        config2 = EngramConfig(data_dir=str(tmp_path / "engram_test2"))
        store2 = MemoryStore(config2)
        store2.initialize()

        imported = store2.import_memories(export_path, mock_embeddings.embed)
        assert imported == 1
        assert store2.count() == 1

        store2.close()
