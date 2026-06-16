"""Tests for MemoryClient — the main public API."""

import pytest

from engram.models import Memory, MemoryType


class TestMemoryInjection:
    """Tests for memory context injection into messages."""

    def test_no_memory_no_injection(self):
        from engram.client import MemoryClient
        messages = [{"role": "user", "content": "hello"}]
        result = MemoryClient._inject_memories(messages, None)
        assert result == messages

    def test_inject_creates_system_message(self):
        from engram.client import MemoryClient
        messages = [{"role": "user", "content": "hello"}]
        result = MemoryClient._inject_memories(messages, "MEMORY: User is Alex")

        assert result[0]["role"] == "system"
        assert "MEMORY: User is Alex" in result[0]["content"]
        assert result[1] == messages[0]

    def test_inject_appends_to_existing_system(self):
        from engram.client import MemoryClient
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        result = MemoryClient._inject_memories(messages, "MEMORY: User is Alex")

        assert len(result) == 2
        assert "You are helpful." in result[0]["content"]
        assert "MEMORY: User is Alex" in result[0]["content"]


class TestGetLastUserMessage:
    """Tests for extracting the last user message."""

    def test_single_message(self):
        from engram.client import MemoryClient
        messages = [{"role": "user", "content": "hello"}]
        assert MemoryClient._get_last_user_message(messages) == "hello"

    def test_multi_turn(self):
        from engram.client import MemoryClient
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        assert MemoryClient._get_last_user_message(messages) == "second"

    def test_no_user_messages(self):
        from engram.client import MemoryClient
        messages = [{"role": "system", "content": "system prompt"}]
        assert MemoryClient._get_last_user_message(messages) is None


class TestClientManualOperations:
    """Tests for add, search, pin, forget, list, stats."""

    def test_add_and_list(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        memory = client.add("User likes Python", type="preference", importance=0.7)
        assert memory.content == "User likes Python"
        assert memory.type == MemoryType.PREFERENCE

        memories = client.list()
        assert len(memories) == 1
        assert memories[0].id == memory.id

    def test_pin_and_unpin(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        memory = client.add("pinnable fact")
        assert not memory.pinned

        assert client.pin(memory.id) is True
        assert store.get(memory.id).pinned is True

        assert client.unpin(memory.id) is True
        assert store.get(memory.id).pinned is False

    def test_forget(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        memory = client.add("forgettable")
        assert client.forget(memory.id) is True
        assert store.get(memory.id) is None

    def test_stats(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        client.add("fact 1")
        client.add("fact 2")

        s = client.stats()
        assert s["active_memories"] == 2
        assert s["total_memories"] == 2


class TestEmbeddingModelMigration:
    """Tests for embedding model change detection."""

    def test_first_run_stores_model(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        client._check_embedding_model()
        assert store.get_meta("embedding_model") == "mock:test-64d"

    def test_model_change_updates_meta(self, store, mock_embeddings, tmp_config):
        from engram.client import MemoryClient

        store.set_meta("embedding_model", "old-model")

        client = MemoryClient.__new__(MemoryClient)
        client.config = tmp_config
        client._store = store
        client._embedding_engine = mock_embeddings
        client._llm_client = None
        client._retriever = None
        client._extractor = None
        client._conflict_resolver = None
        client._session_id = "test"
        client._interaction_count = 0

        client._check_embedding_model()
        assert store.get_meta("embedding_model") == "mock:test-64d"
