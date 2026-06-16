"""Tests for MemoryRetriever — the retrieval pipeline."""

import pytest

from engram.models import Memory, MemoryType
from engram.retriever import MemoryRetriever, DEFAULT_INJECTION_TEMPLATE


class TestRetrieval:
    """Tests for the full retrieval pipeline."""

    def test_retrieve_returns_none_empty_store(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)
        result = retriever.retrieve("hello", model_context_window=4096)
        assert result is None

    def test_retrieve_returns_injection_block(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        # Populate store
        m = Memory(content="The user's name is Alex", importance=0.8)
        store.add(m, mock_embeddings.embed(m.content))

        result = retriever.retrieve("What is my name?", model_context_window=4096)
        assert result is not None
        assert "RELEVANT MEMORY" in result
        assert "Alex" in result

    def test_retrieve_updates_access_count(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        m = Memory(content="testable fact", importance=0.8)
        store.add(m, mock_embeddings.embed(m.content))
        assert m.access_count == 0

        retriever.retrieve("testable fact", model_context_window=4096)

        refreshed = store.get(m.id)
        assert refreshed.access_count == 1

    def test_retrieve_raw(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        m = Memory(content="raw search test", importance=0.5)
        store.add(m, mock_embeddings.embed(m.content))

        results = retriever.retrieve_raw("raw search test")
        assert len(results) > 0
        memory, score = results[0]
        assert isinstance(memory, Memory)
        assert isinstance(score, float)


class TestTokenBudget:
    """Tests for token budget enforcement."""

    def test_token_budget_limits_memories(self, store, mock_embeddings, tmp_config):
        # Set a very small context window
        tmp_config.context_budget_ratio = 0.1  # 10% of 200 = 20 tokens
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        # Add many memories
        for i in range(20):
            m = Memory(
                content=f"Memory number {i} with some extra content to use tokens " * 3,
                importance=0.9,
            )
            store.add(m, mock_embeddings.embed(m.content))

        result = retriever.retrieve("memory", model_context_window=200)
        # With 20 token budget, we can fit very few memories
        # The point is it shouldn't include all 20
        if result is not None:
            lines = [l for l in result.split("\n") if l.strip().startswith("- ")]
            assert len(lines) < 20

    def test_estimate_tokens(self):
        assert MemoryRetriever._estimate_tokens("hello world") >= 1
        assert MemoryRetriever._estimate_tokens("a" * 100) == 25  # 100/4


class TestRelevanceScoring:
    """Tests for the decay-adjusted relevance computation."""

    def test_pinned_memory_no_decay(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)
        from datetime import datetime, timedelta, timezone

        m = Memory(content="pinned memory", importance=0.7, pinned=True)
        m.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)

        now = datetime.now(timezone.utc)
        relevance = retriever._compute_relevance(m, 0.9, now)
        # Pinned → decay_rate = 0, so recency = 1.0
        # relevance = 0.9 × 0.7 × 1.0 × 1.0 = 0.63
        assert relevance > 0.5


class TestInjectionFormatting:
    """Tests for memory injection block formatting."""

    def test_format_includes_all_selected(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        selected = [
            (Memory(content="Fact A"), 0.9, 0.8),
            (Memory(content="Fact B"), 0.85, 0.7),
        ]

        result = retriever._format_injection(selected)
        assert "Fact A" in result
        assert "Fact B" in result
        assert "2 items" in result

    def test_format_includes_type_and_confidence(self, store, mock_embeddings, tmp_config):
        retriever = MemoryRetriever(store, mock_embeddings, tmp_config)

        selected = [(Memory(content="X", type=MemoryType.PREFERENCE), 0.9, 0.75)]
        result = retriever._format_injection(selected)
        assert "preference" in result
        assert "0.75" in result
