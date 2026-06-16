"""Tests for DecayEngine — relevance scoring and pruning."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from engram.decay import DecayEngine
from engram.models import Memory, MemoryType


class TestRelevanceCalculation:
    """Tests for the decay formula."""

    def test_fresh_memory_high_relevance(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        memory = Memory(content="recent fact", importance=0.8)

        relevance = engine.calculate_relevance(memory)
        # Fresh memory: recency ≈ 1.0, frequency_boost = 1.0
        # relevance ≈ 0.8 × 1.0 × 1.0 = 0.8
        assert relevance > 0.7

    def test_old_memory_decays(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        memory = Memory(content="old fact", importance=0.5)
        memory.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)

        relevance = engine.calculate_relevance(memory)
        # After 30 days, relevance should be much lower
        assert relevance < 0.1

    def test_pinned_memory_no_decay(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        memory = Memory(content="pinned", importance=0.7, pinned=True)
        memory.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)

        relevance = engine.calculate_relevance(memory)
        # Pinned memories return importance directly
        assert relevance == 0.7

    def test_instruction_type_no_decay(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        memory = Memory(
            content="Always use bullet points",
            type=MemoryType.INSTRUCTION,
            importance=0.6,
        )
        memory.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)

        relevance = engine.calculate_relevance(memory)
        assert relevance == 0.6

    def test_high_importance_slow_decay(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        now = datetime.now(timezone.utc)

        # Two memories — same age, different importance
        high_imp = Memory(content="critical", importance=0.95)
        high_imp.last_accessed = now - timedelta(days=7)

        normal_imp = Memory(content="normal", importance=0.5)
        normal_imp.last_accessed = now - timedelta(days=7)

        high_rel = engine.calculate_relevance(high_imp, now)
        normal_rel = engine.calculate_relevance(normal_imp, now)

        # High importance should decay much slower (0.1× rate)
        # So high_rel should be proportionally much higher than normal_rel
        ratio = high_rel / normal_rel if normal_rel > 0 else float('inf')
        assert ratio > 3  # Much more than the 0.95/0.5 = 1.9× from importance alone

    def test_frequency_boost(self, store, tmp_config):
        engine = DecayEngine(store, tmp_config)
        now = datetime.now(timezone.utc)

        rarely_used = Memory(content="rare", importance=0.5, access_count=0)
        frequently_used = Memory(content="frequent", importance=0.5, access_count=20)

        rare_rel = engine.calculate_relevance(rarely_used, now)
        freq_rel = engine.calculate_relevance(frequently_used, now)

        assert freq_rel > rare_rel


class TestPruning:
    """Tests for the pruning cycle."""

    def test_prune_low_relevance_memories(self, store, tmp_config, mock_embeddings):
        engine = DecayEngine(store, tmp_config)

        # Add an old, low-importance memory
        old_memory = Memory(content="forgettable", importance=0.1)
        old_memory.last_accessed = datetime.now(timezone.utc) - timedelta(days=60)
        embedding = mock_embeddings.embed(old_memory.content)
        store.add(old_memory, embedding)

        # Add a fresh, high-importance memory
        new_memory = Memory(content="important", importance=0.9)
        embedding2 = mock_embeddings.embed(new_memory.content)
        store.add(new_memory, embedding2)

        result = engine.run_pruning()
        assert result["archived"] >= 1
        assert result["remaining"] >= 1

    def test_prune_skips_pinned(self, store, tmp_config, mock_embeddings):
        engine = DecayEngine(store, tmp_config)

        pinned = Memory(content="pinned old", importance=0.1, pinned=True)
        pinned.last_accessed = datetime.now(timezone.utc) - timedelta(days=60)
        embedding = mock_embeddings.embed(pinned.content)
        store.add(pinned, embedding)

        result = engine.run_pruning()
        assert result["archived"] == 0

    def test_prune_skips_instructions(self, store, tmp_config, mock_embeddings):
        engine = DecayEngine(store, tmp_config)

        inst = Memory(
            content="Always be concise",
            type=MemoryType.INSTRUCTION,
            importance=0.1,
        )
        inst.last_accessed = datetime.now(timezone.utc) - timedelta(days=60)
        embedding = mock_embeddings.embed(inst.content)
        store.add(inst, embedding)

        result = engine.run_pruning()
        assert result["archived"] == 0


class TestRelevanceScoring:
    """Tests for get_all_relevance_scores."""

    def test_scores_sorted_descending(self, store, tmp_config, mock_embeddings):
        engine = DecayEngine(store, tmp_config)

        for i in range(5):
            m = Memory(content=f"memory {i}", importance=0.1 * (i + 1))
            store.add(m, mock_embeddings.embed(m.content))

        scored = engine.get_all_relevance_scores()
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)
