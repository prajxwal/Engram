"""Tests for MemoryExtractor — extraction, parsing, and deduplication."""

import pytest

from engram.extractor import MemoryExtractor, REMEMBER_TRIGGERS
from engram.models import MemoryType


class TestFourTierParser:
    """Tests for the 4-tier fallback parser."""

    def test_tier1_direct_json(self):
        raw = '[{"content": "User likes Python", "type": "preference", "importance": 0.7}]'
        result = MemoryExtractor._parse_extraction(raw)
        assert len(result) == 1
        assert result[0]["content"] == "User likes Python"

    def test_tier2_markdown_code_block(self):
        raw = """Here are the memories:
```json
[{"content": "User's name is Alex", "type": "fact", "importance": 0.8}]
```"""
        result = MemoryExtractor._parse_extraction(raw)
        assert len(result) == 1
        assert result[0]["content"] == "User's name is Alex"

    def test_tier3_array_in_text(self):
        raw = """I found these memories:
Some preamble text [{"content": "Cat named Luna", "type": "fact", "importance": 0.6}] and more text."""
        result = MemoryExtractor._parse_extraction(raw)
        assert len(result) == 1
        assert result[0]["content"] == "Cat named Luna"

    def test_tier4_bullet_points(self):
        raw = """Here are the key facts:
- User prefers dark mode
- User works on Python projects
- User's cat is named Luna"""
        result = MemoryExtractor._parse_extraction(raw)
        assert len(result) == 3

    def test_empty_response(self):
        assert MemoryExtractor._parse_extraction("") == []
        assert MemoryExtractor._parse_extraction("   ") == []

    def test_empty_array(self):
        assert MemoryExtractor._parse_extraction("[]") == []

    def test_filters_entries_without_content(self):
        raw = '[{"type": "fact"}, {"content": "valid", "type": "fact"}]'
        result = MemoryExtractor._parse_extraction(raw)
        assert len(result) == 1
        assert result[0]["content"] == "valid"


class TestExplicitSignals:
    """Tests for explicit memory detection ('remember that...')."""

    def test_remember_that_trigger(self):
        result = MemoryExtractor._detect_explicit_signals(
            "Remember that my birthday is March 15th"
        )
        assert len(result) == 1
        assert "March 15th" in result[0]["content"]
        assert result[0]["importance"] == 0.95
        assert result[0]["pinned"] is True

    def test_dont_forget_trigger(self):
        result = MemoryExtractor._detect_explicit_signals(
            "Don't forget I'm allergic to peanuts"
        )
        assert len(result) == 1
        assert "allergic" in result[0]["content"]

    def test_no_trigger(self):
        result = MemoryExtractor._detect_explicit_signals(
            "What's the weather like?"
        )
        assert result == []

    def test_trigger_too_short_content(self):
        result = MemoryExtractor._detect_explicit_signals("Remember that ok")
        # "ok" is only 2 chars, below the 3-char threshold
        assert result == []


class TestDeduplication:
    """Tests for substring-based deduplication."""

    def test_removes_substrings(self):
        memories = [
            {"content": "User likes Python programming"},
            {"content": "User likes Python"},  # substring of first
        ]
        result = MemoryExtractor._deduplicate(memories)
        assert len(result) == 1

    def test_keeps_unrelated(self):
        memories = [
            {"content": "User likes Python"},
            {"content": "Cat named Luna"},
        ]
        result = MemoryExtractor._deduplicate(memories)
        assert len(result) == 2


class TestSyncExtraction:
    """Tests for the full synchronous extraction flow."""

    def test_extract_stores_memories(self, store, mock_embeddings, mock_llm, tmp_config):
        mock_llm.next_response = '[{"content": "User likes tea", "type": "preference", "importance": 0.6}]'

        extractor = MemoryExtractor(
            store=store,
            embedding_engine=mock_embeddings,
            llm_client=mock_llm,
            config=tmp_config,
        )

        stored = extractor.extract_sync(
            user_message="I really love drinking tea",
            assistant_response="Tea is great!",
            model="test-model",
            session_id="test-session",
        )

        assert len(stored) == 1
        assert stored[0].content == "User likes tea"
        assert stored[0].type == MemoryType.PREFERENCE
        assert store.count() == 1

    def test_extract_with_explicit_signal(self, store, mock_embeddings, mock_llm, tmp_config):
        mock_llm.next_response = "[]"  # LLM finds nothing

        extractor = MemoryExtractor(
            store=store,
            embedding_engine=mock_embeddings,
            llm_client=mock_llm,
            config=tmp_config,
        )

        stored = extractor.extract_sync(
            user_message="Remember that my dog's name is Max",
            assistant_response="Got it!",
            model="test-model",
        )

        # Should still store the explicit signal
        assert len(stored) == 1
        assert "Max" in stored[0].content
        assert stored[0].pinned is True

    def test_extract_handles_llm_failure(self, store, mock_embeddings, tmp_config):
        """Extraction should never crash — failures are swallowed."""

        class FailingLLM:
            def chat_full(self, *a, **kw):
                raise ConnectionError("LLM offline")
            def chat(self, *a, **kw):
                raise ConnectionError("LLM offline")

        extractor = MemoryExtractor(
            store=store,
            embedding_engine=mock_embeddings,
            llm_client=FailingLLM(),
            config=tmp_config,
        )

        # Should not raise
        stored = extractor.extract_sync(
            user_message="test",
            assistant_response="test",
            model="test-model",
        )
        assert stored == []
