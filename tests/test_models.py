"""Tests for core data models."""

from datetime import datetime, timezone

from engram.models import (
    ConflictRecord,
    ConflictResolution,
    ConflictVerdict,
    Memory,
    MemoryType,
)


class TestMemory:
    """Tests for the Memory dataclass."""

    def test_default_values(self):
        m = Memory(content="test fact")
        assert m.content == "test fact"
        assert m.type == MemoryType.FACT
        assert m.importance == 0.5
        assert m.access_count == 0
        assert m.pinned is False
        assert m.archived is False
        assert m.conflict_candidate is False
        assert m.id  # auto-generated

    def test_touch_updates_tracking(self):
        m = Memory(content="test")
        old_accessed = m.last_accessed
        m.touch()
        assert m.access_count == 1
        assert m.last_accessed >= old_accessed

    def test_serialization_roundtrip(self):
        m = Memory(
            content="Luna is 4 years old",
            type=MemoryType.FACT,
            importance=0.85,
            pinned=True,
        )
        d = m.to_dict()
        m2 = Memory.from_dict(d)

        assert m2.content == m.content
        assert m2.type == m.type
        assert m2.importance == m.importance
        assert m2.pinned == m.pinned
        assert m2.id == m.id

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "id": "abc123",
            "content": "test",
            "type": "fact",
            "importance": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
            "source_session": None,
            "pinned": False,
            "archived": False,
            "conflict_candidate": False,
            "embedding_model": None,
            "extra_field": "should be ignored",
        }
        m = Memory.from_dict(d)
        assert m.id == "abc123"
        assert not hasattr(m, "extra_field")


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_values(self):
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.INSTRUCTION.value == "instruction"
        assert MemoryType.CONTEXT.value == "context"

    def test_from_string(self):
        assert MemoryType("fact") == MemoryType.FACT
        assert MemoryType("preference") == MemoryType.PREFERENCE


class TestConflictRecord:
    """Tests for ConflictRecord dataclass."""

    def test_serialization(self):
        r = ConflictRecord(
            old_memory_id="old123",
            old_memory_content="Luna is 3 years old",
            new_memory_id="new456",
            new_memory_content="Luna is 4 years old",
            verdict=ConflictVerdict.UPDATES,
            resolution=ConflictResolution.RECENCY_WINS,
            memory_type=MemoryType.FACT,
            reason="Age updated",
        )
        d = r.to_dict()
        assert d["verdict"] == "updates"
        assert d["resolution"] == "recency_wins"
        assert d["memory_type"] == "fact"
        assert d["reason"] == "Age updated"
        assert "timestamp" in d
