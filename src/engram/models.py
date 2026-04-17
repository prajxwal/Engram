"""Core data models for Engram memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    """Classification of memory content.
    
    Determines contradiction resolution strategy and decay behavior.
    - fact: objective information ("Luna is 4 years old")
    - preference: subjective preference ("I prefer dark mode")
    - instruction: standing instructions ("Always respond in bullet points")
    - context: situational context ("Working on a React project")
    """
    FACT = "fact"
    PREFERENCE = "preference"
    INSTRUCTION = "instruction"
    CONTEXT = "context"


class ConflictResolution(str, Enum):
    """How a contradiction between two memories was resolved."""
    RECENCY_WINS = "recency_wins"       # New memory replaces old (preference/instruction)
    FLAGGED = "flagged"                 # Both kept, conflict marker set (fact)
    KEPT_BOTH = "kept_both"             # Both kept, no conflict (context)
    USER_RESOLVED = "user_resolved"     # User manually resolved
    DEFERRED = "deferred"               # LLM unavailable, resolve later


class ConflictVerdict(str, Enum):
    """LLM's assessment of two memories' relationship."""
    CONTRADICTS = "contradicts"
    UPDATES = "updates"
    UNRELATED = "unrelated"
    COMPLEMENTS = "complements"


@dataclass
class Memory:
    """A single atomic memory unit.
    
    Represents a fact, preference, instruction, or context extracted
    from conversation and stored for cross-session retrieval.
    """
    content: str
    type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    
    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Usage tracking (feeds into decay function)
    access_count: int = 0
    
    # Session provenance
    source_session: Optional[str] = None
    
    # State flags
    pinned: bool = False
    archived: bool = False
    conflict_candidate: bool = False
    
    # Embedding model that created the vector (for migration detection)
    embedding_model: Optional[str] = None
    
    def touch(self) -> None:
        """Update access tracking — called on every retrieval."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/export."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "source_session": self.source_session,
            "pinned": self.pinned,
            "archived": self.archived,
            "conflict_candidate": self.conflict_candidate,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        """Deserialize from dictionary."""
        data = data.copy()
        data["type"] = MemoryType(data["type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        # Drop any extra keys that aren't Memory fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)


@dataclass
class ConflictRecord:
    """Audit trail entry for a contradiction resolution.
    
    Written to ~/.engram/conflicts.jsonl for provenance tracking.
    """
    old_memory_id: str
    old_memory_content: str
    new_memory_id: str
    new_memory_content: str
    verdict: ConflictVerdict
    resolution: ConflictResolution
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSONL logging."""
        return {
            "old_memory_id": self.old_memory_id,
            "old_memory_content": self.old_memory_content,
            "new_memory_id": self.new_memory_id,
            "new_memory_content": self.new_memory_content,
            "verdict": self.verdict.value,
            "resolution": self.resolution.value,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }
