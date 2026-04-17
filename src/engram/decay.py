"""Decay engine — principled memory forgetting with semantic weighting.

Implements:
    relevance(m, t) = base_importance × exp(-λ × Δhours) × (1 + log₂(1 + access_count))
    
Pinned and instruction-type memories are exempt.
High-importance memories (>0.9) decay at 0.1× rate.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from engram.config import EngramConfig
from engram.models import Memory, MemoryType
from engram.store import MemoryStore

logger = logging.getLogger(__name__)


class DecayEngine:
    """Manages memory decay, pruning, and memory count limits.
    
    Called on startup and every N interactions (configurable).
    Memories below the pruning threshold are archived (not deleted).
    """

    def __init__(self, store: MemoryStore, config: EngramConfig):
        self.store = store
        self.config = config

    def calculate_relevance(
        self,
        memory: Memory,
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate the current relevance score for a memory.
        
        Formula:
            relevance = base_importance × recency × frequency_boost
            
        Where:
            recency       = exp(-λ × hours_since_last_access)
            frequency     = 1 + log₂(1 + access_count)
            
        Modifiers:
            - Pinned memories: relevance is always 1.0
            - Instruction type: exempt from decay (treated as pinned)
            - importance > 0.9: decay rate reduced to 0.1×
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Exempt memories
        if memory.pinned or memory.type == MemoryType.INSTRUCTION:
            return memory.importance  # No decay

        # Time since last access
        delta_hours = (now - memory.last_accessed).total_seconds() / 3600.0
        delta_hours = max(delta_hours, 0.0)

        # Effective decay rate
        decay_rate = self.config.decay_rate
        if memory.importance > 0.9:
            decay_rate *= 0.1  # Core memories decay 10× slower

        # Decay components
        recency = math.exp(-decay_rate * delta_hours)
        frequency_boost = 1.0 + math.log2(1.0 + memory.access_count)
        base_importance = memory.importance

        return base_importance * recency * frequency_boost

    def run_pruning(self) -> dict:
        """Run the pruning cycle: archive memories below threshold.
        
        Also enforces memory count limits:
        - Soft limit: trigger aggressive pruning (higher threshold)
        - Hard limit: permanently delete oldest archived
        
        Returns:
            Stats dict: {archived: int, deleted: int, remaining: int}
        """
        now = datetime.now(timezone.utc)
        memories = self.store.list_memories(include_archived=False)
        
        archived_count = 0
        deleted_count = 0

        # Determine pruning threshold
        active_count = len(memories)
        threshold = self.config.prune_threshold

        if active_count > self.config.memory_soft_limit:
            # Aggressive pruning: raise threshold
            threshold = 0.15
            logger.info(
                f"Memory soft limit exceeded ({active_count}/{self.config.memory_soft_limit}). "
                f"Aggressive pruning with threshold={threshold}"
            )

        # Score and prune
        for memory in memories:
            if memory.pinned or memory.type == MemoryType.INSTRUCTION:
                continue  # Never prune exempt memories

            relevance = self.calculate_relevance(memory, now)

            if relevance < threshold:
                self.store.archive(memory.id)
                archived_count += 1
                logger.debug(
                    f"Archived memory {memory.id} "
                    f"(relevance={relevance:.4f} < {threshold}): "
                    f"{memory.content[:50]}"
                )

        # Enforce hard limit on total memories
        total = self.store.count(include_archived=True)
        if total > self.config.memory_hard_limit:
            deleted_count = self._enforce_hard_limit(total)

        remaining = self.store.count(include_archived=False)

        if archived_count > 0 or deleted_count > 0:
            logger.info(
                f"Pruning complete: archived={archived_count}, "
                f"deleted={deleted_count}, remaining={remaining}"
            )

        return {
            "archived": archived_count,
            "deleted": deleted_count,
            "remaining": remaining,
        }

    def _enforce_hard_limit(self, total: int) -> int:
        """Permanently delete oldest archived memories to stay under hard limit.
        
        Returns:
            Number of memories deleted.
        """
        excess = total - self.config.memory_hard_limit
        if excess <= 0:
            return 0

        # Get archived memories sorted by creation date (oldest first)
        archived = self.store.list_memories(include_archived=True)
        archived = [m for m in archived if m.archived]
        archived.sort(key=lambda m: m.created_at)

        deleted = 0
        for memory in archived[:excess]:
            self.store.delete(memory.id)
            deleted += 1
            logger.debug(f"Permanently deleted memory {memory.id}: {memory.content[:50]}")

        return deleted

    def get_all_relevance_scores(self) -> list[tuple[Memory, float]]:
        """Calculate relevance scores for all active memories.
        
        Useful for the `engram memories list` command.
        
        Returns:
            List of (Memory, relevance_score) tuples, sorted by score desc.
        """
        now = datetime.now(timezone.utc)
        memories = self.store.list_memories(include_archived=False)
        
        scored = [
            (memory, self.calculate_relevance(memory, now))
            for memory in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
