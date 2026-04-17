"""Contradiction detection and type-based resolution.

Detection pipeline:
1. Embed new memory → find candidates with cosine sim > threshold
2. If candidates found, LLM contradiction check
3. Apply type-dependent resolution

Resolution strategies by memory type:
- preference/instruction → recency wins, archive old
- fact → flag for review, store both, log conflict
- context → keep both (additive)
"""

from __future__ import annotations

import logging
from typing import Optional

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine
from engram.llm import LLMClient
from engram.models import (
    ConflictRecord,
    ConflictResolution,
    ConflictVerdict,
    Memory,
    MemoryType,
)
from engram.store import MemoryStore

logger = logging.getLogger(__name__)

# Contradiction check prompt
CONTRADICTION_PROMPT = """Do these two statements contradict each other?

Statement A: "{existing_memory}"
Statement B: "{new_memory}"

Choose EXACTLY ONE answer:
- CONTRADICTS: They directly conflict or cannot both be true
- UPDATES: B is a newer version of the information in A
- COMPLEMENTS: They add to each other without conflict
- UNRELATED: They are about different topics

Answer with a single word: CONTRADICTS, UPDATES, COMPLEMENTS, or UNRELATED"""


class ConflictResolver:
    """Detects and resolves contradictions between new and existing memories.
    
    Runs asynchronously after memory extraction.
    Uses LLM for nuanced contradiction detection (only on high-similarity candidates).
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        llm_client: LLMClient,
        config: EngramConfig,
    ):
        self.store = store
        self.embedding_engine = embedding_engine
        self.llm_client = llm_client
        self.config = config

    def check_and_resolve(
        self,
        new_memory: Memory,
        new_embedding: list[float],
        model: str,
    ) -> Optional[ConflictRecord]:
        """Check a new memory for contradictions with existing memories.
        
        Args:
            new_memory: The newly extracted memory.
            new_embedding: Its embedding vector.
            model: LLM model to use for contradiction checking.
            
        Returns:
            ConflictRecord if a contradiction was found and resolved, else None.
        """
        # Step 1: Find high-similarity candidates
        candidates = self.store.search(
            query_embedding=new_embedding,
            top_k=5,
        )

        # Filter to high-similarity candidates (same topic)
        conflicts = [
            (memory, score)
            for memory, score in candidates
            if score > self.config.conflict_similarity
            and memory.id != new_memory.id
        ]

        if not conflicts:
            return None

        # Step 2: LLM contradiction check (max 3 per turn)
        for existing_memory, similarity in conflicts[:3]:
            verdict = self._check_contradiction(
                existing_memory, new_memory, model
            )

            if verdict in (ConflictVerdict.CONTRADICTS, ConflictVerdict.UPDATES):
                # Step 3: Apply type-based resolution
                record = self._resolve(
                    existing_memory, new_memory, verdict
                )
                return record

        return None

    def resolve_deferred(self, model: str) -> int:
        """Resolve memories that were stored with conflict_candidate=true.
        
        Called on startup if there are deferred conflict checks.
        
        Returns:
            Number of conflicts resolved.
        """
        candidates = [
            m for m in self.store.list_memories()
            if m.conflict_candidate
        ]

        resolved = 0
        for memory in candidates:
            embedding = self.embedding_engine.embed(memory.content)
            record = self.check_and_resolve(memory, embedding, model)
            
            # Clear the flag regardless
            memory.conflict_candidate = False
            self.store.update(memory)

            if record:
                resolved += 1

        if resolved > 0:
            logger.info(f"Resolved {resolved} deferred conflicts")
        return resolved

    # ---------- LLM contradiction check ----------

    def _check_contradiction(
        self,
        existing: Memory,
        new: Memory,
        model: str,
    ) -> ConflictVerdict:
        """Ask the LLM whether two memories contradict each other."""
        prompt = CONTRADICTION_PROMPT.format(
            existing_memory=existing.content,
            new_memory=new.content,
        )

        try:
            response = self.llm_client.chat_full(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistency
            )

            # Parse the response — look for the verdict keyword
            response_upper = response.strip().upper()
            
            for verdict in ConflictVerdict:
                if verdict.value.upper() in response_upper:
                    return verdict

            # Could not parse — assume unrelated
            logger.warning(
                f"Could not parse contradiction verdict: {response[:50]}"
            )
            return ConflictVerdict.UNRELATED

        except Exception as e:
            logger.warning(f"Contradiction check LLM call failed: {e}")
            return ConflictVerdict.UNRELATED

    # ---------- Type-based resolution ----------

    def _resolve(
        self,
        existing: Memory,
        new: Memory,
        verdict: ConflictVerdict,
    ) -> ConflictRecord:
        """Apply resolution strategy based on memory type.
        
        | Type        | Strategy                                    |
        |-------------|---------------------------------------------|
        | preference  | Recency wins. Archive old memory.            |
        | instruction | Recency wins. Archive old memory.            |
        | fact        | Flag for review. Keep both. Log conflict.    |
        | context     | Keep both (context is additive).             |
        """
        # Determine effective type (use new memory's type)
        memory_type = new.type

        if memory_type in (MemoryType.PREFERENCE, MemoryType.INSTRUCTION):
            # Recency wins — archive old memory
            self.store.archive(existing.id)
            resolution = ConflictResolution.RECENCY_WINS
            reason = f"Newer {memory_type.value} replaces older one"

        elif memory_type == MemoryType.FACT:
            # Flag for review — keep both, mark conflict
            existing.conflict_candidate = True
            new.conflict_candidate = True
            self.store.update(existing)
            self.store.update(new)
            resolution = ConflictResolution.FLAGGED
            reason = "Factual contradiction — both kept for user review"

        else:  # CONTEXT
            resolution = ConflictResolution.KEPT_BOTH
            reason = "Context is additive — both kept"

        # Create audit record
        record = ConflictRecord(
            old_memory_id=existing.id,
            old_memory_content=existing.content,
            new_memory_id=new.id,
            new_memory_content=new.content,
            verdict=verdict,
            resolution=resolution,
            memory_type=memory_type,
            reason=reason,
        )

        # Log to JSONL audit trail
        self.store.log_conflict(record)

        logger.info(
            f"Conflict resolved [{resolution.value}]: "
            f"'{existing.content[:40]}' vs '{new.content[:40]}'"
        )

        return record
