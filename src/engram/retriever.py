"""Memory retrieval pipeline.

Handles: embed query → ChromaDB top-k → decay rerank → token budget → format injection.
Zero LLM calls — pure embedding similarity + decay scoring.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine
from engram.models import Memory
from engram.store import MemoryStore

logger = logging.getLogger(__name__)

# Default injection template
DEFAULT_INJECTION_TEMPLATE = """[RELEVANT MEMORY — {count} items retrieved]
{memories}
[END MEMORY]

Refer to these memories naturally. Do not mention the memory system to the user.
If memories conflict with what the user is saying now, prioritize what they're saying now."""


class MemoryRetriever:
    """Retrieves and ranks relevant memories for context injection.
    
    The retrieval pipeline:
    1. Embed the incoming user message
    2. Query ChromaDB for top-k candidates
    3. Re-rank using decay formula (importance × recency × frequency)
    4. Apply token budget filter
    5. Format as structured injection block
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        config: EngramConfig,
    ):
        self.store = store
        self.embedding_engine = embedding_engine
        self.config = config

    def retrieve(
        self,
        query: str,
        model_context_window: int = 4096,
    ) -> Optional[str]:
        """Retrieve relevant memories and format for injection.
        
        Args:
            query: The user's message to find relevant memories for.
            model_context_window: The model's total context window size.
            
        Returns:
            Formatted injection string, or None if no relevant memories found.
        """
        # Check if we have any memories at all
        if self.store.count() == 0:
            return None

        # Step 1: Embed query
        query_embedding = self.embedding_engine.embed(query)

        # Step 2: ChromaDB top-k candidates
        candidates = self.store.search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
        )

        if not candidates:
            return None

        # Step 3: Decay re-rank
        now = datetime.now(timezone.utc)
        scored = []
        for memory, similarity in candidates:
            relevance = self._compute_relevance(memory, similarity, now)
            scored.append((memory, similarity, relevance))

        # Sort by combined relevance (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        # Step 4: Select top-n and apply token budget
        max_tokens = int(model_context_window * self.config.context_budget_ratio)
        selected = self._apply_token_budget(scored, max_tokens)

        if not selected:
            return None

        # Update access tracking for selected memories
        for memory, _, _ in selected:
            memory.touch()
            self.store.update(memory)

        # Step 5: Format injection block
        return self._format_injection(selected)

    def retrieve_raw(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Memory, float]]:
        """Retrieve memories without formatting (for search command).
        
        Returns:
            List of (Memory, similarity_score) tuples.
        """
        if self.store.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed(query)
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.config.top_k,
        )

    # ---------- Decay formula ----------

    def _compute_relevance(
        self,
        memory: Memory,
        similarity: float,
        now: datetime,
    ) -> float:
        """Compute decay-adjusted relevance score.
        
        Formula:
            relevance = similarity × base_importance × recency × frequency_boost
            
        Where:
            recency       = exp(-λ × hours_since_last_access)
            frequency     = 1 + log₂(1 + access_count)
            
        Pinned memories get a fixed relevance boost.
        High-importance memories (>0.9) decay at 0.1× rate.
        """
        # Time since last access in hours
        delta_hours = (now - memory.last_accessed).total_seconds() / 3600.0
        delta_hours = max(delta_hours, 0.0)

        # Effective decay rate
        decay_rate = self.config.decay_rate
        if memory.importance > 0.9:
            decay_rate *= 0.1  # Core memories decay 10× slower
        if memory.pinned:
            decay_rate = 0.0  # Pinned memories don't decay

        # Components
        recency = math.exp(-decay_rate * delta_hours)
        frequency_boost = 1.0 + math.log2(1.0 + memory.access_count)
        base_importance = memory.importance

        # Combined score: similarity from embedding × memory quality
        relevance = similarity * base_importance * recency * frequency_boost

        return relevance

    # ---------- Token budget ----------

    def _apply_token_budget(
        self,
        scored: list[tuple[Memory, float, float]],
        max_tokens: int,
    ) -> list[tuple[Memory, float, float]]:
        """Select memories that fit within the token budget.
        
        Uses approximate token counting (chars / 4).
        Memories are already sorted by relevance descending.
        """
        selected = []
        used_tokens = 0
        overhead = 100  # Injection template chrome (headers, footers)

        for memory, similarity, relevance in scored:
            # Approximate token count for this memory
            mem_tokens = self._estimate_tokens(memory.content) + 20  # metadata overhead
            if used_tokens + mem_tokens + overhead > max_tokens:
                break
            selected.append((memory, similarity, relevance))
            used_tokens += mem_tokens
            if len(selected) >= self.config.top_n:
                break

        return selected

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count. Chars/4 is a reasonable estimate for English text."""
        return max(1, len(text) // 4)

    # ---------- Formatting ----------

    def _format_injection(
        self,
        selected: list[tuple[Memory, float, float]],
    ) -> str:
        """Format selected memories into the injection block.
        
        Uses the configured injection template or the default.
        """
        memory_lines = []
        for memory, similarity, relevance in selected:
            line = f"- {memory.content} [type: {memory.type.value}, confidence: {relevance:.2f}]"
            memory_lines.append(line)

        memories_str = "\n".join(memory_lines)

        # Use custom template if configured
        template = DEFAULT_INJECTION_TEMPLATE
        if self.config.injection_template:
            try:
                with open(self.config.injection_template) as f:
                    template = f.read()
            except OSError:
                logger.warning(
                    f"Could not read injection template: {self.config.injection_template}"
                )

        return template.format(
            count=len(selected),
            memories=memories_str,
        )
