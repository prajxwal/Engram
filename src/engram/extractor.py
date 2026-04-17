"""Async memory extraction from conversation turns.

Runs AFTER the LLM response is delivered to the user (user-perceived latency = 0).
Uses a 4-tier fallback parser to handle unreliable JSON output from 7B models.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Optional

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine
from engram.llm import LLMClient
from engram.models import Memory, MemoryType
from engram.store import MemoryStore

logger = logging.getLogger(__name__)

# Trigger phrases for explicit memory signals
REMEMBER_TRIGGERS = [
    "remember that",
    "don't forget",
    "keep in mind",
    "note that",
    "important:",
    "remember this",
    "please remember",
    "make a note",
]

# Default extraction prompt
EXTRACTION_PROMPT = """Extract discrete facts, preferences, or instructions from this conversation turn.
Return ONLY a JSON array. Only extract information worth remembering long-term.
If nothing is worth remembering, return an empty array [].

User: {user_message}
Assistant: {assistant_response}

Return format (strict JSON, no extra text):
[{{"content": "...", "type": "fact|preference|instruction|context", "importance": 0.0-1.0}}]"""


class MemoryExtractor:
    """Extracts memorable facts from conversation turns.
    
    Runs asynchronously after the LLM response is delivered.
    Uses a 4-tier fallback parser for robustness against unreliable JSON output.
    Detects explicit memory signals ("remember that...") for importance boosting.
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

    def extract_async(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Extract memories in a background thread (non-blocking).
        
        This is the primary entry point — called after the LLM response
        is delivered to the user so it doesn't add perceived latency.
        """
        thread = threading.Thread(
            target=self._extract_and_store,
            args=(user_message, assistant_response, model, session_id),
            daemon=True,
        )
        thread.start()

    def extract_sync(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
        """Extract memories synchronously (blocking). Used for testing."""
        return self._extract_and_store(
            user_message, assistant_response, model, session_id
        )

    # ---------- Core extraction ----------

    def _extract_and_store(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
        """Extract memories from a turn and store them.
        
        Returns:
            List of Memory objects that were stored.
        """
        stored = []

        try:
            # Check for explicit memory signals first
            explicit = self._detect_explicit_signals(user_message)

            # Call LLM for extraction
            raw_memories = self._call_extraction_llm(
                user_message, assistant_response, model
            )

            # Parse the response (4-tier fallback)
            parsed = self._parse_extraction(raw_memories)

            if not parsed and not explicit:
                return []

            # Combine explicit signals with LLM-extracted memories
            all_memories = explicit + parsed

            # Deduplicate by content similarity (simple substring check)
            all_memories = self._deduplicate(all_memories)

            # Store each memory
            for mem_data in all_memories:
                memory = Memory(
                    content=mem_data["content"],
                    type=MemoryType(mem_data.get("type", "fact")),
                    importance=float(mem_data.get("importance", 0.5)),
                    source_session=session_id,
                    pinned=mem_data.get("pinned", False),
                    embedding_model=self.embedding_engine.model_name,
                )

                # Generate embedding
                embedding = self.embedding_engine.embed(memory.content)

                # Store
                self.store.add(memory, embedding)
                stored.append(memory)

                logger.debug(
                    f"Extracted memory [{memory.type.value}] "
                    f"(importance={memory.importance:.2f}): {memory.content[:60]}"
                )

        except Exception as e:
            # Extraction failures are never surfaced to the user
            logger.warning(f"Memory extraction failed: {e}")

        return stored

    # ---------- LLM extraction call ----------

    def _call_extraction_llm(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
    ) -> str:
        """Call the LLM with the extraction prompt."""
        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response[:500],  # Truncate long responses
        )

        try:
            response = self.llm_client.chat_full(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temp for structured output
            )
            return response
        except Exception as e:
            logger.warning(f"Extraction LLM call failed: {e}")
            return ""

    # ---------- 4-tier fallback parser ----------

    @staticmethod
    def _parse_extraction(raw_response: str) -> list[dict]:
        """Parse LLM extraction response with 4-tier fallback.
        
        Tier 1: Direct JSON parse
        Tier 2: Extract JSON from markdown code blocks
        Tier 3: Find array anywhere in response
        Tier 4: Bullet point regex extraction
        
        Returns empty list if all tiers fail.
        """
        if not raw_response or not raw_response.strip():
            return []

        raw = raw_response.strip()

        # Tier 1: Direct JSON parse
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [m for m in result if isinstance(m, dict) and "content" in m]
        except json.JSONDecodeError:
            pass

        # Tier 2: JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, list):
                    return [m for m in result if isinstance(m, dict) and "content" in m]
            except json.JSONDecodeError:
                pass

        # Tier 3: Find array anywhere in response
        array_match = re.search(r'\[[\s\S]*\]', raw)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(result, list):
                    return [m for m in result if isinstance(m, dict) and "content" in m]
            except json.JSONDecodeError:
                pass

        # Tier 4: Bullet point regex extraction
        bullets = re.findall(r'[-•*]\s*(.+)', raw)
        if bullets:
            return [
                {"content": b.strip(), "type": "fact", "importance": 0.5}
                for b in bullets
                if len(b.strip()) > 5  # Skip trivially short bullets
            ]

        # All tiers failed
        logger.warning(f"Extraction parse failed (all 4 tiers): {raw[:100]}")
        return []

    # ---------- Explicit memory signals ----------

    @staticmethod
    def _detect_explicit_signals(user_message: str) -> list[dict]:
        """Detect explicit memory requests in the user's message.
        
        Trigger phrases like "remember that..." cause:
        - Importance boosted to 0.95
        - Memory auto-pinned
        """
        message_lower = user_message.lower()
        
        for trigger in REMEMBER_TRIGGERS:
            if trigger in message_lower:
                # Extract the content after the trigger phrase
                idx = message_lower.index(trigger)
                content = user_message[idx + len(trigger):].strip()
                
                # Clean up leading punctuation/whitespace
                content = content.lstrip(",:;- ")
                
                if content and len(content) > 3:
                    return [{
                        "content": content,
                        "type": "fact",
                        "importance": 0.95,
                        "pinned": True,
                    }]
        
        return []

    # ---------- Deduplication ----------

    @staticmethod
    def _deduplicate(memories: list[dict]) -> list[dict]:
        """Remove near-duplicate memories from extraction results.
        
        Simple substring-based dedup. Full semantic dedup happens
        in the conflict resolver.
        """
        seen = []
        unique = []
        for mem in memories:
            content = mem["content"].lower().strip()
            # Check if any existing content is a substring or vice versa
            is_dup = False
            for s in seen:
                if content in s or s in content:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(content)
                unique.append(mem)
        return unique
