"""MemoryClient — the main public API for Engram.

This is the primary interface developers use:
    from engram import MemoryClient
    memory = MemoryClient()
    response = memory.chat(model="llama3", messages=[...])
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterator, Optional

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine, create_embedding_engine
from engram.extractor import MemoryExtractor
from engram.llm import LLMClient
from engram.models import Memory, MemoryType
from engram.retriever import MemoryRetriever
from engram.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryClient:
    """Drop-in memory layer for local LLMs.
    
    Provides:
    - chat() — chat with automatic memory retrieval + storage
    - add() — manually add a memory
    - search() — semantic search over memories
    - pin() / forget() — memory management
    - list() — list stored memories
    - export() / import_memories() — data portability
    
    Example:
        memory = MemoryClient()
        for chunk in memory.chat(model="llama3", messages=[...]):
            print(chunk, end="")
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        default_model: Optional[str] = None,
        decay_rate: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        context_budget_ratio: Optional[float] = None,
    ):
        # Load config with layered priority
        self.config = EngramConfig.load(
            data_dir=data_dir,
            llm_base_url=llm_base_url,
            embedding_model=embedding_model,
            default_model=default_model,
            decay_rate=decay_rate,
            top_k=top_k,
            top_n=top_n,
            context_budget_ratio=context_budget_ratio,
        )

        # Initialize components (lazy where possible)
        self._store: Optional[MemoryStore] = None
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._llm_client: Optional[LLMClient] = None
        self._retriever: Optional[MemoryRetriever] = None
        self._extractor: Optional[MemoryExtractor] = None

        # Session tracking
        self._session_id = uuid.uuid4().hex
        self._interaction_count = 0

    # ---------- Lazy initialization ----------

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = MemoryStore(self.config)
            self._store.initialize()
            self._check_embedding_model()
        return self._store

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            self._embedding_engine = create_embedding_engine(
                model_spec=self.config.embedding_model,
                ollama_base_url=self.config.llm_base_url,
            )
        return self._embedding_engine

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(base_url=self.config.llm_base_url)
        return self._llm_client

    @property
    def retriever(self) -> MemoryRetriever:
        if self._retriever is None:
            self._retriever = MemoryRetriever(
                store=self.store,
                embedding_engine=self.embedding_engine,
                config=self.config,
            )
        return self._retriever

    @property
    def extractor(self) -> MemoryExtractor:
        if self._extractor is None:
            self._extractor = MemoryExtractor(
                store=self.store,
                embedding_engine=self.embedding_engine,
                llm_client=self.llm_client,
                config=self.config,
            )
        return self._extractor

    # ---------- Chat (main interface) ----------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        stream: bool = True,
        extract: bool = True,
    ) -> Iterator[str]:
        """Chat with automatic memory retrieval and extraction.
        
        Pipeline:
        1. Retrieve relevant memories → inject into system message
        2. Send enriched prompt to LLM → stream response to user
        3. After response delivered, extract memories asynchronously
        
        Args:
            messages: Chat messages in OpenAI format.
            model: LLM model name. Defaults to config.default_model.
            stream: If True, yield response chunks as they arrive.
            extract: If True, run async memory extraction after response.
            
        Yields:
            Response text chunks.
        """
        model = model or self.config.default_model

        # Step 1: Retrieve relevant memories
        user_message = self._get_last_user_message(messages)
        context_window = self.llm_client.get_context_window(model)
        
        memory_context = None
        if user_message:
            memory_context = self.retriever.retrieve(
                query=user_message,
                model_context_window=context_window,
            )

        # Step 2: Inject memories into messages
        enriched_messages = self._inject_memories(messages, memory_context)

        # Step 3: Send to LLM and stream response
        full_response = []
        for chunk in self.llm_client.chat(
            model=model,
            messages=enriched_messages,
            stream=stream,
        ):
            full_response.append(chunk)
            yield chunk

        # Step 4: Async memory extraction
        response_text = "".join(full_response)
        if extract and user_message and response_text:
            self.extractor.extract_async(
                user_message=user_message,
                assistant_response=response_text,
                model=model,
                session_id=self._session_id,
            )

        # Track interactions for periodic pruning
        self._interaction_count += 1
        if self._interaction_count % self.config.prune_interval == 0:
            self._run_maintenance()

    # ---------- Manual memory operations ----------

    def add(
        self,
        content: str,
        type: str = "fact",
        importance: float = 0.5,
        pin: bool = False,
    ) -> Memory:
        """Manually add a memory.
        
        Args:
            content: The memory content string.
            type: Memory type (fact, preference, instruction, context).
            importance: Importance score (0.0 - 1.0).
            pin: If True, memory is pinned (exempt from decay).
            
        Returns:
            The created Memory object.
        """
        memory = Memory(
            content=content,
            type=MemoryType(type),
            importance=importance,
            source_session=self._session_id,
            pinned=pin,
            embedding_model=self.embedding_engine.model_name,
        )
        embedding = self.embedding_engine.embed(content)
        self.store.add(memory, embedding)
        logger.info(f"Added memory: {content[:50]}...")
        return memory

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[Memory, float]]:
        """Semantic search over memories.
        
        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            
        Returns:
            List of (Memory, similarity_score) tuples.
        """
        return self.retriever.retrieve_raw(query, top_k)

    def pin(self, memory_id: str) -> bool:
        """Pin a memory (exempt from decay).
        
        Returns:
            True if the memory was found and pinned.
        """
        memory = self.store.get(memory_id)
        if memory is None:
            return False
        memory.pinned = True
        self.store.update(memory)
        return True

    def unpin(self, memory_id: str) -> bool:
        """Unpin a memory (subject to decay again)."""
        memory = self.store.get(memory_id)
        if memory is None:
            return False
        memory.pinned = False
        self.store.update(memory)
        return True

    def forget(self, memory_id: str) -> bool:
        """Permanently delete a memory.
        
        Returns:
            True if the memory existed and was deleted.
        """
        return self.store.delete(memory_id)

    def list(self, include_archived: bool = False) -> list[Memory]:
        """List all memories.
        
        Args:
            include_archived: If True, include archived/decayed memories.
        """
        return self.store.list_memories(include_archived=include_archived)

    def stats(self) -> dict:
        """Get memory store statistics."""
        return self.store.stats()

    def export(self, path: str) -> int:
        """Export all memories to a JSON file.
        
        Returns:
            Number of memories exported.
        """
        return self.store.export_memories(path)

    def import_memories(self, path: str) -> int:
        """Import memories from a JSON file.
        
        Returns:
            Number of memories imported.
        """
        return self.store.import_memories(
            path, embed_fn=self.embedding_engine.embed
        )

    # ---------- Internal helpers ----------

    @staticmethod
    def _get_last_user_message(messages: list[dict]) -> Optional[str]:
        """Extract the last user message from the conversation."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    @staticmethod
    def _inject_memories(
        messages: list[dict],
        memory_context: Optional[str],
    ) -> list[dict]:
        """Inject memory context into the message list.
        
        If memory context exists, prepend it as a system message.
        If no memories, return messages unchanged (zero overhead).
        """
        if not memory_context:
            return messages

        # Build enriched message list
        enriched = []

        # Check if there's already a system message
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system:
            # Append memory context to existing system message
            for msg in messages:
                if msg["role"] == "system":
                    enriched.append({
                        "role": "system",
                        "content": msg["content"] + "\n\n" + memory_context,
                    })
                else:
                    enriched.append(msg)
        else:
            # Prepend as new system message
            enriched.append({"role": "system", "content": memory_context})
            enriched.extend(messages)

        return enriched

    def _check_embedding_model(self) -> None:
        """Detect embedding model changes and warn about migration."""
        stored_model = self.store.get_meta("embedding_model")
        current_model = self.embedding_engine.model_name

        if stored_model is None:
            # First run — store the model name
            self.store.set_meta("embedding_model", current_model)
        elif stored_model != current_model:
            logger.warning(
                f"Embedding model changed: {stored_model} → {current_model}. "
                f"Run 'engram reindex' to re-embed all memories. "
                f"Until then, retrieval quality will be degraded."
            )
            # Update the stored model name
            self.store.set_meta("embedding_model", current_model)

    def _run_maintenance(self) -> None:
        """Periodic maintenance: pruning, memory count management."""
        # Import here to avoid circular dependency
        from engram.decay import DecayEngine
        decay = DecayEngine(self.store, self.config)
        decay.run_pruning()

    # ---------- Lifecycle ----------

    def close(self) -> None:
        """Clean up resources."""
        if self._store:
            self._store.close()
        if self._llm_client:
            self._llm_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
