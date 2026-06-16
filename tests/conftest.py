"""Shared test fixtures for Engram test suite."""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine
from engram.models import Memory, MemoryType
from engram.store import MemoryStore


# ---------------------------------------------------------------------------
# Deterministic embedding engine (no model download needed)
# ---------------------------------------------------------------------------

class MockEmbeddingEngine(EmbeddingEngine):
    """Produces deterministic 64-dim embeddings from text hash.
    
    Used for testing — no model download, no GPU, instant results.
    Produces vectors that are somewhat meaningful: similar texts
    will produce similar (but not identical) vectors.
    """

    DIMS = 64

    @property
    def model_name(self) -> str:
        return "mock:test-64d"

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Convert bytes to floats in [-1, 1]
        raw = [(b / 127.5) - 1.0 for b in h[:self.DIMS]]
        # L2 normalize
        norm = sum(x * x for x in raw) ** 0.5
        if norm == 0:
            return [0.0] * self.DIMS
        return [x / norm for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Mock LLM client (no Ollama needed)
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Mock LLM client that returns canned responses.
    
    Set .next_response to control what chat_full() returns.
    Set .extraction_response for extraction-specific calls.
    """

    def __init__(self):
        self.next_response: str = "[]"
        self.call_log: list[dict] = []

    def chat(self, model, messages, stream=True, temperature=0.7):
        self.call_log.append({
            "method": "chat",
            "model": model,
            "messages": messages,
        })
        yield self.next_response

    def chat_full(self, model, messages, temperature=0.7):
        self.call_log.append({
            "method": "chat_full",
            "model": model,
            "messages": messages,
        })
        return self.next_response

    def check_health(self):
        return True

    def is_ollama(self):
        return True

    def get_context_window(self, model):
        return 4096

    def list_models(self):
        return ["test-model"]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """EngramConfig pointing at a temporary directory."""
    return EngramConfig(
        data_dir=str(tmp_path / "engram_test"),
        llm_base_url="http://localhost:11434",
        embedding_model="mock:test-64d",
        default_model="test-model",
        decay_rate=0.005,
        top_k=10,
        top_n=5,
        context_budget_ratio=0.25,
    )


@pytest.fixture
def store(tmp_config):
    """Initialized MemoryStore backed by temp dir."""
    s = MemoryStore(tmp_config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def mock_embeddings():
    """MockEmbeddingEngine instance."""
    return MockEmbeddingEngine()


@pytest.fixture
def mock_llm():
    """MockLLMClient instance."""
    return MockLLMClient()


@pytest.fixture
def sample_memory():
    """Factory to create Memory objects with defaults."""
    def _make(
        content: str = "The user's name is Alex",
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        pinned: bool = False,
    ) -> Memory:
        return Memory(
            content=content,
            type=memory_type,
            importance=importance,
            pinned=pinned,
            source_session="test-session",
            embedding_model="mock:test-64d",
        )
    return _make
