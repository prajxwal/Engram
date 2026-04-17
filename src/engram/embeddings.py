"""Embedding engines for Engram.

Supports two backends:
1. SentenceTransformerEngine — local, uses all-MiniLM-L6-v2 (default)
2. OllamaEmbeddingEngine — uses Ollama's /api/embeddings endpoint

Both are lazy-loaded: the model isn't loaded until the first embed() call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingEngine(ABC):
    """Abstract base class for embedding engines."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: loop over embed()."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for migration detection."""
        ...


class SentenceTransformerEngine(EmbeddingEngine):
    """Embedding via sentence-transformers (local CPU inference).
    
    Default model: all-MiniLM-L6-v2 (384-dim, ~80MB download).
    Lazy-loaded on first embed() call.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # Lazy

    @property
    def model_name(self) -> str:
        return f"st:{self._model_name}"

    def _ensure_model(self) -> None:
        """Load the model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Embedding model loaded (dim={self._model.get_sentence_embedding_dimension()})")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers\n"
                "Or use Ollama embeddings: engram config set embedding_model ollama:nomic-embed-text"
            )

    def embed(self, text: str) -> list[float]:
        self._ensure_model()
        vector = self._model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        vectors = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vectors]


class OllamaEmbeddingEngine(EmbeddingEngine):
    """Embedding via Ollama's /api/embeddings endpoint.
    
    Requires Ollama to be running with an embedding model pulled.
    Lighter footprint than sentence-transformers (~0 extra install).
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)
        self._verified = False

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model_name}"

    def _verify(self) -> None:
        """Check that Ollama is running and the model is available."""
        if self._verified:
            return
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if model is available (with or without tag suffix)
            available = any(
                m == self._model_name or m.startswith(f"{self._model_name}:")
                for m in models
            )
            if not available:
                raise RuntimeError(
                    f"Ollama embedding model '{self._model_name}' not found. "
                    f"Available models: {', '.join(models)}\n"
                    f"Pull it with: ollama pull {self._model_name}"
                )
            self._verified = True
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )

    def embed(self, text: str) -> list[float]:
        self._verify()
        resp = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model_name, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama doesn't support batch embedding natively — loop
        return [self.embed(text) for text in texts]

    def close(self) -> None:
        self._client.close()


def create_embedding_engine(
    model_spec: str,
    ollama_base_url: str = "http://localhost:11434",
) -> EmbeddingEngine:
    """Factory: create the appropriate embedding engine from a model spec.
    
    Args:
        model_spec: Model identifier. Prefixed with "ollama:" for Ollama models,
                     otherwise treated as a sentence-transformers model name.
                     Examples:
                       - "all-MiniLM-L6-v2" → SentenceTransformerEngine
                       - "ollama:nomic-embed-text" → OllamaEmbeddingEngine
        ollama_base_url: Ollama API base URL (used only for Ollama models).
        
    Returns:
        Configured EmbeddingEngine instance.
    """
    if model_spec.startswith("ollama:"):
        model_name = model_spec[len("ollama:"):]
        return OllamaEmbeddingEngine(model_name=model_name, base_url=ollama_base_url)
    
    return SentenceTransformerEngine(model_name=model_spec)
