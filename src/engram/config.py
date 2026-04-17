"""Configuration management for Engram.

Loads config with this priority: explicit args > env vars > config file > defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.join(Path.home(), ".engram")
DEFAULT_LLM_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DECAY_RATE = 0.005          # per hour — half-life ≈ 5.8 days
DEFAULT_TOP_K = 10                  # ChromaDB candidates before rerank
DEFAULT_TOP_N = 5                   # Memories injected after rerank
DEFAULT_CONTEXT_BUDGET_RATIO = 0.25 # Max 25% of context window for memories
DEFAULT_DEFAULT_MODEL = "llama3"
DEFAULT_PRUNE_THRESHOLD = 0.05      # Relevance below this → archive
DEFAULT_PRUNE_INTERVAL = 100        # Run pruning every N interactions
DEFAULT_MEMORY_SOFT_LIMIT = 1000    # Trigger aggressive pruning
DEFAULT_MEMORY_HARD_LIMIT = 5000    # Permanently delete oldest archived
DEFAULT_CONFLICT_SIMILARITY = 0.75  # Cosine sim threshold for conflict candidates
DEFAULT_API_PORT = 8100


@dataclass
class EngramConfig:
    """Central configuration for all Engram components.
    
    Attributes:
        data_dir: Root directory for all Engram data (~/.engram/)
        llm_base_url: OpenAI-compatible API endpoint
        embedding_model: Name of the embedding model to use
        default_model: Default LLM model name for chat
        decay_rate: λ in the decay formula (per hour)
        top_k: Number of ChromaDB candidates to retrieve
        top_n: Number of memories to inject after reranking
        context_budget_ratio: Max fraction of context window for memories
        prune_threshold: Relevance score below which memories are archived
        prune_interval: Run pruning every N interactions
        memory_soft_limit: Active memory count that triggers aggressive pruning
        memory_hard_limit: Total memory count (active + archived) hard cap
        conflict_similarity: Cosine similarity threshold for conflict detection
        api_port: Port for the REST API server
        injection_template: Optional path to custom injection template file
    """
    data_dir: str = DEFAULT_DATA_DIR
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_model: str = DEFAULT_DEFAULT_MODEL
    decay_rate: float = DEFAULT_DECAY_RATE
    top_k: int = DEFAULT_TOP_K
    top_n: int = DEFAULT_TOP_N
    context_budget_ratio: float = DEFAULT_CONTEXT_BUDGET_RATIO
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD
    prune_interval: int = DEFAULT_PRUNE_INTERVAL
    memory_soft_limit: int = DEFAULT_MEMORY_SOFT_LIMIT
    memory_hard_limit: int = DEFAULT_MEMORY_HARD_LIMIT
    conflict_similarity: float = DEFAULT_CONFLICT_SIMILARITY
    api_port: int = DEFAULT_API_PORT
    injection_template: Optional[str] = None

    # ---------- Derived paths ----------

    @property
    def db_path(self) -> str:
        """SQLite database file path."""
        return os.path.join(self.data_dir, "engram.db")

    @property
    def chroma_path(self) -> str:
        """ChromaDB persistent storage directory."""
        return os.path.join(self.data_dir, "chroma")

    @property
    def config_file_path(self) -> str:
        """Path to the JSON config file."""
        return os.path.join(self.data_dir, "config.json")

    @property
    def conflicts_log_path(self) -> str:
        """Path to the conflicts JSONL audit log."""
        return os.path.join(self.data_dir, "conflicts.jsonl")

    @property
    def lock_file_path(self) -> str:
        """File lock for single-writer concurrency."""
        return os.path.join(self.data_dir, ".lock")

    # ---------- Initialization ----------

    def ensure_dirs(self) -> None:
        """Create data directory structure if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

    # ---------- Persistence ----------

    def save(self) -> None:
        """Write current config to config.json."""
        self.ensure_dirs()
        # Only persist user-configurable fields (not derived paths)
        data = {
            k: v for k, v in asdict(self).items()
            if v is not None
        }
        with open(self.config_file_path, "w") as f:
            json.dump(data, f, indent=2)

    # ---------- Loading ----------

    @classmethod
    def load(cls, **overrides) -> EngramConfig:
        """Load config with priority: explicit overrides > env vars > config file > defaults.
        
        Args:
            **overrides: Explicit values that take highest priority.
            
        Returns:
            Fully resolved EngramConfig.
        """
        # Start with defaults
        config = cls()

        # Layer 1: Config file (if it exists)
        if os.path.exists(config.config_file_path):
            try:
                with open(config.config_file_path) as f:
                    file_data = json.load(f)
                for key, value in file_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except (json.JSONDecodeError, OSError):
                pass  # Corrupted config file — use defaults

        # Layer 2: Environment variables
        env_map = {
            "ENGRAM_DATA_DIR": ("data_dir", str),
            "ENGRAM_LLM_BASE_URL": ("llm_base_url", str),
            "ENGRAM_EMBEDDING_MODEL": ("embedding_model", str),
            "ENGRAM_DEFAULT_MODEL": ("default_model", str),
            "ENGRAM_DECAY_RATE": ("decay_rate", float),
            "ENGRAM_TOP_K": ("top_k", int),
            "ENGRAM_TOP_N": ("top_n", int),
            "ENGRAM_CONTEXT_BUDGET_RATIO": ("context_budget_ratio", float),
            "ENGRAM_API_PORT": ("api_port", int),
        }
        for env_var, (attr, type_fn) in env_map.items():
            env_val = os.environ.get(env_var)
            if env_val is not None:
                try:
                    setattr(config, attr, type_fn(env_val))
                except (ValueError, TypeError):
                    pass  # Bad env var value — skip

        # Layer 3: Explicit overrides (highest priority)
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

        return config

    def set_value(self, key: str, value: str) -> None:
        """Set a config value by key name (from CLI), auto-casting types.
        
        Args:
            key: Config field name (e.g., 'decay_rate')
            value: String value to set (will be cast to field's type)
            
        Raises:
            ValueError: If key is not a valid config field.
        """
        if not hasattr(self, key) or key.startswith("_"):
            raise ValueError(
                f"Unknown config key: '{key}'. "
                f"Valid keys: {', '.join(f.name for f in self.__dataclass_fields__.values())}"
            )

        # Get the type annotation for the field
        field_type = self.__dataclass_fields__[key].type
        
        # Cast string value to appropriate type
        type_map = {"str": str, "int": int, "float": float}
        cast_fn = type_map.get(field_type, str)
        
        try:
            setattr(self, key, cast_fn(value))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot set '{key}' to '{value}': {e}")
        
        self.save()
