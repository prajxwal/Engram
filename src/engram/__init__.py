"""Engram — Persistent, semantic memory for local LLMs.

Give your local LLM a hippocampus.

Usage:
    from engram import MemoryClient

    memory = MemoryClient()
    response = memory.chat(
        model="llama3",
        messages=[{"role": "user", "content": "My name is Alex"}]
    )
"""

__version__ = "0.1.0"

from engram.models import Memory, MemoryType, ConflictRecord, ConflictResolution
from engram.config import EngramConfig

# MemoryClient imported lazily to avoid heavy deps on import
def __getattr__(name):
    if name == "MemoryClient":
        from engram.client import MemoryClient
        return MemoryClient
    raise AttributeError(f"module 'engram' has no attribute {name}")

__all__ = [
    "MemoryClient",
    "Memory",
    "MemoryType",
    "EngramConfig",
    "ConflictRecord",
    "ConflictResolution",
    "__version__",
]
