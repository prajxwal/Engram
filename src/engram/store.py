"""Dual-backend memory storage: ChromaDB (vectors) + SQLite (metadata).

This is the persistence layer — everything above (retriever, extractor,
client) talks to MemoryStore, never directly to ChromaDB or SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb

from engram.config import EngramConfig
from engram.models import ConflictRecord, Memory, MemoryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'fact',
    importance      REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    last_accessed   TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    source_session  TEXT,
    pinned          INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0,
    conflict_candidate INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);
"""


# ---------------------------------------------------------------------------
# File lock (single-writer concurrency)
# ---------------------------------------------------------------------------

class _FileLock:
    """Simple file-based lock for single-writer access.
    
    On Windows, uses msvcrt; on Unix, uses fcntl.
    Falls back to a simple existence check if neither is available.
    """

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired, False if held by another process."""
        try:
            # O_CREAT | O_EXCL is atomic on all platforms
            self._fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            # Write our PID for debugging
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            # Lock already held — check if the process is still alive
            return self._check_stale()
        except OSError:
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    def _check_stale(self) -> bool:
        """Check if an existing lock file is stale (owner process is dead)."""
        try:
            with open(self.lock_path) as f:
                pid = int(f.read().strip())
            # Check if process exists
            try:
                os.kill(pid, 0)  # Signal 0 = check existence
                return False  # Process alive — lock is valid
            except (OSError, ProcessLookupError):
                # Process dead — lock is stale, take it over
                logger.info(f"Removing stale lock (PID {pid} is dead)")
                os.unlink(self.lock_path)
                return self.acquire()
        except (ValueError, OSError):
            # Can't read lock file — remove it and try again
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass
            return self.acquire()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Dual-backend persistent storage for memories.
    
    ChromaDB handles vector embeddings and similarity search.
    SQLite handles structured metadata, decay tracking, and audit trails.
    
    Thread safety: single-writer via file lock. Multiple readers OK (SQLite WAL mode).
    """

    COLLECTION_NAME = "engram_memories"

    def __init__(self, config: EngramConfig):
        self.config = config
        self._read_only = False
        self._lock: Optional[_FileLock] = None
        self._db: Optional[sqlite3.Connection] = None
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None
        self._initialized = False

    # ---------- Lifecycle ----------

    def initialize(self) -> None:
        """Set up storage backends. Call once before use.
        
        Creates directories, acquires lock, initializes SQLite schema
        and ChromaDB collection. Lazy — not called on __init__.
        """
        if self._initialized:
            return

        self.config.ensure_dirs()

        # Acquire file lock
        self._lock = _FileLock(self.config.lock_file_path)
        if not self._lock.acquire():
            logger.warning(
                "Another Engram instance is running. "
                "Memory writes disabled in this session."
            )
            self._read_only = True

        # SQLite — WAL mode for concurrent reads
        self._db = sqlite3.connect(self.config.db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA_SQL)
        self._db.commit()

        # ChromaDB
        self._chroma_client = chromadb.PersistentClient(
            path=self.config.chroma_path,
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._initialized = True
        logger.info(
            f"MemoryStore initialized (data_dir={self.config.data_dir}, "
            f"read_only={self._read_only})"
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._db:
            self._db.close()
            self._db = None
        if self._lock:
            self._lock.release()
            self._lock = None
        self._initialized = False

    def _ensure_init(self) -> None:
        """Lazy initialization guard."""
        if not self._initialized:
            self.initialize()

    def _ensure_writable(self) -> None:
        """Raise if in read-only mode."""
        if self._read_only:
            raise RuntimeError(
                "MemoryStore is in read-only mode. "
                "Another Engram instance holds the write lock."
            )

    # ---------- CRUD: Add ----------

    def add(self, memory: Memory, embedding: list[float]) -> Memory:
        """Store a new memory with its embedding vector.
        
        Args:
            memory: The Memory object to store.
            embedding: Pre-computed embedding vector.
            
        Returns:
            The stored Memory (unchanged).
        """
        self._ensure_init()
        self._ensure_writable()

        # ChromaDB — store embedding
        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[{"type": memory.type.value}],
        )

        # SQLite — store metadata
        self._db.execute(
            """INSERT INTO memories 
               (id, content, type, importance, created_at, last_accessed,
                access_count, source_session, pinned, archived, 
                conflict_candidate, embedding_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.content,
                memory.type.value,
                memory.importance,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.source_session,
                int(memory.pinned),
                int(memory.archived),
                int(memory.conflict_candidate),
                memory.embedding_model,
            ),
        )
        self._db.commit()

        logger.debug(f"Stored memory {memory.id}: {memory.content[:50]}...")
        return memory

    # ---------- CRUD: Get ----------

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        self._ensure_init()
        row = self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    # ---------- CRUD: Update ----------

    def update(self, memory: Memory, embedding: Optional[list[float]] = None) -> None:
        """Update a memory's metadata (and optionally its embedding).
        
        Args:
            memory: Memory with updated fields.
            embedding: If provided, also update the ChromaDB embedding.
        """
        self._ensure_init()
        self._ensure_writable()

        self._db.execute(
            """UPDATE memories SET
                content = ?, type = ?, importance = ?, last_accessed = ?,
                access_count = ?, pinned = ?, archived = ?,
                conflict_candidate = ?, embedding_model = ?
               WHERE id = ?""",
            (
                memory.content,
                memory.type.value,
                memory.importance,
                memory.last_accessed.isoformat(),
                memory.access_count,
                int(memory.pinned),
                int(memory.archived),
                int(memory.conflict_candidate),
                memory.embedding_model,
                memory.id,
            ),
        )
        self._db.commit()

        if embedding is not None:
            self._collection.update(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[{"type": memory.type.value}],
            )

    # ---------- CRUD: Delete ----------

    def delete(self, memory_id: str) -> bool:
        """Permanently delete a memory from both backends.
        
        Returns:
            True if the memory existed and was deleted.
        """
        self._ensure_init()
        self._ensure_writable()

        row = self._db.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self._db.commit()

        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass  # May not exist in ChromaDB if it was archived

        return row.rowcount > 0

    # ---------- Search (vector similarity) ----------

    def search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        include_archived: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Semantic search: find memories most similar to a query embedding.
        
        Args:
            query_embedding: The query vector to search against.
            top_k: Number of candidates to retrieve from ChromaDB.
            include_archived: Whether to include archived memories.
            
        Returns:
            List of (Memory, similarity_score) tuples, sorted by score desc.
        """
        self._ensure_init()
        k = top_k or self.config.top_k

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count() or 1),
            include=["distances", "documents"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        memories_with_scores = []
        ids = results["ids"][0]
        # ChromaDB returns distances; for cosine, distance = 1 - similarity
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for memory_id, distance in zip(ids, distances):
            memory = self.get(memory_id)
            if memory is None:
                continue
            if memory.archived and not include_archived:
                continue
            similarity = 1.0 - distance  # Convert distance to similarity
            memories_with_scores.append((memory, similarity))

        return memories_with_scores

    # ---------- List ----------

    def list_memories(
        self,
        include_archived: bool = False,
        memory_type: Optional[MemoryType] = None,
        pinned_only: bool = False,
    ) -> list[Memory]:
        """List memories with optional filters.
        
        Args:
            include_archived: If True, include archived memories.
            memory_type: Filter by memory type.
            pinned_only: If True, only return pinned memories.
            
        Returns:
            List of Memory objects.
        """
        self._ensure_init()

        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if not include_archived:
            query += " AND archived = 0"
        if memory_type is not None:
            query += " AND type = ?"
            params.append(memory_type.value)
        if pinned_only:
            query += " AND pinned = 1"

        query += " ORDER BY last_accessed DESC"

        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    # ---------- Archive / Restore ----------

    def archive(self, memory_id: str) -> bool:
        """Move a memory to archived state (soft delete).
        
        Returns:
            True if the memory existed and was archived.
        """
        self._ensure_init()
        self._ensure_writable()

        row = self._db.execute(
            "UPDATE memories SET archived = 1 WHERE id = ? AND archived = 0",
            (memory_id,),
        )
        self._db.commit()

        if row.rowcount > 0:
            # Remove from ChromaDB (archived memories aren't searchable)
            try:
                self._collection.delete(ids=[memory_id])
            except Exception:
                pass
            return True
        return False

    def restore(self, memory_id: str, embedding: list[float]) -> bool:
        """Restore an archived memory back to active state.
        
        Args:
            memory_id: ID of the archived memory.
            embedding: Re-computed embedding vector.
            
        Returns:
            True if the memory existed and was restored.
        """
        self._ensure_init()
        self._ensure_writable()

        memory = self.get(memory_id)
        if memory is None or not memory.archived:
            return False

        memory.archived = False
        memory.last_accessed = datetime.now(timezone.utc)
        self.update(memory)

        # Re-add to ChromaDB
        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[{"type": memory.type.value}],
        )

        return True

    # ---------- Stats ----------

    def count(self, include_archived: bool = False) -> int:
        """Count total memories."""
        self._ensure_init()
        if include_archived:
            row = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 0"
            ).fetchone()
        return row[0] if row else 0

    def stats(self) -> dict:
        """Get aggregate statistics about the memory store."""
        self._ensure_init()
        active = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]
        archived = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 1"
        ).fetchone()[0]
        pinned = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE pinned = 1"
        ).fetchone()[0]
        conflict_candidates = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE conflict_candidate = 1"
        ).fetchone()[0]
        avg_importance = self._db.execute(
            "SELECT AVG(importance) FROM memories WHERE archived = 0"
        ).fetchone()[0]
        
        type_counts = {}
        for row in self._db.execute(
            "SELECT type, COUNT(*) FROM memories WHERE archived = 0 GROUP BY type"
        ).fetchall():
            type_counts[row[0]] = row[1]

        return {
            "active_memories": active,
            "archived_memories": archived,
            "total_memories": active + archived,
            "pinned_memories": pinned,
            "conflict_candidates": conflict_candidates,
            "avg_importance": round(avg_importance, 3) if avg_importance else 0.0,
            "type_breakdown": type_counts,
        }

    # ---------- Conflict log ----------

    def log_conflict(self, record: ConflictRecord) -> None:
        """Append a conflict record to the JSONL audit log."""
        self._ensure_init()
        with open(self.config.conflicts_log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def list_conflicts(self) -> list[dict]:
        """Read all conflict records from the audit log."""
        conflicts = []
        log_path = self.config.conflicts_log_path
        if not os.path.exists(log_path):
            return conflicts
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conflicts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return conflicts

    # ---------- Meta (key-value store for internal state) ----------

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        self._ensure_init()
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self._ensure_init()
        self._ensure_writable()
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db.commit()

    # ---------- Export / Import ----------

    def export_memories(self, path: str) -> int:
        """Export all memories (including archived) to a JSON file.
        
        Returns:
            Number of memories exported.
        """
        self._ensure_init()
        memories = self.list_memories(include_archived=True)
        data = {
            "version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": [m.to_dict() for m in memories],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return len(memories)

    def import_memories(
        self, path: str, embed_fn: callable
    ) -> int:
        """Import memories from a JSON file.
        
        Args:
            path: Path to the JSON export file.
            embed_fn: Function to generate embeddings for imported memories.
            
        Returns:
            Number of memories imported.
        """
        self._ensure_init()
        self._ensure_writable()

        with open(path) as f:
            data = json.load(f)

        count = 0
        for mem_data in data.get("memories", []):
            memory = Memory.from_dict(mem_data)
            # Skip if already exists
            if self.get(memory.id) is not None:
                continue
            embedding = embed_fn(memory.content)
            self.add(memory, embedding)
            count += 1

        logger.info(f"Imported {count} memories from {path}")
        return count

    # ---------- Helpers ----------

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        """Convert a SQLite row to a Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            type=MemoryType(row["type"]),
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            source_session=row["source_session"],
            pinned=bool(row["pinned"]),
            archived=bool(row["archived"]),
            conflict_candidate=bool(row["conflict_candidate"]),
            embedding_model=row["embedding_model"],
        )

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()
