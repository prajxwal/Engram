"""REST API — FastAPI server for Engram.

Start with: engram serve --port 8100

Endpoints:
    POST   /v1/chat              — chat with memory
    POST   /v1/memories          — add a memory
    GET    /v1/memories          — list memories
    GET    /v1/memories/search   — semantic search
    DELETE /v1/memories/{id}     — delete a memory
    POST   /v1/memories/{id}/pin — pin/unpin a memory
    GET    /v1/stats             — memory statistics
    POST   /v1/export            — export all memories
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engram.client import MemoryClient
from engram.config import EngramConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[dict]
    stream: bool = True
    extract: bool = True


class AddMemoryRequest(BaseModel):
    content: str
    type: str = "fact"
    importance: float = 0.5
    pin: bool = False


class MemoryResponse(BaseModel):
    id: str
    content: str
    type: str
    importance: float
    pinned: bool
    archived: bool
    access_count: int
    created_at: str
    last_accessed: str


class StatsResponse(BaseModel):
    active_memories: int
    archived_memories: int
    total_memories: int
    pinned_memories: int
    conflict_candidates: int
    avg_importance: float
    type_breakdown: dict


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[EngramConfig] = None) -> FastAPI:
    """Create the FastAPI application with an Engram client."""
    
    app = FastAPI(
        title="Engram API",
        description="Persistent, semantic memory for local LLMs",
        version="0.1.0",
    )
    
    # Shared client instance
    client = MemoryClient()

    # ------ Chat ------

    @app.post("/v1/chat")
    async def chat(request: ChatRequest):
        """Chat with memory-augmented LLM."""
        model = request.model or client.config.default_model

        if request.stream:
            def generate():
                for chunk in client.chat(
                    messages=request.messages,
                    model=model,
                    stream=True,
                    extract=request.extract,
                ):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/plain")
        else:
            chunks = list(client.chat(
                messages=request.messages,
                model=model,
                stream=False,
                extract=request.extract,
            ))
            return {"response": "".join(chunks)}

    # ------ Memories CRUD ------

    @app.post("/v1/memories", response_model=MemoryResponse)
    async def add_memory(request: AddMemoryRequest):
        """Add a memory manually."""
        memory = client.add(
            content=request.content,
            type=request.type,
            importance=request.importance,
            pin=request.pin,
        )
        return _memory_to_response(memory)

    @app.get("/v1/memories")
    async def list_memories(
        archived: bool = Query(False, description="Include archived memories"),
    ):
        """List all memories."""
        memories = client.list(include_archived=archived)
        return [_memory_to_response(m) for m in memories]

    @app.get("/v1/memories/search")
    async def search_memories(
        q: str = Query(..., description="Search query"),
        top_k: int = Query(5, description="Number of results"),
    ):
        """Semantic search over memories."""
        results = client.search(q, top_k=top_k)
        return [
            {**_memory_to_response(m).__dict__, "score": round(score, 4)}
            for m, score in results
        ]

    @app.delete("/v1/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a memory."""
        # Support prefix matching
        all_memories = client.list(include_archived=True)
        matches = [m for m in all_memories if m.id.startswith(memory_id)]

        if not matches:
            raise HTTPException(404, f"Memory not found: {memory_id}")
        if len(matches) > 1:
            raise HTTPException(400, f"Ambiguous ID prefix. Matches: {[m.id for m in matches]}")

        client.forget(matches[0].id)
        return {"deleted": matches[0].id}

    @app.post("/v1/memories/{memory_id}/pin")
    async def pin_memory(memory_id: str, unpin: bool = Query(False)):
        """Pin or unpin a memory."""
        all_memories = client.list(include_archived=True)
        matches = [m for m in all_memories if m.id.startswith(memory_id)]

        if not matches:
            raise HTTPException(404, f"Memory not found: {memory_id}")

        if unpin:
            success = client.unpin(matches[0].id)
        else:
            success = client.pin(matches[0].id)

        if not success:
            raise HTTPException(500, "Failed to update pin status")

        return {"id": matches[0].id, "pinned": not unpin}

    # ------ Stats ------

    @app.get("/v1/stats", response_model=StatsResponse)
    async def get_stats():
        """Get memory statistics."""
        return client.stats()

    # ------ Export ------

    @app.post("/v1/export")
    async def export_memories():
        """Export all memories as JSON."""
        memories = client.list(include_archived=True)
        return {
            "version": "0.1.0",
            "memories": [m.to_dict() for m in memories],
        }

    # ------ Shutdown ------

    @app.on_event("shutdown")
    async def shutdown():
        client.close()

    return app


def _memory_to_response(memory) -> MemoryResponse:
    """Convert a Memory object to an API response model."""
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        type=memory.type.value,
        importance=memory.importance,
        pinned=memory.pinned,
        archived=memory.archived,
        access_count=memory.access_count,
        created_at=memory.created_at.isoformat(),
        last_accessed=memory.last_accessed.isoformat(),
    )
