"""Benchmark: Retrieval latency at various memory store sizes.

Measures the end-to-end retrieval pipeline:
    embed query → ChromaDB top-k → decay rerank → token budget → format

Usage:
    python benchmarks/bench_retrieval_latency.py

Target: < 100ms at 1000 memories.
"""

import hashlib
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engram.config import EngramConfig
from engram.embeddings import EmbeddingEngine
from engram.models import Memory, MemoryType
from engram.retriever import MemoryRetriever
from engram.store import MemoryStore


# ---------------------------------------------------------------------------
# Lightweight embedding engine for benchmarking
# ---------------------------------------------------------------------------

class BenchmarkEmbeddingEngine(EmbeddingEngine):
    """Fast deterministic embeddings for benchmarking (no model download)."""

    DIMS = 384  # Match all-MiniLM-L6-v2 dimensions

    @property
    def model_name(self) -> str:
        return "benchmark:hash-384d"

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha384(text.encode()).digest()
        raw = [(b / 127.5) - 1.0 for b in h[:self.DIMS]]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw] if norm > 0 else [0.0] * self.DIMS

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Sample memories
# ---------------------------------------------------------------------------

SAMPLE_MEMORIES = [
    "The user's name is Alex and they work as a software engineer in San Francisco",
    "User prefers dark mode in all their applications",
    "User's cat is named Luna and she is 4 years old",
    "User is allergic to peanuts and shellfish",
    "User's favorite programming language is Python",
    "User uses Neovim as their primary code editor",
    "User's birthday is March 15th",
    "User prefers tea over coffee",
    "User is currently learning Rust programming language",
    "User works at a startup called TechFlow",
    "User's favorite book is Dune by Frank Herbert",
    "User runs 5 miles every morning before work",
    "User has a standing desk setup at home",
    "User prefers bullet-point responses over paragraphs",
    "User is working on a React project for their company",
    "User has a dog named Max who is 2 years old",
    "User prefers functional programming over OOP when possible",
    "User's favorite color is deep navy blue",
    "User takes notes in Obsidian and uses Zettelkasten method",
    "User is interested in machine learning and NLP",
]

SAMPLE_QUERIES = [
    "What is my name?",
    "Do I have any pets?",
    "What editor do I use?",
    "What am I allergic to?",
    "What project am I working on?",
    "What language am I learning?",
    "When is my birthday?",
    "What is my preferred response format?",
    "What company do I work for?",
    "What are my hobbies?",
]


def generate_memories(n: int) -> list[str]:
    """Generate n unique memory strings from the sample set."""
    memories = []
    for i in range(n):
        base = SAMPLE_MEMORIES[i % len(SAMPLE_MEMORIES)]
        if i >= len(SAMPLE_MEMORIES):
            memories.append(f"{base} (variant {i})")
        else:
            memories.append(base)
    return memories


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_retrieval(n_memories: int, n_queries: int = 50) -> dict:
    """Benchmark retrieval latency at a given store size.
    
    Args:
        n_memories: Number of memories to populate the store with.
        n_queries: Number of queries to run for latency measurement.
        
    Returns:
        Dict with p50, p95, p99, mean, min, max latency in ms.
    """
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="engram_bench_")
    
    config = EngramConfig(
        data_dir=tmpdir,
        top_k=10,
        top_n=5,
        context_budget_ratio=0.25,
    )
    
    store = MemoryStore(config)
    store.initialize()
    embed_engine = BenchmarkEmbeddingEngine()
    retriever = MemoryRetriever(store, embed_engine, config)
    
    # Populate store
    print(f"  Populating store with {n_memories} memories...", end=" ", flush=True)
    memory_strings = generate_memories(n_memories)
    for content in memory_strings:
        m = Memory(content=content, importance=0.5 + (hash(content) % 50) / 100.0)
        store.add(m, embed_engine.embed(content))
    print("done.")
    
    # Warm up
    for q in SAMPLE_QUERIES[:3]:
        retriever.retrieve(q, model_context_window=4096)
    
    # Benchmark
    latencies = []
    queries = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(n_queries)]
    
    for query in queries:
        start = time.perf_counter()
        retriever.retrieve(query, model_context_window=4096)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
    
    store.close()
    
    # Clean up
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    latencies.sort()
    return {
        "n_memories": n_memories,
        "n_queries": n_queries,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(latencies[len(latencies) // 2], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def main():
    print("=" * 70)
    print("Engram Retrieval Latency Benchmark")
    print("=" * 70)
    print(f"Pipeline: embed query → ChromaDB top-k → decay rerank → token budget → format")
    print(f"Embedding: hash-based (384d, no model download)")
    print()
    
    sizes = [100, 500, 1000]
    results = []
    
    for n in sizes:
        print(f"[{n} memories]")
        result = benchmark_retrieval(n, n_queries=50)
        results.append(result)
        print(f"  Mean: {result['mean_ms']:.1f}ms | "
              f"P50: {result['p50_ms']:.1f}ms | "
              f"P95: {result['p95_ms']:.1f}ms | "
              f"P99: {result['p99_ms']:.1f}ms")
        print()
    
    # Summary table
    print("-" * 70)
    print(f"{'Memories':>10} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Status':>10}")
    print("-" * 70)
    for r in results:
        status = "✓ PASS" if r["p95_ms"] < 100 else "✗ FAIL"
        print(f"{r['n_memories']:>10} {r['mean_ms']:>9.1f}ms {r['p50_ms']:>9.1f}ms "
              f"{r['p95_ms']:>9.1f}ms {r['p99_ms']:>9.1f}ms {status:>10}")
    print("-" * 70)
    
    # Check target
    at_1000 = results[-1]
    if at_1000["p95_ms"] < 100:
        print(f"\n✓ TARGET MET: P95 latency at 1000 memories = {at_1000['p95_ms']:.1f}ms (< 100ms)")
    else:
        print(f"\n✗ TARGET MISSED: P95 latency at 1000 memories = {at_1000['p95_ms']:.1f}ms (target: < 100ms)")


if __name__ == "__main__":
    main()
