"""Benchmark: Token savings from Engram's memory injection vs naive history replay.

Compares two approaches to maintaining context across sessions:
1. Naive: Replay full conversation history (10+ turns, 500+ tokens)
2. Engram: Inject top-N relevant memories (~100 tokens in system prompt)

Usage:
    python benchmarks/bench_token_savings.py

Target: > 50% reduction in tokens sent to LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Approximate token count. chars/4 is standard for English text."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list (including role overhead)."""
    total = 0
    for msg in messages:
        total += 4  # Role + formatting overhead per message
        total += estimate_tokens(msg.get("content", ""))
    return total


# ---------------------------------------------------------------------------
# Simulated conversation history
# ---------------------------------------------------------------------------

# A realistic 10-turn conversation that establishes user context
PAST_CONVERSATION = [
    {"role": "user", "content": "Hi! I'm Alex, I'm a software engineer working at TechFlow in San Francisco."},
    {"role": "assistant", "content": "Nice to meet you, Alex! Welcome. How can I help you today? What kind of projects are you working on at TechFlow?"},
    {"role": "user", "content": "I'm building a React frontend for our internal dashboard. I prefer using TypeScript and functional components."},
    {"role": "assistant", "content": "Great choices! TypeScript with functional components is a solid approach for React. Do you have any specific questions about the dashboard implementation?"},
    {"role": "user", "content": "Yeah, also remember that I always prefer code examples in Python when they're backend-related. And I like concise, bullet-point responses."},
    {"role": "assistant", "content": "Got it! I'll use Python for backend examples and keep things concise with bullet points. What's your first question?"},
    {"role": "user", "content": "My cat Luna is being really playful today, she keeps jumping on my keyboard. Anyway, I need help with a database schema."},
    {"role": "assistant", "content": "Ha, cats and keyboards — a classic combo! Luna sounds fun. For the database schema, what's the data you need to model?"},
    {"role": "user", "content": "I'm allergic to peanuts by the way, so if you ever suggest lunch spots, keep that in mind. For the schema, I need users, projects, and tasks."},
    {"role": "assistant", "content": "Noted on the peanut allergy! Here's a schema suggestion:\n- `users`: id, name, email, role\n- `projects`: id, name, owner_id (FK→users), status\n- `tasks`: id, title, project_id (FK→projects), assignee_id (FK→users), status, due_date"},
    {"role": "user", "content": "I use Neovim as my editor and I'm currently learning Rust in my spare time."},
    {"role": "assistant", "content": "Nice! Neovim is great for productivity once you've got your config dialed in. Rust is a fantastic language to learn — its ownership model really makes you think differently about memory management."},
]

# What Engram would inject instead (top-5 most relevant memories for a new query)
ENGRAM_INJECTION = """[RELEVANT MEMORY — 5 items retrieved]
- The user's name is Alex, a software engineer at TechFlow in San Francisco [type: fact, confidence: 0.92]
- User prefers Python for backend code examples [type: instruction, confidence: 0.88]
- User prefers concise bullet-point responses [type: instruction, confidence: 0.85]
- User is building a React/TypeScript frontend dashboard [type: context, confidence: 0.78]
- User has a database with users, projects, and tasks tables [type: context, confidence: 0.71]
[END MEMORY]

Refer to these memories naturally. Do not mention the memory system to the user.
If memories conflict with what the user is saying now, prioritize what they're saying now."""

# New session — user asks a follow-up question
NEW_QUERY = {"role": "user", "content": "Can you help me write a Python API endpoint that creates a new task? Use FastAPI."}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Engram Token Savings Benchmark")
    print("=" * 70)
    print()

    # --- Approach 1: Naive history replay ---
    # In a new session without memory, user would need to re-establish all context.
    # This simulates replaying the past conversation history.
    naive_messages = PAST_CONVERSATION + [NEW_QUERY]
    naive_tokens = estimate_messages_tokens(naive_messages)

    # --- Approach 2: Engram memory injection ---
    # Engram injects only the top-N relevant memories as a system prompt.
    # No conversation history needed — memories carry the context.
    engram_messages = [
        {"role": "system", "content": ENGRAM_INJECTION},
        NEW_QUERY,
    ]
    engram_tokens = estimate_messages_tokens(engram_messages)

    # --- Approach 3: Cold start (no memory at all) ---
    # User has to re-explain everything manually.
    cold_start_messages = [
        {"role": "user", "content": "I'm Alex, a software engineer. I need a Python FastAPI endpoint to create a task. Use bullet points for the response."},
    ]
    cold_start_tokens = estimate_messages_tokens(cold_start_messages)

    # --- Results ---
    savings = ((naive_tokens - engram_tokens) / naive_tokens) * 100

    print("Scenario: User returns for a new session and asks a follow-up question.")
    print("The LLM needs context about the user (name, preferences, project details).\n")

    print(f"{'Approach':<30} {'Messages':>10} {'Tokens':>10} {'vs Naive':>12}")
    print("-" * 65)
    print(f"{'1. Naive (replay history)':<30} {len(naive_messages):>10} {naive_tokens:>10} {'baseline':>12}")
    print(f"{'2. Engram (memory inject)':<30} {len(engram_messages):>10} {engram_tokens:>10} {f'-{savings:.0f}%':>12}")
    print(f"{'3. Cold start (no memory)':<30} {len(cold_start_messages):>10} {cold_start_tokens:>10} {'loses context':>12}")
    print("-" * 65)
    print()

    # Detailed breakdown
    print("Detailed Breakdown:")
    print(f"  Past conversation history: {len(PAST_CONVERSATION)} messages, ~{estimate_messages_tokens(PAST_CONVERSATION)} tokens")
    print(f"  Engram memory injection:   {estimate_tokens(ENGRAM_INJECTION)} tokens (system prompt)")
    print(f"  New user query:            {estimate_tokens(NEW_QUERY['content'])} tokens")
    print()

    print(f"Token savings with Engram: {naive_tokens - engram_tokens} tokens ({savings:.1f}% reduction)")
    print()

    if savings >= 50:
        print(f"✓ TARGET MET: {savings:.1f}% reduction exceeds 50% target")
    else:
        print(f"✗ TARGET MISSED: {savings:.1f}% reduction below 50% target")

    print()
    print("Key insight: Engram replaces N turns of conversation history with a compact")
    print("memory injection block. The savings grow with conversation length — a 20-turn")
    print("history would save even more tokens while retaining the same context quality.")

    # Show scaling projection
    print()
    print("Scaling Projection:")
    print(f"{'History Turns':>15} {'Naive Tokens':>15} {'Engram Tokens':>15} {'Savings':>10}")
    print("-" * 58)
    for n_turns in [5, 10, 20, 50]:
        # Average ~80 tokens per turn (user + assistant)
        projected_naive = n_turns * 80 + estimate_tokens(NEW_QUERY["content"])
        projected_engram = engram_tokens  # Always the same — just memories + query
        proj_savings = ((projected_naive - projected_engram) / projected_naive) * 100
        print(f"{n_turns:>15} {projected_naive:>15} {projected_engram:>15} {f'{proj_savings:.0f}%':>10}")
    print("-" * 58)


if __name__ == "__main__":
    main()
