"""Layer 3 comparison — shows how embedding-based filtering reduces context size.

Run from project root (VESSL tunnel must be active):
    RETRIEVER=duckduckgo .venv/bin/python benchmark/layer3_comparison.py
"""

import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

from researcher import DeepResearcher  # noqa: E402


async def compare() -> None:
    query = "How does transformer attention mechanism work?"

    print("=" * 60)
    print(f"Layer 3 RAG Comparison: {query}")
    print("=" * 60)

    researcher = DeepResearcher(query=query)

    start = time.time()
    await researcher.conduct_research()
    research_time = time.time() - start

    vector_store = researcher.get_vector_store()
    store_stats = vector_store.get_stats() if vector_store else {"total_chunks": 0}
    context = researcher.get_research_context()
    raw_sources = researcher.get_research_sources()

    raw_token_estimate = sum(len(s.get("content", "")) // 4 for s in raw_sources)
    filtered_token_estimate = len(context) // 4
    reduction_pct = (
        (1 - filtered_token_estimate / max(raw_token_estimate, 1)) * 100
    )

    print(f"\n📊 LAYER 3 FILTERING RESULTS")
    print(f"  Research time:           {research_time:.1f}s")
    print(f"  Sources found:           {len(researcher.get_source_urls())}")
    print(f"  Sub-queries used:        {len(researcher.get_sub_queries())}")
    print(f"  Total chunks in store:   {store_stats['total_chunks']}")
    print(f"  Raw content tokens:      ~{raw_token_estimate}")
    print(f"  Filtered context tokens: ~{filtered_token_estimate}")
    print(f"  Reduction:               {reduction_pct:.0f}%")
    print(f"  Similarity threshold:    {researcher.cfg.SIMILARITY_THRESHOLD}")
    print(f"  Max context chunks:      {researcher.cfg.MAX_CONTEXT_CHUNKS}")

    if reduction_pct >= 40:
        print(f"\n✅ Layer 3 filtering achieved {reduction_pct:.0f}% token reduction")
    elif filtered_token_estimate == 0:
        print("\n⚠️  No context retrieved — check EMBEDDING config and tunnel")
    else:
        print(f"\n ℹ️  Reduction was {reduction_pct:.0f}% (expected 40–70% for typical queries)")

    if context:
        preview = context[:400].replace("\n", " ")
        print(f"\n--- Context preview (first 400 chars) ---")
        print(preview)
        print("...")


if __name__ == "__main__":
    asyncio.run(compare())
