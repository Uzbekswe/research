"""Benchmark script — measures real Layer 2 research + report performance.

Run from the project root:
    RETRIEVER=duckduckgo .venv/bin/python benchmark/compare_layers.py

Requires the VESSL SSH tunnel to be active (or any configured LLM provider).
Saves the full report to benchmark_report.md.
"""

import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

from researcher import DeepResearcher  # noqa: E402


async def benchmark() -> None:
    query = "What are the main differences between RAG and fine-tuning for LLMs?"

    print("=" * 60)
    print(f"Benchmarking: {query}")
    print("=" * 60)

    # ── Research phase ────────────────────────────────────────────────
    start = time.time()
    researcher = DeepResearcher(query=query)
    context = await researcher.conduct_research()
    research_time = time.time() - start

    # ── Report phase ──────────────────────────────────────────────────
    start = time.time()
    report = await researcher.write_report()
    write_time = time.time() - start

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n📊 BENCHMARK RESULTS")
    print(f"  Research time:    {research_time:.2f}s")
    print(f"  Sources found:    {len(researcher.get_source_urls())}")
    print(f"  Context tokens:   ~{len(context) // 4}")
    print(f"  Report time:      {write_time:.2f}s")
    print(f"  Report length:    {len(report)} chars")
    print(f"  Total cost:       ${researcher.get_costs():.4f}")
    print(f"  Total time:       {research_time + write_time:.2f}s")

    out = "benchmark_report.md"
    with open(out, "w") as f:
        f.write(report)
    print(f"\n✅ Report saved to {out}")


if __name__ == "__main__":
    asyncio.run(benchmark())
