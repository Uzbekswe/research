# Before running: copy .env.example to .env and fill in at minimum:
# OPENAI_API_KEY (for default LLM) and TAVILY_API_KEY or leave RETRIEVER=duckduckgo
#
# Quick start:
#   cp .env.example .env
#   # edit .env — add OPENAI_API_KEY at minimum
#   python run_test.py

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()  # load .env before anything reads environment variables

from researcher import DeepResearcher  # noqa: E402  (after dotenv)


async def main():
    researcher = DeepResearcher(
        query="What is retrieval augmented generation?",
        report_type="research_report",
    )

    print("conducting research...")
    context = await researcher.conduct_research()
    print(f"Sources found: {len(researcher.get_source_urls())}")
    print(f"Context length: ~{len(context) // 4} tokens")
    print()

    print("writing report...")
    report = await researcher.write_report()
    print(report[:500])
    print("...")
    print(f"\nTotal cost: ${researcher.get_costs():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
