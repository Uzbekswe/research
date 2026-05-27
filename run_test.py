import asyncio

from dotenv import load_dotenv

load_dotenv()

from researcher import DeepResearcher  # noqa: E402


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

    out = "report.md"
    with open(out, "w") as f:
        f.write(report)
    print(f"Full report saved to {out} ({len(report)} chars)")
    print()
    print(report)
    print(f"\nTotal cost: ${researcher.get_costs():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
