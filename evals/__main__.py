import argparse
import asyncio

from evals.metrics import compute_metrics
from evals.reporter import print_metrics_summary, print_results_table, save_markdown_report
from evals.runner import run_evaluation


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep Researcher factual accuracy"
    )
    parser.add_argument(
        "--num_examples", type=int, default=10,
        help="Number of questions to evaluate (default: 10, max: 25)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=["ai_ml", "software_engineering", "science_facts",
                 "history_geography", "tech_companies"],
        help="Filter to specific categories",
    )
    parser.add_argument(
        "--concurrent", type=int, default=2,
        help="Max concurrent researcher calls (default: 2, keep low for cost)",
    )
    parser.add_argument(
        "--save_report", action="store_true",
        help="Save markdown report to evals/results/",
    )
    args = parser.parse_args()

    print("\n🧪 DEEP RESEARCHER EVALUATION")
    print(f"   Questions: {args.num_examples}")
    print(f"   Categories: {args.categories or 'all'}")
    print(f"   Concurrent: {args.concurrent}")

    results, categories = await run_evaluation(
        num_examples=args.num_examples,
        categories=args.categories,
        max_concurrent=args.concurrent,
    )

    if not results:
        print("❌ No results collected")
        return

    metrics = compute_metrics(results, categories)

    print_results_table(results, categories)
    print_metrics_summary(metrics)

    if args.save_report:
        save_markdown_report(results, categories, metrics)


if __name__ == "__main__":
    asyncio.run(main())
