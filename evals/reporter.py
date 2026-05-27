import os
from datetime import datetime

from evals.grader import GradeResult, Verdict
from evals.metrics import EvalMetrics


def print_results_table(results: list[GradeResult], categories: list[str]) -> None:
    """Prints a detailed per-question results table to console."""

    icons = {Verdict.CORRECT: "✅", Verdict.INCORRECT: "❌", Verdict.NOT_ATTEMPTED: "—"}

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    current_category = None
    for i, (result, cat) in enumerate(zip(results, categories)):
        if cat != current_category:
            print(f"\n── {cat.upper()} ──")
            current_category = cat

        icon = icons[result.verdict]
        q_short = result.question[:55] + "..." if len(result.question) > 55 else result.question
        print(f"  {icon} {q_short}")
        if result.verdict != Verdict.CORRECT:
            print(f"     Gold:      {result.gold_answer[:80]}")
            print(f"     Reasoning: {result.reasoning[:80]}")


def print_metrics_summary(metrics: EvalMetrics) -> None:
    """Prints the full metrics dashboard to console."""

    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    print(f"\n📊 OVERALL RESULTS ({metrics.total_questions} questions)")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'F1 Score (primary)':<30} {metrics.f1_score:>10.3f}")
    print(f"  {'Accuracy (when attempted)':<30} {metrics.accuracy:>10.3f}")
    print(f"  {'Correct Rate':<30} {metrics.correct_rate:>10.3f}")
    print(f"  {'Answer Rate':<30} {metrics.answer_rate:>10.3f}")
    print(f"  {'Not Attempted Rate':<30} {metrics.not_attempted_rate:>10.3f}")
    print(f"  {'Incorrect Rate':<30} {metrics.incorrect_rate:>10.3f}")

    print(f"\n  {'Counts':<30} {'Value':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'Correct':<30} {metrics.total_correct:>10}")
    print(f"  {'Incorrect':<30} {metrics.total_incorrect:>10}")
    print(f"  {'Not Attempted':<30} {metrics.total_not_attempted:>10}")
    print(f"  {'Total':<30} {metrics.total_questions:>10}")

    print(f"\n  {'Costs':<30} {'Value':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'Total Cost':<30} ${metrics.total_cost:>9.4f}")
    print(f"  {'Avg Cost Per Query':<30} ${metrics.avg_cost_per_query:>9.4f}")

    if metrics.by_category:
        print(f"\n📋 BY CATEGORY")
        print(f"  {'Category':<25} {'F1':>7} {'Correct':>8} {'Total':>7}")
        print(f"  {'-' * 49}")
        for cat, counts in sorted(metrics.by_category.items()):
            print(f"  {cat:<25} {counts['f1']:>7.3f} "
                  f"{counts['correct']:>8} {counts['total']:>7}")

    print(f"\n{'=' * 80}")

    # Interpretation guide
    f1 = metrics.f1_score
    if f1 >= 0.80:
        interpretation = "🟢 EXCELLENT — system is highly reliable"
    elif f1 >= 0.65:
        interpretation = "🟡 GOOD — solid baseline, room to improve"
    elif f1 >= 0.50:
        interpretation = "🟠 FAIR — retrieval or prompts need tuning"
    else:
        interpretation = "🔴 NEEDS WORK — check retriever and context pipeline"

    print(f"\n  Interpretation: {interpretation}")
    print(f"  {metrics.summary_line()}")
    print()


def save_markdown_report(
    results: list[GradeResult],
    categories: list[str],
    metrics: EvalMetrics,
    output_path: str = "evals/results",
) -> str:
    """Saves a full markdown evaluation report."""

    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{output_path}/eval_report_{timestamp}.md"

    icons = {Verdict.CORRECT: "✅", Verdict.INCORRECT: "❌", Verdict.NOT_ATTEMPTED: "—"}

    lines = [
        "# Evaluation Report",
        f"*Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}*",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **F1 Score** | **{metrics.f1_score:.3f}** |",
        f"| Accuracy (when attempted) | {metrics.accuracy:.3f} |",
        f"| Correct Rate | {metrics.correct_rate:.3f} |",
        f"| Answer Rate | {metrics.answer_rate:.3f} |",
        f"| Not Attempted Rate | {metrics.not_attempted_rate:.3f} |",
        f"| Total Questions | {metrics.total_questions} |",
        f"| Correct | {metrics.total_correct} |",
        f"| Incorrect | {metrics.total_incorrect} |",
        f"| Not Attempted | {metrics.total_not_attempted} |",
        f"| Total Cost | ${metrics.total_cost:.4f} |",
        f"| Avg Cost/Query | ${metrics.avg_cost_per_query:.4f} |",
        "",
        "## Results by Category",
        "",
        "| Category | F1 | Correct | Total |",
        "|----------|----|---------|-------|",
    ]

    for cat, counts in sorted(metrics.by_category.items()):
        lines.append(f"| {cat} | {counts['f1']:.3f} | {counts['correct']} | {counts['total']} |")

    lines += ["", "## Detailed Results", ""]

    current_cat = None
    for result, cat in zip(results, categories):
        if cat != current_cat:
            lines.append(f"### {cat.replace('_', ' ').title()}")
            lines.append("")
            current_cat = cat

        icon = icons[result.verdict]
        lines.append(f"**{icon} {result.question}**")
        lines.append(f"- Gold: {result.gold_answer}")
        if result.verdict != Verdict.CORRECT:
            lines.append(f"- Predicted: {result.predicted_answer[:200]}")
            lines.append(f"- Reasoning: {result.reasoning}")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 Markdown report saved: {filepath}")
    return filepath
