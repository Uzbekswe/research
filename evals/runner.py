import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evals.grader import (
    GradeResult,
    GraderConfig,
    Verdict,
    extract_answer_from_report,
    grade_single,
)
from researcher.config import Config


@dataclass
class EvalQuestion:
    question: str
    gold_answer: str
    category: str


def load_dataset(csv_path: str = "evals/dataset/eval_questions.csv") -> list[EvalQuestion]:
    """Loads evaluation questions from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(EvalQuestion(
                question=row["question"].strip(),
                gold_answer=row["gold_answer"].strip(),
                category=row["category"].strip(),
            ))

    print(f"📂 Loaded {len(questions)} questions from {csv_path}")
    return questions


async def evaluate_single_question(
    eval_q: EvalQuestion,
    grader_config: GraderConfig,
    semaphore: asyncio.Semaphore,
    question_num: int,
    total: int,
) -> GradeResult:
    """
    Runs one full evaluation cycle:
    1. Call DeepResearcher with the question
    2. Extract answer from report
    3. Grade with LLM-as-judge

    Wrapped in semaphore to limit concurrent researcher calls.
    """
    async with semaphore:
        print(f"  [{question_num}/{total}] {eval_q.question[:60]}...")

        start = time.time()
        report = ""
        cost = 0.0

        try:
            from researcher import DeepResearcher

            researcher = DeepResearcher(
                query=eval_q.question,
                report_type="research_report",
            )
            await researcher.conduct_research()
            report = await researcher.write_report()
            cost = researcher.get_costs()

        except Exception as e:
            print(f"    ⚠️ Researcher failed: {e}")
            return GradeResult(
                question=eval_q.question,
                gold_answer=eval_q.gold_answer,
                predicted_answer="",
                verdict=Verdict.NOT_ATTEMPTED,
                reasoning=f"Researcher error: {str(e)[:100]}",
                cost_estimate=cost,
            )

        elapsed = time.time() - start
        predicted_answer = extract_answer_from_report(report, eval_q.question)

        grade = await grade_single(
            question=eval_q.question,
            gold_answer=eval_q.gold_answer,
            predicted_answer=predicted_answer,
            grader_config=grader_config,
            cost_estimate=cost,
        )

        verdict_icons = {"CORRECT": "✅", "INCORRECT": "❌", "NOT_ATTEMPTED": "—"}
        icon = verdict_icons.get(grade.verdict.value, "?")
        print(f"    {icon} {grade.verdict.value} ({elapsed:.1f}s) — {grade.reasoning[:60]}")

        return grade


async def run_evaluation(
    num_examples: int = 25,
    # OPUS FIX: list[str] | None matches the actual nullable default.
    categories: list[str] | None = None,
    max_concurrent: int = 3,
    dataset_path: str = "evals/dataset/eval_questions.csv",
    output_path: str = "evals/results",
) -> tuple[list[GradeResult], list[str]]:
    """
    Main evaluation runner.

    Args:
        num_examples: how many questions to evaluate (use 5-10 during dev)
        categories: filter to specific categories (None = all)
        max_concurrent: max parallel researcher calls (keep low — expensive)
        dataset_path: path to eval_questions.csv
        output_path: where to save results JSON

    Returns: (results, categories_list) for metrics computation

    IMPORTANT: max_concurrent=3 means 3 researcher instances run at once.
    Each researcher uses LLM API calls. Keep this at 2-3 to control costs.
    """
    all_questions = load_dataset(dataset_path)

    # Filter by category if specified
    if categories:
        all_questions = [q for q in all_questions if q.category in categories]
        print(f"📋 Filtered to {len(all_questions)} questions in: {categories}")

    # Limit to num_examples
    questions = all_questions[:num_examples]

    grader_config = GraderConfig()
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"\n🚀 Starting evaluation: {len(questions)} questions, "
          f"max {max_concurrent} concurrent")
    print(f"   Researcher: {Config().SMART_LLM}")
    print(f"   Grader: {grader_config.model}")
    print("-" * 60)

    start_time = time.time()

    tasks = [
        evaluate_single_question(q, grader_config, semaphore, i + 1, len(questions))
        for i, q in enumerate(questions)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert unexpected exceptions to NOT_ATTEMPTED rather than crashing
    clean_results: list[GradeResult] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"⚠️ Unexpected error: {r}")
        else:
            clean_results.append(r)

    elapsed = time.time() - start_time
    categories_list = [q.category for q in questions[:len(clean_results)]]

    print(f"\n✅ Evaluation complete in {elapsed:.1f}s")
    print(f"   Graded {len(clean_results)}/{len(questions)} questions")

    # Save raw results to JSON
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_path}/eval_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump([{
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer[:500],
            "verdict": r.verdict.value,
            "reasoning": r.reasoning,
            "cost": r.cost_estimate,
            "category": categories_list[i] if i < len(categories_list) else "unknown",
        } for i, r in enumerate(clean_results)], f, indent=2)

    print(f"   Raw results saved: {results_file}")

    return clean_results, categories_list
