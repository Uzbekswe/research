from dataclasses import dataclass
from typing import Optional

from evals.grader import GradeResult, Verdict


@dataclass
class EvalMetrics:
    # Counts
    total_questions: int
    total_attempted: int        # CORRECT + INCORRECT
    total_correct: int
    total_incorrect: int
    total_not_attempted: int

    # Rates (out of all questions)
    correct_rate: float         # correct / total
    incorrect_rate: float       # incorrect / total
    not_attempted_rate: float   # not_attempted / total
    answer_rate: float          # attempted / total

    # Quality (out of attempted only)
    accuracy: float             # correct / attempted (precision when it tries)
    precision: float            # same as accuracy in this context
    recall: float               # correct / total (same as correct_rate)
    f1_score: float             # harmonic mean of precision and recall

    # Cost
    total_cost: float
    avg_cost_per_query: float

    # By category
    by_category: dict           # {category: {"correct": int, "total": int, "f1": float}}

    def summary_line(self) -> str:
        return (
            f"F1: {self.f1_score:.3f} | "
            f"Accuracy: {self.accuracy:.3f} | "
            f"Answer Rate: {self.answer_rate:.3f} | "
            f"Not Attempted: {self.not_attempted_rate:.3f} | "
            f"Cost: ${self.total_cost:.4f}"
        )


def compute_metrics(
    results: list[GradeResult],
    # OPUS FIX: list[str] | None matches the actual nullable default.
    categories: list[str] | None = None,
) -> EvalMetrics:
    """
    Computes all evaluation metrics from a list of graded results.

    Formula explanations (important for interviews):

    correct_rate  = correct / total
      → "What fraction of ALL questions did we answer correctly?"
      → Primary quality signal. Low = system is bad or refuses a lot.

    accuracy      = correct / (correct + incorrect)
      → "When the system commits to an answer, how often is it right?"
      → High accuracy + low answer_rate = cautious but reliable system
      → Low accuracy = system hallucinates when it tries

    f1_score      = 2 * (precision * recall) / (precision + recall)
      → Harmonic mean. Punishes both extremes:
      →   Always answers wrong: recall=0 → F1=0
      →   Never answers: precision undefined, recall=0 → F1=0
      →   Ideal: answers correctly as often as possible → F1 approaches 1.0

    not_attempted_rate = not_attempted / total
      → "How often does the system refuse or fail to find information?"
      → High = system is too cautious or retrieval is failing
      → Some NOT_ATTEMPTED is healthy — better than hallucinating
    """

    total = len(results)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty results list")

    correct = sum(1 for r in results if r.verdict == Verdict.CORRECT)
    incorrect = sum(1 for r in results if r.verdict == Verdict.INCORRECT)
    not_attempted = sum(1 for r in results if r.verdict == Verdict.NOT_ATTEMPTED)
    attempted = correct + incorrect

    def safe_div(a, b): return a / b if b > 0 else 0.0

    correct_rate = safe_div(correct, total)
    incorrect_rate = safe_div(incorrect, total)
    not_attempted_rate = safe_div(not_attempted, total)
    answer_rate = safe_div(attempted, total)
    accuracy = safe_div(correct, attempted)
    precision = accuracy  # same formula in this context
    recall = correct_rate

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    total_cost = sum(r.cost_estimate for r in results)
    avg_cost = safe_div(total_cost, total)

    # Compute per-category breakdown if categories provided
    by_category = {}
    if categories:
        for i, result in enumerate(results):
            cat = categories[i] if i < len(categories) else "unknown"
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0, "attempted": 0}
            by_category[cat]["total"] += 1
            if result.verdict == Verdict.CORRECT:
                by_category[cat]["correct"] += 1
            if result.verdict != Verdict.NOT_ATTEMPTED:
                by_category[cat]["attempted"] += 1

        # Compute F1 per category
        for cat, counts in by_category.items():
            cat_precision = safe_div(counts["correct"], counts["attempted"])
            cat_recall = safe_div(counts["correct"], counts["total"])
            if cat_precision + cat_recall > 0:
                counts["f1"] = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall)
            else:
                counts["f1"] = 0.0

    return EvalMetrics(
        total_questions=total,
        total_attempted=attempted,
        total_correct=correct,
        total_incorrect=incorrect,
        total_not_attempted=not_attempted,
        correct_rate=correct_rate,
        incorrect_rate=incorrect_rate,
        not_attempted_rate=not_attempted_rate,
        answer_rate=answer_rate,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        total_cost=total_cost,
        avg_cost_per_query=avg_cost,
        by_category=by_category,
    )
