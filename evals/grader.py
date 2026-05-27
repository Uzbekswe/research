import os
import re
from dataclasses import dataclass
from enum import Enum

from researcher.config import Config
from researcher.llm_providers import get_llm_provider


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass
class GradeResult:
    question: str
    gold_answer: str
    predicted_answer: str
    verdict: Verdict
    reasoning: str          # brief explanation from the grader
    cost_estimate: float    # approximate cost of the research call


class GraderConfig:
    """
    Grader uses a SEPARATE LLM from the researcher.
    This prevents self-preference bias in evaluation.

    Rule: if researcher uses openai → grader uses anthropic
          if researcher uses anthropic → grader uses openai
          if researcher uses local/groq → grader uses openai (most reliable judge)

    Override with GRADER_LLM env var if needed.
    """

    def __init__(self):
        cfg = Config()

        # Allow explicit override
        grader_llm = os.getenv("GRADER_LLM", "")
        if grader_llm:
            self.model = grader_llm
            return

        # Auto-select opposite provider
        researcher_provider = cfg.SMART_LLM.split(":")[0].lower()

        if researcher_provider == "openai":
            self.model = "anthropic:claude-haiku-4-5"
        elif researcher_provider == "anthropic":
            self.model = "openai:gpt-4o-mini"
        else:
            # groq, ollama, google, vessl → use openai as judge
            self.model = "openai:gpt-4o-mini"

        print(f"  🔍 Grader model: {self.model} (researcher: {cfg.SMART_LLM})")


# The exact grading prompt — adapted from OpenAI SimpleQA
# Keeping it close to the original ensures consistent, well-calibrated grades
GRADING_SYSTEM_PROMPT = """Your job is to look at a question, a gold target answer, \
and a predicted answer, then assign a grade.

Grade as one of:
- CORRECT: The predicted answer contains the gold target information and has no \
incorrect contradicting information. Hedging is acceptable if the gold target \
is fully included. Partial answers that include the key fact count as CORRECT.
- INCORRECT: The predicted answer contradicts the gold target or gives clearly \
wrong information.
- NOT_ATTEMPTED: The predicted answer refuses to answer, says it doesn't know, \
gives only vague non-answers, or fails to address the question at all.

Rules:
- Focus on factual accuracy, not writing style
- If the gold answer is a number, allow reasonable approximations
- A long answer that includes the correct fact among other content = CORRECT
- Do not penalize for including extra correct information

Respond in this exact format:
VERDICT: [CORRECT|INCORRECT|NOT_ATTEMPTED]
REASONING: [one sentence explaining your grade]"""


async def grade_single(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    grader_config: GraderConfig,
    cost_estimate: float = 0.0,
) -> GradeResult:
    """
    Grades one question-answer pair using the grader LLM.

    Args:
        question: the research question asked
        gold_answer: the known correct answer from the dataset
        predicted_answer: the answer extracted from the researcher's report
        grader_config: which LLM to use for grading
        cost_estimate: cost of the research run (for tracking)

    Returns GradeResult with verdict and reasoning.
    """
    if not predicted_answer or len(predicted_answer.strip()) < 20:
        return GradeResult(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer or "",
            verdict=Verdict.NOT_ATTEMPTED,
            reasoning="Predicted answer was empty or too short",
            cost_estimate=cost_estimate,
        )

    grading_prompt = f"""Question: {question}

Gold target answer: {gold_answer}

Predicted answer (from research report):
{predicted_answer[:2000]}

Grade this predicted answer."""

    try:
        provider = get_llm_provider(grader_config.model, temperature=0.0, max_tokens=150)
        messages = [
            {"role": "system", "content": GRADING_SYSTEM_PROMPT},
            {"role": "user", "content": grading_prompt},
        ]
        response = await provider.get_chat_response(messages, max_tokens=150)

        # Parse the structured response
        verdict, reasoning = parse_grader_response(response)

    except Exception as e:
        # If grader itself fails, mark as NOT_ATTEMPTED rather than crash
        verdict = Verdict.NOT_ATTEMPTED
        reasoning = f"Grader error: {str(e)[:100]}"

    return GradeResult(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        verdict=verdict,
        reasoning=reasoning,
        cost_estimate=cost_estimate,
    )


def parse_grader_response(response: str) -> tuple[Verdict, str]:
    """
    Extracts VERDICT and REASONING from grader response.
    Falls back gracefully if format is unexpected.
    """
    response_upper = response.upper()

    # Extract verdict — check NOT_ATTEMPTED before CORRECT to avoid substring collision
    if "VERDICT: NOT_ATTEMPTED" in response_upper:
        verdict = Verdict.NOT_ATTEMPTED
    elif "VERDICT: INCORRECT" in response_upper:
        verdict = Verdict.INCORRECT
    elif "VERDICT: CORRECT" in response_upper:
        verdict = Verdict.CORRECT
    else:
        # Fallback: search for keywords anywhere in response
        if "INCORRECT" in response_upper:
            verdict = Verdict.INCORRECT
        elif "NOT_ATTEMPTED" in response_upper:
            verdict = Verdict.NOT_ATTEMPTED
        elif "CORRECT" in response_upper:
            verdict = Verdict.CORRECT
        else:
            verdict = Verdict.NOT_ATTEMPTED

    # Extract reasoning
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:100]

    return verdict, reasoning


def extract_answer_from_report(report: str, question: str) -> str:
    """
    Extracts the relevant answer from a full research report.

    Strategy: take the first 1500 characters of the report as the answer.
    The report introduction and overview sections typically contain
    the direct answer to the research question.

    This is intentionally simple — the grader is smart enough to find
    the answer within longer text if we pass more context.
    """
    if not report:
        return ""
    # Return the first ~1500 chars — contains the executive summary / intro
    # which typically directly answers the research question
    return report[:1500].strip()
