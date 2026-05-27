import logging

from orchestrator.agents.utils.llms import call_model
from orchestrator.agents.utils.views import log_research_progress
from orchestrator.state import DraftState

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Validates a section draft against the task guidelines.

    Return value of ``run()`` drives the conditional edge in the sub-graph:
      ``{"review": None}``   → END (approved)
      ``{"review": <str>}``  → reviser node (needs revision)
    """

    async def run(self, draft_state: DraftState) -> dict:
        from researcher.config import Config

        cfg = Config()
        task = draft_state["task"]
        topic = draft_state["topic"]
        draft = draft_state["draft"]
        guidelines = draft_state.get("guidelines", [])
        revision_notes = draft_state.get("revision_notes", "")

        log_research_progress(topic, "reviewing")

        # No guidelines configured → auto-approve
        if not task.get("follow_guidelines", True) or not guidelines:
            return {"review": None}

        # Empty or too-short draft → immediate revision request
        if not draft or len(draft.strip()) < 100:
            return {
                "review": "Draft is empty or too short. Please write a complete section."
            }

        # Guard against infinite revision loops
        revision_count = len([r for r in revision_notes.split("REVISION") if r.strip()])
        max_revisions = task.get("max_revisions", 2)
        if revision_count >= max_revisions:
            log_research_progress(
                topic, "approved", f"max revisions ({max_revisions}) reached"
            )
            return {"review": None}

        guidelines_text = "\n".join(f"- {g}" for g in guidelines)

        prompt = f"""You are a strict research editor reviewing a draft section.

Section topic: {topic}

Quality guidelines that MUST be met:
{guidelines_text}

Draft to review:
{draft[:3000]}

Previous revision notes (if any):
{revision_notes[:500] if revision_notes else "None"}

Your task:
1. Check if the draft meets EVERY guideline listed above
2. If it meets all guidelines: respond with exactly the word: APPROVED
3. If it fails any guideline: respond with specific, actionable revision notes

Be specific. Say exactly what is missing and how to fix it.
If only minor issues remain, approve anyway (perfect is the enemy of good).

Your response (APPROVED or revision notes):"""

        response = await call_model(
            prompt=prompt,
            model=task.get("model", cfg.SMART_LLM),
            max_tokens=500,
            temperature=0.2,
        )

        if "APPROVED" in response.upper()[:20]:
            log_research_progress(topic, "approved")
            return {"review": None}
        else:
            log_research_progress(topic, "revising", response[:80])
            return {"review": response}
