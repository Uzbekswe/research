import logging

from orchestrator.agents.utils.llms import call_model
from orchestrator.agents.utils.views import log_research_progress
from orchestrator.state import DraftState

logger = logging.getLogger(__name__)


class ReviserAgent:
    """Rewrites a section draft based on reviewer feedback.

    After revision, control returns to the reviewer.  The loop continues
    until the reviewer returns ``None`` (approved) or the max-revision
    count guard in ReviewerAgent fires.
    """

    async def run(self, draft_state: DraftState) -> dict:
        from researcher.config import Config

        cfg = Config()
        task = draft_state["task"]
        topic = draft_state["topic"]
        draft = draft_state["draft"]
        review = draft_state["review"]
        revision_notes = draft_state.get("revision_notes", "")

        log_research_progress(topic, "revising")

        prompt = f"""You are a research writer revising a section draft.

Section topic: {topic}

Current draft:
{draft}

Reviewer feedback (you MUST address all points):
{review}

Instructions:
- Rewrite the draft to address ALL reviewer feedback
- Keep all correct content from the original draft
- Do not remove information that was not criticized
- Maintain the same approximate length or longer
- Write in formal academic style
- Keep all source citations that were in the original

Revised draft:"""

        revised_draft = await call_model(
            prompt=prompt,
            model=task.get("model", cfg.SMART_LLM),
            max_tokens=cfg.SMART_TOKEN_LIMIT,
            temperature=0.4,
        )

        # Append this round's feedback to the running revision history.
        updated_notes = revision_notes + f"\nREVISION:\n{review}\n"

        return {
            "draft": revised_draft,
            "revision_notes": updated_notes,
            "review": None,  # reset so the reviewer evaluates the new draft
        }
