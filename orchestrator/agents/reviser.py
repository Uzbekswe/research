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

        # OPUS FIX (3B): use the centralised prompt template instead of an inline string.
        from researcher.prompts import get_section_revise_prompt
        prompt = get_section_revise_prompt(topic=topic, draft=draft, review=review)

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
