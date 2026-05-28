"""
All LLM prompt templates for the deep-researcher system.

Each function returns a fully-formatted prompt string ready to be sent to an LLM.
No business logic lives here — only string construction.

Call sites:
  - get_agent_role_prompt      → system message for every LLM call
  - get_search_queries_prompt  → STRATEGIC_LLM, planning phase
  - generate_report_prompt     → SMART_LLM, final report writing
  - get_summarize_prompt       → FAST_LLM, per-source summarisation
  - get_report_by_type         → dispatcher used by the researcher agent
  - get_retrievers              → config helper, not an LLM prompt
"""

import re
from datetime import date, datetime, timezone
from typing import Callable


def _has_keyword(query_lower: str, keywords: tuple[str, ...]) -> bool:
    """Return True if any keyword matches a whole word in query_lower."""
    return any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in keywords)


# ---------------------------------------------------------------------------
# Agent role / persona
# ---------------------------------------------------------------------------

def get_agent_role_prompt(query: str) -> str:
    """Return a system prompt that defines the researcher's persona for a query.

    The domain is inferred from keywords in the query so that the LLM adopts a
    more appropriate expert voice (finance, tech, science, etc.).  Falls back to
    a general research analyst persona when no domain is detected.

    Called once per research session as the system message for all LLM calls.
    """
    q = query.lower()

    if _has_keyword(q, (
        "stock", "invest", "finance", "market", "crypto", "bitcoin", "etf",
        "portfolio", "bond", "dividend", "trading", "hedge", "forex", "valuation",
        "revenue", "earnings", "ipo", "venture capital",
    )):
        return (
            "You are a seasoned financial analyst AI assistant. "
            "Your primary goal is to compose comprehensive, astute, impartial, and "
            "methodically arranged financial reports based on provided data and trends. "
            "Cite statistics, valuations, and risk factors precisely."
        )

    if _has_keyword(q, (
        "software", "code", "programming", "ai", "machine learning", "llm", "api",
        "cloud", "saas", "startup", "framework", "algorithm", "neural", "deep learning",
        "gpt", "gpu", "chip", "semiconductor", "cybersecurity", "blockchain",
        "langchain", "langgraph", "transformer", "model",
    )):
        return (
            "You are an expert technology research analyst AI assistant. "
            "Your primary goal is to produce thorough, technically accurate, and "
            "well-structured reports on technology topics, products, and trends. "
            "Ground your analysis in concrete benchmarks, architecture details, and industry data."
        )

    if _has_keyword(q, (
        "health", "medicine", "drug", "disease", "clinical", "biology", "genomic",
        "pharma", "gene", "genome", "mutation", "cancer", "virus", "treatment",
        "therapy", "crispr", "protein", "cell", "neuroscience", "vaccine",
    )):
        return (
            "You are a rigorous medical and life-sciences research AI assistant. "
            "Your primary goal is to synthesise peer-reviewed evidence into clear, "
            "accurate, and cautious reports. Always distinguish between established "
            "findings and preliminary or contested research."
        )

    if _has_keyword(q, (
        "science", "physics", "chemistry", "research", "study", "paper",
        "experiment", "data", "particle", "quantum", "atom", "molecule", "evolution",
        "astronomy", "climate", "geology", "thermodynamics",
    )):
        return (
            "You are a meticulous scientific research AI assistant. "
            "Your primary goal is to analyse and synthesise scientific literature into "
            "precise, well-structured reports that correctly represent methodology, "
            "findings, and limitations."
        )

    if _has_keyword(q, (
        "law", "legal", "regulation", "policy", "court", "legislation", "compliance",
        "gdpr", "hipaa", "rights", "contract", "liability", "patent", "copyright",
        "lawsuit", "statute", "regulatory", "act",
    )):
        return (
            "You are a knowledgeable legal and policy research AI assistant. "
            "Your primary goal is to provide accurate, well-cited analyses of laws, "
            "regulations, and policy developments. Always note jurisdictional context "
            "and avoid providing formal legal advice."
        )

    if _has_keyword(q, (
        "travel", "tourism", "country", "city", "visit", "culture", "destination",
        "hiking", "trails", "hotel", "flight", "visa", "itinerary", "sightseeing",
    )):
        return (
            "You are a world-travelled AI tour guide and travel research assistant. "
            "Your primary goal is to draft engaging, insightful, and well-structured "
            "travel reports covering history, attractions, practical tips, and cultural nuances."
        )

    return (
        "You are an expert research AI assistant. "
        "Your primary goal is to produce comprehensive, unbiased, well-structured, "
        "and evidence-based research reports on any topic. "
        "Prioritise accuracy, depth, and clarity. Always cite your sources."
    )


# ---------------------------------------------------------------------------
# Search query generation
# ---------------------------------------------------------------------------

def get_search_queries_prompt(
    query: str,
    parent_query: str,
    report_type: str,
    max_iterations: int,
    context: str = "",
) -> str:
    """Return a prompt that instructs the LLM to emit search queries as a JSON array.

    For sub-topic or detailed reports the task is framed as
    ``parent_query – query`` so the LLM understands the broader context.

    If *context* is provided (e.g. from a previous search pass) the LLM is
    asked to generate queries that fill gaps rather than repeat what is already
    known.

    Called by the planning phase using STRATEGIC_LLM.

    Args:
        query:          The specific question or sub-topic to research.
        parent_query:   The top-level research question (may equal query for
                        simple reports).
        report_type:    One of "research_report", "outline_report",
                        "resource_report", "subtopic_report".
        max_iterations: How many distinct search queries to generate.
        context:        Already-gathered context that the new queries should
                        complement rather than duplicate.

    Returns:
        A prompt string whose expected LLM output is a JSON array of strings,
        e.g. ``["query one", "query two", "query three"]``.
    """
    if report_type in ("subtopic_report", "detailed_report") and parent_query and parent_query != query:
        task = f"{parent_query} — {query}"
    else:
        task = query

    dynamic_example = ", ".join(f'"search query {i + 1}"' for i in range(max_iterations))

    context_block = ""
    if context:
        context_block = f"""
The following information has already been gathered. Generate queries that surface NEW information \
and fill gaps in what is already known — do not repeat topics already well-covered below.

Already gathered context:
\"\"\"
{context}
\"\"\"
"""

    return f"""You are a strategic research planner.

Your task is to write {max_iterations} distinct Google search queries that together build an \
objective, well-rounded understanding of the following research task:

Task: "{task}"

Today's date: {datetime.now(timezone.utc).strftime("%B %d, %Y")}
{context_block}
Guidelines:
- Each query should target a different angle (definition, recent news, expert opinion, data/statistics, criticism, etc.)
- Queries should be concise and specific — avoid vague or overly broad phrasing.
- Prefer queries that would surface authoritative sources (academic, government, industry reports).
- Do NOT number or label the queries; return only the JSON array.

You MUST respond with a JSON array of exactly {max_iterations} strings and nothing else:
[{dynamic_example}]
"""


# ---------------------------------------------------------------------------
# Report generation prompts
# ---------------------------------------------------------------------------

def generate_report_prompt(
    question: str,
    context: str,
    report_source: str,
    report_format: str,
    total_words: int,
    tone: str | None = None,
    language: str = "english",
) -> str:
    """Return the main report-generation prompt for SMART_LLM.

    The LLM must write a detailed, well-cited report using ONLY the supplied
    context.  Strong anti-hallucination instructions are included to prevent
    the model from adding information not present in the sources.

    Args:
        question:      The research question / task.
        context:       All scraped and summarised source material.
        report_source: "web" or "local" — controls how citations are formatted.
        report_format: Citation style, e.g. "APA", "MLA", "Chicago".
        total_words:   Minimum word count for the finished report.
        tone:          Optional tone descriptor (e.g. "objective", "critical").
        language:      Output language (default "english").

    Returns:
        A prompt string whose expected output is a complete Markdown report.
    """
    if report_source == "web":
        reference_instruction = (
            "You MUST list all source URLs used at the end of the report under a ## References section.\n"
            "Each URL must appear only once and be formatted as a hyperlink: [Page Title](url)\n"
            "You MUST also embed inline hyperlinks wherever a source is cited in the body, like this: "
            "([Author, Year](url))"
        )
    else:
        reference_instruction = (
            "You MUST list all source document names used at the end of the report under a ## References section.\n"
            "Each document should appear only once."
        )

    tone_instruction = f"Write the entire report in a {tone} tone.\n" if tone else ""

    return f"""You are writing a detailed research report.

RESEARCH QUESTION:
\"\"\"{question}\"\"\"

SOURCE MATERIAL:
\"\"\"{context}\"\"\"

━━━ STRICT INSTRUCTIONS ━━━

ACCURACY — CRITICAL:
• Use ONLY the information present in the SOURCE MATERIAL above.
• Do NOT add any facts, statistics, names, dates, or claims that are not explicitly stated in the sources.
• If the sources do not contain enough information to answer part of the question, clearly state that the information was not found in the available sources.
• Do NOT speculate, infer beyond what the sources say, or fill gaps with general knowledge.

STRUCTURE:
• Begin with a clear # Title.
• Use ## for major sections and ### for subsections.
• Do NOT include a Table of Contents.
• Do NOT add an introduction or conclusion section unless organically needed.

CITATION FORMAT ({report_format.upper()}):
• Use in-text citations placed at the end of the relevant sentence or paragraph.
• Format: ([Author or Site Name, Year](url))
• {reference_instruction}

CONTENT QUALITY:
• The report must be comprehensive, analytical, and evidence-based.
• Include specific facts, numbers, and statistics from the sources wherever available.
• Do not hedge every sentence — form a concrete, well-supported position.
• Minimum length: {total_words} words.

FORMATTING:
• Use Markdown tables when presenting comparisons or structured data.
• Use bullet lists sparingly — prefer flowing prose for analysis.
• {tone_instruction}

LANGUAGE: Write the entire report in {language}.

Today's date for reference: {date.today()}

Now write the report.
"""


def generate_outline_prompt(
    question: str,
    context: str,
    report_source: str = "web",
    report_format: str = "APA",
    total_words: int = 1200,
    tone: str | None = None,
    language: str = "english",
) -> str:
    """Return a prompt that produces only a structured Markdown outline.

    Used when the researcher needs to plan a long-form report before writing
    individual sections.  Called with STRATEGIC_LLM.

    Returns:
        A prompt whose expected output is a Markdown outline (headers only,
        with brief bullet notes under each section — no full prose).
    """
    return f"""You are a research report planner.

Based on the research material below, produce a detailed Markdown outline for a report that \
answers the following question:

QUESTION: "{question}"

RESEARCH MATERIAL:
\"\"\"{context}\"\"\"

OUTLINE REQUIREMENTS:
• Use # for the report title, ## for major sections, ### for subsections.
• Under each header add 2–4 bullet points summarising the key points that section will cover.
• The outline must be comprehensive enough that a writer could produce a {total_words}-word report from it.
• Do NOT write full paragraphs — only headers and bullet-point notes.
• Do NOT include a Table of Contents section.
• Language: {language}
• Today's date: {date.today()}

Output ONLY the Markdown outline.
"""


def generate_resource_report_prompt(
    question: str,
    context: str,
    report_source: str = "web",
    report_format: str = "APA",
    total_words: int = 1200,
    tone: str | None = None,
    language: str = "english",
) -> str:
    """Return a prompt that produces a curated bibliography / resource list.

    Rather than synthesising a narrative report, the LLM evaluates each source
    and explains its relevance, reliability, and what specific value it adds to
    the research question.

    Called with SMART_LLM when report_type == "resource_report".

    Returns:
        A prompt whose expected output is a Markdown resource-recommendation
        report with one entry per source.
    """
    if report_source == "web":
        url_instruction = (
            "Hyperlink every source title: [Source Title](url)"
        )
    else:
        url_instruction = "Refer to each source by its document filename."

    return f"""You are a research librarian and source evaluator.

RESEARCH QUESTION: "{question}"

AVAILABLE SOURCES:
\"\"\"{context}\"\"\"

Your task is to produce a bibliography recommendation report — a curated, annotated list of the \
most relevant and reliable sources for the research question above.

For EACH source include:
1. **Title / Name** — {url_instruction}
2. **Relevance** — How directly does this source address the research question?
3. **Reliability** — Author credentials, publication, date, potential bias.
4. **Key contribution** — What specific facts, data, or perspectives does it add?
5. **Limitations** — Any gaps, bias, or caveats the reader should know.

FORMATTING:
• Use ## for the report title and ### for each source entry.
• Use Markdown tables for a comparative summary at the top (source, type, date, relevance score 1–10).
• Write in {language}.
• Minimum length: {total_words} words.
• Today's date: {date.today()}

Use ONLY sources present in the AVAILABLE SOURCES block.  Do not recommend sources not listed there.

Now write the resource report.
"""


# ---------------------------------------------------------------------------
# Per-source summarisation
# ---------------------------------------------------------------------------

def get_summarize_prompt(query: str, raw_data: str) -> str:
    """Return a short summarisation prompt for FAST_LLM.

    This is called once per scraped source to compress raw page content down to
    only the information relevant to *query*.  The summary preserves all
    factual details (numbers, quotes, dates) while discarding boilerplate,
    navigation text, and off-topic material.

    The caller is responsible for prepending the source URL to the summary so
    that attribution is preserved through later processing steps.

    Args:
        query:    The research question the summary should serve.
        raw_data: Raw text scraped from a single web page or document.

    Returns:
        A compact prompt string.  Expected LLM output: a short paragraph (3–8
        sentences) containing only query-relevant information from raw_data,
        or the string "IRRELEVANT" if the page contains nothing useful.
    """
    return f"""You are a precise information extractor.

RESEARCH QUERY: "{query}"

SOURCE TEXT:
\"\"\"{raw_data}\"\"\"

TASK:
Extract ONLY the information from the source text that is directly relevant to the research query above.

Rules:
• Keep ALL specific facts: numbers, statistics, percentages, dates, names, quotes, and figures.
• Discard everything unrelated to the query (navigation text, ads, unrelated topics).
• Do NOT add any information not present in the source text.
• Do NOT rephrase facts in a way that changes their meaning.
• Write in fluent, concise prose — typically 3–8 sentences.
• If the source contains NO useful information for the query, respond with exactly: IRRELEVANT

Output only the extracted summary (or IRRELEVANT). No preamble, no labels.
"""


# ---------------------------------------------------------------------------
# Report type dispatcher
# ---------------------------------------------------------------------------

def get_report_by_type(report_type: str) -> Callable:
    """Return the correct prompt function for the given report type string.

    Supported values:
      "research_report"  → generate_report_prompt   (full narrative report)
      "outline_report"   → generate_outline_prompt   (structured outline only)
      "resource_report"  → generate_resource_report_prompt (annotated bibliography)

    Falls back to generate_report_prompt for unrecognised types and logs a
    warning so callers can detect misconfiguration without crashing.

    Args:
        report_type: One of the strings listed above.

    Returns:
        A prompt-generating callable with the signature:
        ``(question, context, report_source, report_format, total_words,
           tone, language) -> str``
    """
    import warnings

    mapping: dict[str, Callable] = {
        "research_report": generate_report_prompt,
        "outline_report": generate_outline_prompt,
        "resource_report": generate_resource_report_prompt,
    }

    prompt_fn = mapping.get(report_type)
    if prompt_fn is None:
        warnings.warn(
            f"Unknown report_type {report_type!r}. "
            f"Valid options: {', '.join(mapping)}. "
            "Falling back to 'research_report'.",
            UserWarning,
            stacklevel=2,
        )
        prompt_fn = generate_report_prompt

    return prompt_fn


# ---------------------------------------------------------------------------
# Orchestrator prompts — moved here so EVERY prompt lives in one file (Phase 3B)
# ---------------------------------------------------------------------------


def get_plan_outline_prompt(query: str, initial_research: str, max_sections: int) -> str:
    """Prompt for EditorAgent.plan_research (STRATEGIC_LLM)."""
    return f"""You are a research editor planning a structured report.

Research query: {query}

Initial research context:
{initial_research[:3000]}

Plan a report outline with exactly {max_sections} main sections.

Rules:
- Each section title must be specific and non-overlapping
- Do NOT include Introduction, Conclusion, References, or Summary as sections
- Sections should together form a comprehensive answer to the query
- Make section titles concise (5-10 words each)

Respond with a JSON object:
{{
  "title": "Full report title here",
  "sections": ["Section 1 Title", "Section 2 Title", "Section 3 Title"]
}}"""


def get_section_review_prompt(
    topic: str,
    guidelines: list[str],
    draft: str,
    revision_notes: str,
) -> str:
    """Prompt for ReviewerAgent.run (SMART_LLM)."""
    guidelines_text = "\n".join(f"- {g}" for g in guidelines)
    return f"""You are a strict research editor reviewing a draft section.

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


def get_section_revise_prompt(topic: str, draft: str, review: str) -> str:
    """Prompt for ReviserAgent.run (SMART_LLM)."""
    return f"""You are a research writer revising a section draft.

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


def get_report_introduction_prompt(
    title: str,
    query: str,
    section_titles: list[str],
) -> str:
    """Prompt for WriterAgent intro generation (SMART_LLM)."""
    section_list = "\n".join(f"- {s}" for s in section_titles)
    return f"""Write a 2-3 paragraph introduction for a research report.

Report title: {title}
Query: {query}

The report covers these sections:
{section_list}

Write only the introduction paragraphs. No heading needed."""


def get_report_conclusion_prompt(
    title: str,
    query: str,
    sections_text: str,
) -> str:
    """Prompt for WriterAgent conclusion generation (SMART_LLM)."""
    return f"""Write a 2-3 paragraph conclusion for this research report.

Report title: {title}
Query: {query}

Report content summary:
{sections_text[:4000]}

Write only the conclusion paragraphs. No heading needed.
Synthesize key findings. Do not introduce new information."""


def get_short_conclusion_prompt(query: str, report_body: str) -> str:
    """Prompt for researcher.actions.report_generation.write_report_conclusion (FAST_LLM)."""
    return (
        f"Based on the research report below, write a concise conclusion (2–3 paragraphs) "
        f"that summarises the main findings and their implications for the question: \"{query}\"\n\n"
        f"If the report does not already end with a '## Conclusion' heading, prepend one.\n\n"
        f"Report:\n{report_body}"
    )


# ---------------------------------------------------------------------------
# Retriever registry  (not an LLM prompt — config helper)
# ---------------------------------------------------------------------------

_RETRIEVER_MAP: dict[str, list[str]] = {
    "tavily": ["TavilySearch"],
    "exa": ["ExaSearch"],
    "duckduckgo": ["DuckDuckGoSearch"],
    "serper": ["SerperSearch"],
    "bing": ["BingSearch"],
    "arxiv": ["ArxivSearch"],
    "pubmed": ["PubMedSearch"],
    "google": ["GoogleSearch"],
    "serpapi": ["SerpApiSearch"],
    "custom": ["CustomSearch"],
}


def get_retrievers(retriever: str) -> list[str]:
    """Return the list of retriever class names for the configured retriever string.

    Supports comma-separated hybrid retrieval, e.g. ``"tavily,duckduckgo"``
    will return ``["TavilySearch", "DuckDuckGoSearch"]``.

    Args:
        retriever: A single retriever name or a comma-separated list of names.

    Returns:
        Ordered list of retriever class name strings.

    Raises:
        ValueError: If any name in the string is not a recognised retriever.
    """
    names = [r.strip().lower() for r in retriever.split(",") if r.strip()]
    result: list[str] = []
    unknown: list[str] = []

    for name in names:
        classes = _RETRIEVER_MAP.get(name)
        if classes is None:
            unknown.append(name)
        else:
            result.extend(classes)

    if unknown:
        supported = ", ".join(sorted(_RETRIEVER_MAP))
        raise ValueError(
            f"Unknown retriever(s): {', '.join(unknown)}. "
            f"Supported: {supported}"
        )

    return result
