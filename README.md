# deep-researcher

An agentic RAG research system that autonomously searches the web, scrapes sources, synthesises context through embeddings, and produces structured multi-section reports — served over a REST + WebSocket API.

Modelled after [GPT Researcher](https://github.com/assafelovic/gpt-researcher) and built layer-by-layer so each piece is independently testable.

---

## Architecture overview

The system is composed of six layers, each building on the previous:

```
Layer 1 — researcher/          Core research agent (search → scrape → summarise → report)
Layer 2 — researcher/          Parallel sub-query execution (asyncio.gather)
Layer 3 — researcher/          Embedding-based RAG context (HuggingFace / OpenAI / Ollama)
Layer 4 — orchestrator/        Multi-agent LangGraph pipeline (6 specialised agents)
Layer 5 — backend/             FastAPI REST + WebSocket API with async task queue
Layer 6 — evals/               LLM-as-judge evaluation framework (25-question dataset)
```

### Package map

```
researcher/
├── agent.py                   DeepResearcher — public entry point for Layers 1-3
├── config/config.py           Config dataclass (env vars + JSON override)
├── prompts.py                 Every LLM prompt template lives here (single source of truth)
├── actions/
│   ├── query_processing.py    Sub-query generation (STRATEGIC_LLM)
│   ├── web_scraping.py        Parallel search + scrape with asyncio workers
│   └── report_generation.py  LLM report writer and summariser
├── retrievers/                DuckDuckGo · Tavily · Serper
├── scraper/                   BeautifulSoup4 HTML scraper (article → main → semantic div → largest div → body)
├── embeddings/                OpenAI · HuggingFace · Ollama embedders
├── vector_store/              In-memory cosine similarity store (numpy)
├── context/                   Sliding-window chunker + RAG context manager
├── memory/                    Source deduplication and cost accumulation
└── llm_providers/             OpenAI · Anthropic · Google · Groq · Ollama · VESSL
    └── pricing.py             Per-1M-token USD rates for cost estimation

orchestrator/
├── state.py                   ResearchState + DraftState TypedDicts
├── graph.py                   build_main_graph() + build_section_subgraph()
└── agents/
    ├── chief_editor.py        Entry point — builds and runs the graph
    ├── editor.py              Plans sections, spawns parallel sub-graphs
    ├── researcher.py          Wraps DeepResearcher for each section
    ├── reviewer.py            Quality-gates each draft (capped by Config.MAX_REVISIONS, default 2)
    ├── reviser.py             Rewrites drafts on reviewer feedback
    ├── writer.py              Writes intro + conclusion; fills table_of_contents and headers
    └── publisher.py           Saves markdown (+ optional PDF / DOCX), uses writer's TOC when present

backend/
├── main.py                    FastAPI app factory (CORS, lifespan, router mount)
├── schemas.py                 Pydantic models (ResearchRequest, ResearchResponse, …)
├── task_manager.py            Async in-memory task store with asyncio.Lock + log buffer
└── routes/
    ├── health.py              GET /health
    └── research.py            POST/GET/DELETE /research, WS /research/{id}/stream

evals/
├── dataset/eval_questions.csv 25 ground-truth Q&A pairs (5 categories × 5 questions)
├── grader.py                  LLM-as-judge (opposite provider to avoid self-preference bias)
├── metrics.py                 F1, accuracy, answer rate, per-category breakdown
├── runner.py                  Async eval runner with semaphore-gated concurrency
├── reporter.py                Console dashboard + timestamped markdown reports
└── __main__.py                python -m evals CLI
```

---

## Quick start

### Prerequisites

- Python 3.11+
- At least one LLM API key (OpenAI, Anthropic, Google, Groq) **or** a local VESSL / Ollama endpoint

### Install

```bash
git clone https://github.com/Uzbekswe/research.git
cd research
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` — the minimum required fields:

```env
# Pick one LLM provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Pick a retriever (duckduckgo needs no key)
RETRIEVER=duckduckgo
# or RETRIEVER=tavily  +  TAVILY_API_KEY=tvly-...

# Pick an embedding model
EMBEDDING=huggingface:all-MiniLM-L6-v2   # free, runs locally
# or EMBEDDING=openai:text-embedding-3-small

# Set your LLM models (tiered: FAST = per-source summarisation,
# SMART = final report, STRATEGIC = planning / outlines)
FAST_LLM=openai:gpt-4o-mini
SMART_LLM=openai:gpt-4o
STRATEGIC_LLM=openai:gpt-4o

# Optional — reviewer revision cap for the multi-agent path (default 2)
MAX_REVISIONS=2
```

#### Using VESSL AI (vLLM on GPU)

```env
VESSL_BASE_URL=http://localhost:8000/v1
VESSL_API_KEY=mytoken

FAST_LLM=vessl:Qwen/Qwen2.5-7B-Instruct
SMART_LLM=vessl:Qwen/Qwen2.5-7B-Instruct
STRATEGIC_LLM=vessl:Qwen/Qwen2.5-7B-Instruct
EMBEDDING=huggingface:all-MiniLM-L6-v2
```

Forward the VESSL port with: `ssh -N -L 8000:localhost:8000 <vessl-host>`

---

## Usage

### Option A — REST API (recommended)

Start the server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# or: python -m backend.main
```

Interactive docs at `http://localhost:8000/docs`.

**Start a research task:**

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does in-context learning work in large language models?",
    "max_sections": 3,
    "follow_guidelines": false
  }'
# → {"task_id": "abc-123", "status": "queued", "poll_url": "/research/abc-123"}
```

**Poll until complete:**

```bash
curl http://localhost:8000/research/abc-123
# → {"status": "complete", "report": "...", "sources": [...], "elapsed_seconds": 87.4}
```

**Stream logs in real time (WebSocket):**

```js
const ws = new WebSocket("ws://localhost:8000/research/abc-123/stream");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// {"type": "log",      "data": "[12:01:04] Research started: How does..."}
// {"type": "status",   "data": "running"}
// {"type": "complete", "data": {"report_length": 4821, "sources": 12, "elapsed": 91.2}}
```

**End-to-end live test** (requires running server):

```bash
python scripts/test_api_live.py
```

### Option B — CLI (single report)

Edit `task.json`, then:

```bash
python main.py task.json
# Report saved to ./outputs/
```

`task.json` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Research question |
| `max_sections` | int | 3 | Number of report sections |
| `follow_guidelines` | bool | false | Enforce `guidelines` list |
| `guidelines` | list[str] | [] | Quality rules for the reviewer |
| `model` | string | from `.env` | Override `provider:model` for this run |
| `publish_formats` | dict | `{"markdown": true}` | Output formats |

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check + researcher import probe |
| `POST` | `/research` | Queue task → 202 + `task_id` |
| `GET` | `/research` | List recent tasks (newest first) |
| `GET` | `/research/{task_id}` | Poll status and retrieve report |
| `DELETE` | `/research/{task_id}` | Remove task from in-memory store |
| `WS` | `/research/{task_id}/stream` | Real-time log streaming |

**Task status lifecycle:** `queued → running → complete | failed`

**WebSocket message types:**

```
{"type": "log",      "data": "<timestamped log line>"}
{"type": "status",   "data": "queued|running|complete|failed"}
{"type": "complete", "data": {"report_length": int, "sources": int, "elapsed": float}}
{"type": "error",    "data": "<error message>"}
```

---

## Evaluation framework

Run factual accuracy evals against the 25-question ground-truth dataset:

```bash
# Quick dev run — 5 AI/ML questions
python -m evals --num_examples 5 --categories ai_ml --concurrent 2 --save_report

# Full evaluation — all 25 questions
python -m evals --num_examples 25 --concurrent 3 --save_report
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num_examples` | 10 | Questions to evaluate (max 25) |
| `--categories` | all | Filter: `ai_ml` `software_engineering` `science_facts` `history_geography` `tech_companies` |
| `--concurrent` | 2 | Parallel researcher calls (controls cost) |
| `--save_report` | off | Write timestamped `.md` to `evals/results/` |

**Grading methodology:** LLM-as-judge using a *different* provider than the researcher (prevents self-preference bias). VESSL/Groq/Ollama researchers → OpenAI judge. OpenAI researcher → Anthropic judge. Override with `GRADER_LLM=provider:model`.

**Metrics reported:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| F1 score | `2·P·R / (P+R)` | Primary signal — punishes both refusals and hallucinations |
| Accuracy | `correct / attempted` | Precision when the system commits to an answer |
| Answer rate | `attempted / total` | How often the system tries (vs. NOT_ATTEMPTED) |
| Correct rate | `correct / total` | Fraction of all questions answered correctly |

Results saved to `evals/results/eval_YYYYMMDD_HHMMSS.json` (raw) and `eval_report_*.md` (formatted).

---

## LLM provider format

All LLM and embedding fields use `provider:model` format:

| Provider | Example |
|----------|---------|
| OpenAI | `openai:gpt-4o-mini` |
| Anthropic | `anthropic:claude-haiku-4-5` |
| Google | `google:gemini-2.0-flash` |
| Groq | `groq:llama-3.1-8b-instant` |
| Ollama | `ollama:llama3.2` |
| VESSL (vLLM) | `vessl:Qwen/Qwen2.5-7B-Instruct` |
| HuggingFace (embeddings only) | `huggingface:all-MiniLM-L6-v2` |

---

## Cost tracking

Every LLM provider populates `last_usage = {"prompt_tokens": int, "completion_tokens": int}` after each call. The action layer forwards this to `DeepResearcher.add_costs`, which looks up per-1M-token USD rates in [`researcher/llm_providers/pricing.py`](researcher/llm_providers/pricing.py) and accumulates the running total.

```python
from researcher import DeepResearcher

r = DeepResearcher(query="What year was the Transformer introduced?")
await r.conduct_research()
print(f"${r.get_costs():.4f}")     # → "$0.0023"
```

The pricing table covers the current OpenAI, Anthropic, and Google model families; unknown models fall back to `0.0` rather than crashing. Update the table in `pricing.py` when new models ship.

---

## Testing

```bash
# All unit + integration tests (no LLM calls required)
pytest

# Individual layers
pytest tests/test_layer1_smoke.py    # researcher config + construction
pytest tests/test_layer3_rag.py      # embeddings, vector store, chunker, context manager
pytest tests/test_layer4_graph.py    # LangGraph compilation, state TypedDicts, ChiefEditorAgent
pytest tests/test_layer5_api.py      # FastAPI routes, task lifecycle, validation

# Live end-to-end (requires running server + configured LLM)
python scripts/test_api_live.py
```

Test coverage: 52 tests across all layers. No test requires a live LLM or internet connection.

---

## Deployment

The `Procfile` is included for Heroku / Railway / Render:

```
web: uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API server is stateless except for the in-memory task store. For multi-process deployment, replace `TaskManager` in `backend/task_manager.py` with a Redis-backed store.

---

## Audit

A senior-engineering audit report (Opus, 2026-05-28) lives at [`AUDIT_REPORT.md`](AUDIT_REPORT.md). It documents the GPT Researcher fidelity checks (3A–3I), every bug fixed, every file changed, and the recommended follow-up work.
