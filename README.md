# deep-researcher

An agentic RAG research system that autonomously searches, scrapes, and synthesizes information into structured reports.

## Quick start

```bash
cp .env.example .env   # fill in API keys
pip install -e .
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Docs available at `http://localhost:8000/docs`.

## CLI (single report)

```bash
python main.py task.json
```

## API Reference

### Start research
```
POST /research
Body: {"query": "...", "max_sections": 3}
Returns: {"task_id": "uuid", "status": "queued", "poll_url": "/research/{id}"}
```

### Check status / get report
```
GET /research/{task_id}
Returns: {"status": "complete", "report": "...", "sources": [...]}
```

### Stream logs (WebSocket)
```
WS /research/{task_id}/stream
Messages: {"type": "log"|"status"|"complete", "data": ...}
```

### Health
```
GET /health
Returns: {"status": "ok", "researcher_ready": true}
```

## Architecture

```
researcher/          # Layer 1-3: scraping, sub-queries, RAG context
orchestrator/        # Layer 4: multi-agent LangGraph pipeline
backend/             # Layer 5: FastAPI HTTP + WebSocket API
  schemas.py         # Pydantic request / response models
  task_manager.py    # In-memory async task store
  routes/
    health.py        # GET /health
    research.py      # POST/GET /research, WS /research/{id}/stream
```

## Running tests

```bash
pytest                             # unit + integration (no LLM required)
python scripts/test_api_live.py    # end-to-end (requires running server + LLM)
```
