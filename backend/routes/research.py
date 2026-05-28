import asyncio
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from backend.schemas import ResearchRequest, ResearchResponse, TaskCreateResponse, TaskStatus
from backend.task_manager import task_manager

router = APIRouter()


async def run_research_task(task_id: str, request: ResearchRequest) -> None:
    """
    Background coroutine — runs the full research pipeline.
    Called by FastAPI's BackgroundTasks after POST /research returns.

    Uses ChiefEditorAgent (multi-agent, Layer 4) for multi-section reports.
    Falls back to DeepResearcher directly (Layer 1-3) for simple queries.
    """
    await task_manager.set_running(task_id)
    await task_manager.add_log(task_id, f"Research started: {request.query}")

    start_time = time.time()

    try:
        # Build task config same format as task.json
        task_config = {
            "query": request.query,
            "max_sections": request.max_sections,
            "follow_guidelines": request.follow_guidelines,
            "guidelines": request.guidelines,
            "publish_formats": request.publish_formats,
            "source": request.report_source.value,
        }
        if request.model:
            task_config["model"] = request.model

        await task_manager.add_log(task_id, f"Using {request.max_sections} sections")

        from orchestrator import ChiefEditorAgent

        chief = ChiefEditorAgent(
            task=task_config,
            output_dir="./outputs"
        )

        await task_manager.add_log(task_id, "Running multi-agent research pipeline...")
        final_state = await chief.run_research_task()

        report = final_state.get("report", "")
        sources = final_state.get("sources", [])
        sections = [item["section"] for item in final_state.get("research_data", [])]
        elapsed = time.time() - start_time

        await task_manager.add_log(task_id, f"Complete in {elapsed:.1f}s — {len(sources)} sources")

        await task_manager.set_complete(
            task_id=task_id,
            report=report,
            sources=sources,
            sections=sections,
            elapsed=elapsed
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        await task_manager.add_log(task_id, f"FAILED: {error_msg}")
        await task_manager.set_failed(task_id, error_msg)
        print(f"❌ Task {task_id} failed after {elapsed:.1f}s: {error_msg}")


@router.post("", response_model=TaskCreateResponse, status_code=202)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a research task asynchronously.

    Returns immediately with a task_id.
    Poll GET /research/{task_id} to check status and retrieve the report.

    Status codes:
      202 Accepted — task created and queued
    """
    task_id = await task_manager.create_task(query=request.query)

    # Schedule research to run in the background
    # FastAPI's BackgroundTasks runs after the response is sent
    background_tasks.add_task(run_research_task, task_id, request)

    return TaskCreateResponse(
        task_id=task_id,
        status=TaskStatus.queued,
        message="Research task created. Poll the status endpoint for updates.",
        poll_url=f"/research/{task_id}"
    )


@router.get("/{task_id}", response_model=ResearchResponse)
async def get_research_status(task_id: str):
    """
    Get the status and result of a research task.

    Status values:
      queued   — waiting to start
      running  — research in progress
      complete — finished, report is in the response
      failed   — error occurred, check the error field

    Poll this endpoint every 3-5 seconds until status is complete or failed.
    """
    task = await task_manager.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    return task


@router.get("", response_model=list[ResearchResponse])
async def list_research_tasks(limit: int = 10):
    """
    List recent research tasks, newest first.
    Useful for building a task history view.
    """
    return await task_manager.list_tasks(limit=limit)


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: str):
    """Remove a completed task from the store."""
    # OPUS FIX: use the lock-protected TaskManager.delete_task instead of
    # reaching into the private _tasks/_logs dicts directly. The previous code
    # bypassed self._lock and could race with concurrent set_running/set_complete.
    existed = await task_manager.delete_task(task_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Task not found")


@router.websocket("/{task_id}/stream")
async def stream_research_logs(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time log streaming.
    Mirrors the WebSocket pattern from GPT Researcher's backend.

    Connect immediately after POST /research.
    Receives log lines as they're added during the research run.
    Connection closes automatically when task completes or fails.

    Message format: {"type": "log"|"status"|"complete", "data": string}
    """
    await websocket.accept()

    task = await task_manager.get_task(task_id)
    if not task:
        await websocket.send_json({"type": "error", "data": "Task not found"})
        await websocket.close()
        return

    sent_index = 0

    try:
        while True:
            task = await task_manager.get_task(task_id)

            # Send any new log lines
            new_logs = await task_manager.get_logs_since(task_id, sent_index)
            for log_line in new_logs:
                await websocket.send_json({"type": "log", "data": log_line})
                sent_index += 1

            # Send status update
            await websocket.send_json({
                "type": "status",
                "data": task.status.value
            })

            # If terminal state, send final message and close
            if task.status == TaskStatus.complete:
                await websocket.send_json({
                    "type": "complete",
                    "data": {
                        "report_length": len(task.report or ""),
                        "sources": len(task.sources),
                        "elapsed": task.elapsed_seconds
                    }
                })
                break

            if task.status == TaskStatus.failed:
                await websocket.send_json({
                    "type": "error",
                    "data": task.error
                })
                break

            # Poll every 1 second for new logs
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass  # Client disconnected — normal, don't log as error
