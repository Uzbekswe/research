from fastapi import APIRouter

from backend.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Liveness check. Returns 200 if the API is running.
    Also verifies the researcher package imports correctly.
    """
    try:
        from researcher import DeepResearcher
        researcher_ready = True
    except Exception:
        researcher_ready = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        researcher_ready=researcher_ready
    )
