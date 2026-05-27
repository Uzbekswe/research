from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.health import router as health_router
from backend.routes.research import router as research_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Deep Researcher API starting...")
    print("   Docs available at: http://localhost:8000/docs")
    yield
    print("👋 Deep Researcher API shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Researcher API",
        description="Agentic RAG research system — async research pipeline with multi-agent orchestration",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, tags=["Health"])
    app.include_router(research_router, prefix="/research", tags=["Research"])

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
