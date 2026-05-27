from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class ReportType(str, Enum):
    research_report = "research_report"
    outline_report = "outline_report"
    resource_report = "resource_report"


class ReportSource(str, Enum):
    web = "web"
    local = "local"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500,
                       description="The research question or topic")
    report_type: ReportType = ReportType.research_report
    report_source: ReportSource = ReportSource.web
    max_sections: int = Field(default=3, ge=1, le=10)
    follow_guidelines: bool = False
    guidelines: list[str] = Field(default_factory=list)
    model: Optional[str] = None   # overrides config if provided
    publish_formats: dict = Field(
        default_factory=lambda: {"markdown": True, "pdf": False, "docx": False}
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest advances in quantum computing?",
                "report_type": "research_report",
                "max_sections": 3,
                "follow_guidelines": False,
                "guidelines": [],
            }
        }


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


class ResearchResponse(BaseModel):
    task_id: str
    status: TaskStatus
    query: str
    created_at: datetime
    updated_at: datetime
    report: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    poll_url: str


class HealthResponse(BaseModel):
    status: str
    version: str
    researcher_ready: bool
