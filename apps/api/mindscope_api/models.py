from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class JobSubmitRequest(BaseModel):
    """Request to submit a TRIBE v2 inference job."""
    video_path: str = Field(..., description="Path to input video file")
    audio_path: str = Field(..., description="Path to input audio file")
    text: str = Field(..., description="Text narration or description")


class JobResponse(BaseModel):
    """Response with job metadata."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, processing, success, error")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class JobStatusResponse(BaseModel):
    """Detailed job status and result."""
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DemoJobRequest(BaseModel):
    """Request to run the preset demo (cat video)."""
    pass
