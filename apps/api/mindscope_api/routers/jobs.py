import os
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from celery.result import AsyncResult

from mindscope_api.models import JobSubmitRequest, JobResponse, JobStatusResponse, DemoJobRequest
from mindscope_api.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


# Demo assets (paths relative to worker context)
DEMO_VIDEO_PATH = "/assets/cat_demo_20s.mp4"
DEMO_AUDIO_PATH = "/assets/cat_demo_20s_audio.wav"
DEMO_TEXT = "A tabby cat sits on a windowsill, stretching and purring contentedly."


@router.post("", response_model=JobResponse)
async def submit_job(request: JobSubmitRequest) -> JobResponse:
    """Submit a TRIBE v2 inference job."""
    job_id = str(uuid4())

    logger.info(f"Submitting job {job_id}")

    # Queue the Celery task
    task = celery_app.send_task(
        "tribe.predict",
        args=(
            request.video_path,
            request.audio_path,
            request.text,
            job_id,
        ),
    )

    return JobResponse(
        job_id=job_id,
        status="queued",
    )


@router.post("/demo", response_model=JobResponse)
async def submit_demo_job(request: DemoJobRequest) -> JobResponse:
    """Submit the preset cat demo job."""
    job_id = f"demo-{uuid4().hex[:8]}"

    logger.info(f"Submitting demo job {job_id}")

    # Queue the Celery task with demo assets
    task = celery_app.send_task(
        "tribe.predict",
        args=(
            DEMO_VIDEO_PATH,
            DEMO_AUDIO_PATH,
            DEMO_TEXT,
            job_id,
        ),
    )

    return JobResponse(
        job_id=job_id,
        status="queued",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a submitted job."""

    # For demo jobs, check if they're in our local store (stubbed for now)
    if job_id.startswith("demo-"):
        # Return a pre-filled response for demo job
        return JobStatusResponse(
            job_id=job_id,
            status="success",
            result={
                "bold_shape": (20484, 40),
                "timestamps_length": 40,
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    # For Celery tasks, retrieve from Redis backend
    task_result = AsyncResult(job_id, app=celery_app)

    # Map Celery state to our status
    status_map = {
        "PENDING": "queued",
        "STARTED": "processing",
        "SUCCESS": "success",
        "FAILURE": "error",
        "RETRY": "processing",
    }

    status = status_map.get(task_result.state, "unknown")

    response = JobStatusResponse(
        job_id=job_id,
        status=status,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Attach result if successful
    if task_result.state == "SUCCESS":
        response.result = task_result.result
    # Attach error if failed
    elif task_result.state == "FAILURE":
        response.error = str(task_result.info)

    return response
