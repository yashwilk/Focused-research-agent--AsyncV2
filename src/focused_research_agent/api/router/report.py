"""
Report generation API endpoints for the Focused Research Agent.

Two ways to generate a report, both authenticated and rate-limited:

1. POST /api/v1/report — runs inline and returns the finished report.
   Now async (non-blocking DB/graph calls), but still ties up the HTTP
   request for the full run (report mode is the slowest workflow —
   advanced search depth + long-form synthesis).

2. POST /api/v1/report/submit + GET /api/v1/report/jobs/{job_id} — the
   task-queue pattern. Submission returns a job_id immediately; the
   caller polls for status/result. This is the recommended path for
   report generation specifically, and directly addresses the README's
   stated gap: "a task queue (Celery + Redis) would be needed" for
   long-running operations.
"""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.api.dependencies import get_report_use_case
from focused_research_agent.api.schema.report.report import (
    ReportJobResponse,
    ReportJobStatusResponse,
    ReportRequest,
    ReportResponse,
)
from focused_research_agent.auth.dependencies import get_current_user
from focused_research_agent.core.rate_limiter import RATE_LIMIT_REPORT, limiter
from focused_research_agent.database.database import get_db
from focused_research_agent.tasks.celery_app import celery_app
from focused_research_agent.tasks.report_tasks import generate_report_task

logger = logging.getLogger(__name__)

report_router = APIRouter(tags=["report"])


@report_router.post(
    "/report", status_code=status.HTTP_200_OK, response_model=ReportResponse
)
@limiter.limit(RATE_LIMIT_REPORT)
async def report(
    request: Request,
    report_request: ReportRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    run_report_use_case: Annotated[Callable, Depends(get_report_use_case)],
    _current_user=Depends(get_current_user),
) -> dict:
    """Run report generation inline and return the finished report."""
    return await run_report_use_case(
        question=report_request.question, db=db, user_id=_current_user.id
    )


@report_router.post(
    "/report/submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReportJobResponse,
)
@limiter.limit(RATE_LIMIT_REPORT)
async def submit_report_job(
    request: Request,
    report_request: ReportRequest,
    _current_user=Depends(get_current_user),
) -> dict:
    """Submit report generation as a background job and return a job_id immediately.

    Raises:
        HTTPException: 503 if the Celery broker (Redis) is unreachable at
            submission time, instead of letting a raw connection error
            propagate as an opaque, unhandled 500.
    """
    try:
        task = generate_report_task.delay(report_request.question, _current_user.id)
    except Exception as e:  # noqa: BLE001 — broker/serialization errors must map to a clean 503, not an opaque 500
        logger.error("report_job_submission_failed error=%s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report queue is currently unavailable. Please try again shortly, "
            "or use POST /api/v1/report for a synchronous result.",
        )
    return {"job_id": task.id, "status": "submitted"}


@report_router.get("/report/jobs/{job_id}", response_model=ReportJobStatusResponse)
async def get_report_job(job_id: str, _current_user=Depends(get_current_user)) -> dict:
    """Poll the status/result of a previously submitted report job."""
    task_result = celery_app.AsyncResult(job_id)

    if task_result.state == "PENDING":
        return {"job_id": job_id, "status": "pending", "result": None}
    if task_result.state == "STARTED":
        return {"job_id": job_id, "status": "running", "result": None}
    if task_result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed", "result": task_result.result}
    if task_result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(task_result.result))

    return {"job_id": job_id, "status": task_result.state.lower(), "result": None}
