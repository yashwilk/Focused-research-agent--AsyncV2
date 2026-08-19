"""
Celery task wrapping the report use case.

Celery tasks run in a separate worker process with their own event loop
lifecycle — they cannot reuse a FastAPI request's AsyncSession. This
module opens its own short-lived async DB session per task run via
asyncio.run(), which is the standard bridge pattern for calling async
application code from Celery's synchronous task execution model.
"""

import asyncio
import logging

from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.application.report_use_case import execute_report
from focused_research_agent.database.database import AsyncSessionLocal
from focused_research_agent.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _run_report(question: str, user_id: int | None) -> dict:
    """Open a fresh async DB session and run the report use case within it."""
    async with AsyncSessionLocal() as db:
        return await execute_report(question, db, user_id=user_id)


@celery_app.task(name="generate_report", bind=True)
def generate_report_task(self, question: str, user_id: int | None = None) -> dict:
    """Run report generation as a background Celery task.

    Args:
        question: The user's research question for the report.
        user_id: The authenticated user who submitted the job, for
            conversation-history scoping. None for anonymous/CLI use.

    Returns:
        dict: The normalized report result (same shape as the synchronous
            /api/v1/report endpoint), or an error payload if the question
            failed validation.
    """
    logger.info("Report task started. task_id=%s question='%s'", self.request.id, question[:50])
    try:
        result = asyncio.run(_run_report(question, user_id))
        logger.info("Report task completed. task_id=%s status=%s", self.request.id, result.get("status"))
        return result
    except ApplicationError as e:
        logger.warning("Report task rejected invalid question. task_id=%s error=%s", self.request.id, e)
        return {"status": "error", "errors": [str(e)]}
