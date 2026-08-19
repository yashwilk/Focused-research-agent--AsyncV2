"""
Celery application for the Focused Research Agent.


Uses Redis as both broker and result backend — the same Redis instance
already used for response caching and rate-limit storage, so no extra
infrastructure is required beyond what the other production-grade
improvements already need.
"""

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "focused_research_agent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["focused_research_agent.tasks.report_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    task_soft_time_limit=180,
    task_time_limit=210,
)
