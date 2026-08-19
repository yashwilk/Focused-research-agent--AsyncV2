"""
Prometheus metrics for the Focused Research Agent.
"""

from prometheus_client import Counter, Histogram, make_asgi_app

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration (s)", ["method", "endpoint"]
)
graph_run_duration_seconds = Histogram(
    "graph_run_duration_seconds",
    "Full LangGraph run duration (s)",
    ["mode"],
    buckets=[1, 3, 5, 10, 15, 20, 30, 60, 120],
)
graph_run_total = Counter("graph_run_total", "Total LangGraph runs", ["mode", "status"])
provider_call_duration_seconds = Histogram(
    "provider_call_duration_seconds",
    "External provider call duration (s)",
    ["provider"],
)
circuit_breaker_state = Counter(
    "circuit_breaker_open_total", "Times a circuit breaker tripped open", ["provider"]
)
cache_lookups_total = Counter(
    "cache_lookups_total",
    "Response cache lookups",
    ["result"],  # hit | miss
)
search_reflection_triggered_total = Counter(
    "search_reflection_triggered_total", "Times the reflection loop re-searched"
)

metrics_asgi_app = make_asgi_app()
