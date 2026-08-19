"""
OpenTelemetry tracing for the Focused Research Agent.

Directly addresses the README's stated gap: "No distributed tracing
(OpenTelemetry)." Exports to an OTLP collector (Jaeger, Tempo, etc.) when
OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise traces are created but not
exported anywhere, so this is safe to leave enabled even without a
collector running.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_TRACER_PROVIDER_INITIALIZED = False


def setup_tracing(service_name: str = "focused-research-agent") -> None:
    """Initialize the global OpenTelemetry tracer provider.

    Safe to call multiple times — only configures the provider once.
    """
    global _TRACER_PROVIDER_INITIALIZED
    if _TRACER_PROVIDER_INITIALIZED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
            logger.info("otel_exporting_to_otlp endpoint=%s", otlp_endpoint)
        except ImportError:
            logger.warning(
                "otel_otlp_exporter_not_installed_traces_created_but_not_exported"
            )

    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER_INITIALIZED = True


def get_tracer():
    """Return the module tracer for creating custom spans around graph nodes."""
    return trace.get_tracer("focused_research_agent")
