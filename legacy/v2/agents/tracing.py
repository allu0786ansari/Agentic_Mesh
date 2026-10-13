from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

LOGGER = logging.getLogger(__name__)


_PHOENIX_CONFIGURED = False


class Tracer:
    """Phoenix-compatible tracer facade with a deterministic local fallback."""

    def __init__(self, name: str):
        self.name = name
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise RuntimeError("OpenTelemetry is required for Phoenix tracing") from exc
        self._phoenix_tracer = trace.get_tracer(name)

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[object]:
        if not _PHOENIX_CONFIGURED:
            raise RuntimeError(
                "Phoenix tracing is not configured. Set PHOENIX_OTEL_ENDPOINT "
                "and call configure_phoenix() before running the agent graph."
            )
        with self._phoenix_tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span


def get_tracer(name: str = "agentic-mesh") -> Tracer:
    """Return the project tracing facade for local or Phoenix-backed spans."""
    return Tracer(name)


def launch_phoenix() -> object:
    """Launch the Phoenix application or raise if the installation is unusable."""
    try:
        import phoenix as phoenix_app
    except ImportError as exc:
        raise RuntimeError("Phoenix cannot be imported; repair the Phoenix installation") from exc
    return phoenix_app.launch_app()


def configure_phoenix(endpoint: str | None = None) -> None:
    """Enable mandatory OpenTelemetry export to a Phoenix OTLP endpoint."""
    global _PHOENIX_CONFIGURED
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("OpenTelemetry OTLP dependencies are required for Phoenix tracing") from exc
    endpoint = endpoint or os.getenv("PHOENIX_OTEL_ENDPOINT")
    if not endpoint:
        raise ValueError("PHOENIX_OTEL_ENDPOINT must be configured for Phoenix tracing")
    provider = TracerProvider(resource=Resource.create({"service.name": "agentic-mesh"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _PHOENIX_CONFIGURED = True
