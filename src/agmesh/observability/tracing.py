"""One trace per session, archived as JSON (Proposal §9.5; ADR 0014 §7).

**Spans.** They use OpenTelemetry with OpenInference semantic conventions
(``openinference.span.kind``, ``session.id``, ``llm.*``). The mesh creates them by
hand:

* ``session`` is the root;
* ``triage``, ``investigate``, ``attribute`` and ``dispatch`` sit under it;
* ``hop``, ``retrieve`` and ``llm`` sit inside those.

Prompts and evidence text are never put into a span.

**Export.** :class:`SessionFileExporter` buffers the spans of each trace. When the
root span ends, it writes them to ``<out>/<session_id>.json``. A replayed session
adds a second trace to the same file.

**Phoenix.** It is optional. When ``PHOENIX_COLLECTOR_ENDPOINT`` is set and the OTLP
exporter is installed (the ``obs`` group), spans also go there.

**The tracer is passed explicitly, never set globally.** A process may build several
runs, as tests do, and OpenTelemetry's global provider can be set only once.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
SESSION_ID = SpanAttributes.SESSION_ID
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value
LLM = OpenInferenceSpanKindValues.LLM.value
ROOT = "session"


def noop_tracer() -> trace.Tracer:
    return trace.NoOpTracer()


def trace_id_hex(span: trace.Span) -> str:
    return f"{span.get_span_context().trace_id:032x}"


class SessionFileExporter(SpanExporter):
    """Writes each trace to ``<out>/<session_id>.json`` when its root span ends."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer: dict[int, list[dict[str, Any]]] = {}
        self.lock = threading.Lock()
        self.written: list[Path] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.lock:
            for span in spans:
                ctx = span.get_span_context()
                if ctx is None:
                    continue
                self.buffer.setdefault(ctx.trace_id, []).append(json.loads(span.to_json()))
                if span.parent is None and span.name == ROOT:
                    self._flush(ctx.trace_id, span)
        return SpanExportResult.SUCCESS

    def _flush(self, trace_id: int, root: ReadableSpan) -> None:
        spans = self.buffer.pop(trace_id, [])
        attrs = dict(root.attributes or {})
        sid = str(attrs.get(SESSION_ID) or f"no-session-{trace_id:032x}")
        path = self.out / f"{sid}.json"
        doc: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"schema_version": "1.0", "session_id": sid, "traces": []}
        )
        doc["traces"].append({"trace_id": f"{trace_id:032x}", "spans": spans})
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.written.append(path)

    def shutdown(self) -> None:
        self.buffer.clear()


def setup(out: Path, service: str = "agmesh-mesh") -> tuple[trace.Tracer, TracerProvider]:
    """A tracer that archives per-session traces under ``out`` (and Phoenix, if set)."""
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(SessionFileExporter(out)))
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:  # the obs group is not installed
            raise SystemExit(
                "PHOENIX_COLLECTOR_ENDPOINT is set but the OTLP exporter is missing "
                "(uv sync --group obs)"
            ) from exc
        url = endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
    return provider.get_tracer("agmesh.mesh"), provider


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def check_run(trace_dir: Path, session_ids: Sequence[str]) -> list[str]:
    """Problems with a run's traces (ADR 0014 §7); empty when every session has one."""
    problems = []
    for sid in session_ids:
        path = trace_dir / f"{sid}.json"
        if not path.is_file():
            problems.append(f"{sid}: no trace file")
            continue
        doc = load(path)
        roots = [
            s
            for t in doc.get("traces", [])
            for s in t.get("spans", [])
            if s.get("name") == ROOT and s.get("parent_id") is None
        ]
        if not roots:
            problems.append(f"{sid}: no root span")
        elif any((r.get("attributes") or {}).get(SESSION_ID) != sid for r in roots):
            problems.append(f"{sid}: root span carries another session.id")
    return problems
