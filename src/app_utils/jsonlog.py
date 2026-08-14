"""JSON-line logging for the agent — one object per record, on stdout.

Why JSON rather than the default text format
--------------------------------------------
Cloud Run and the Agent Engine runtime both parse a JSON object printed on
stdout into Cloud Logging's ``jsonPayload``, so every key becomes something a
log filter can select on: ``jsonPayload.request_id="…"`` pulls one turn out of
the stream, ``jsonPayload.duration_ms>2000`` finds the slow nodes. A text line
gives you a substring search and nothing else.

Three of the keys emitted here are magic to Cloud Logging rather than to us:
``severity`` drives the log-level colouring and filters, ``message`` fills the
collapsed summary line in the log viewer, and ``logging.googleapis.com/trace``
joins a record to the Cloud Trace span it happened inside. The last one is what
stitches these lines to the spans ``LanggraphAgent(enable_tracing=True)`` already
exports — see :mod:`src.app_utils.telemetry` for why those spans deliberately
carry no prompt or response content.

Standard library only. ``google-cloud-logging`` is declared in requirements.txt
but imported nowhere, and is not needed: its value is shipping logs from
somewhere that *isn't* a Google runtime, and both places this runs already
collect stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

# The logger our handler is attached to. Scoped to the package rather than the
# root logger on purpose: root would wrap every langchain, urllib3 and grpc
# record in our envelope too, which is a lot of noise for no gain and takes the
# platform's own logging away from whatever configured it.
PACKAGE_LOGGER = "src"

# Marks our handler so configure_logging can be called repeatedly — Agent Engine
# may call set_up() more than once per process, and the tests import this module
# many times — without stacking a duplicate handler each time.
_HANDLER_MARK = "_causaltrace_json_handler"

DEFAULT_LEVEL = "INFO"

# Attributes every LogRecord carries. Anything outside this set arrived through
# `extra={...}` and is what we actually want in the payload, so the formatter
# diffs against it rather than maintaining a whitelist of our own field names.
_RESERVED_ATTRIBUTES = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)

try:  # pragma: no cover - depends on what the runtime installed
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover - local runs without OTel
    _otel_trace = None


def content_logging_enabled() -> bool:
    """Whether raw user text may be written to the log stream.

    Off by default, mirroring the NO_CONTENT policy in
    :mod:`src.app_utils.telemetry`: user questions carry business context, and
    Cloud Logging has a different retention and IAM story than the Firestore
    conversation history they are already stored in. Turn it on only where that
    trade is acceptable — a local run, a scratch project.

    Read per call rather than cached at import so a test (or a shell) can flip
    it without re-configuring logging.
    """
    return os.environ.get("CAUSAL_LOG_CONTENT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _iso_timestamp(created: float) -> str:
    """RFC 3339 UTC, milliseconds, ``Z`` suffix.

    Built from ``record.created`` rather than "now" so the stamp is when the
    event happened, not when the handler got round to formatting it.
    """
    stamp = datetime.fromtimestamp(created, tz=timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _trace_fields() -> dict[str, str]:
    """Cloud Logging's trace-correlation keys, when there is a span to join to.

    Returns empty rather than raising anywhere OTel is absent or no span is
    recording, which is every local run and the whole test suite.
    """
    if _otel_trace is None:
        return {}
    try:
        span = _otel_trace.get_current_span()
        context = span.get_span_context()
        if not context.is_valid:
            return {}
        fields = {"span_id": format(context.span_id, "016x")}
        trace_id = format(context.trace_id, "032x")
        # Cloud Logging only links the trace when it is fully qualified with the
        # project; without one the bare id is still worth having as a join key.
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            fields["logging.googleapis.com/trace"] = (
                f"projects/{project}/traces/{trace_id}"
            )
        else:
            fields["trace_id"] = trace_id
        return fields
    except Exception:  # pragma: no cover - defensive; never break a log call
        return {}


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _iso_timestamp(record.created),
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        payload.update(_trace_fields())

        try:
            # default=str so an unexpected type (a tensor, a pydantic model)
            # degrades to its repr instead of taking the call site down.
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception as exc:
            # A logging failure must never be the reason a node dies. Emit
            # something valid and self-describing and move on.
            return json.dumps(
                {
                    "timestamp": _iso_timestamp(record.created),
                    "severity": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "log_error": f"payload was not serialisable: {exc}",
                },
                default=str,
                ensure_ascii=False,
            )


def _resolve_level(level: Optional[str]) -> int:
    """Read a level name, falling back to INFO rather than failing at start-up.

    Same reasoning as ``_int_env`` in src/agent.py: a typo in an observability
    knob should not take the container down.
    """
    name = (level or os.environ.get("CAUSAL_LOG_LEVEL") or DEFAULT_LEVEL).upper()
    resolved = logging.getLevelName(name)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(
    level: Optional[str] = None, stream: Optional[TextIO] = None
) -> logging.Logger:
    """Attach the JSON handler to the package logger. Idempotent.

    Writes to stdout, which is what both runtimes collect. ``propagate`` is
    turned off because the Agent Engine runtime installs its own root handler:
    left on, every record would print twice — once as JSON from here, once as
    plain text from there.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARK, False):
            if stream is not None:
                handler.setStream(stream)  # type: ignore[attr-defined]
            return logger

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)
    return logger
