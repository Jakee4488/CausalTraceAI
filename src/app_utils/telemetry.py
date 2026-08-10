"""Telemetry policy for the Agent Engine deployment.

Content-capture policy — deliberate, not a leftover default
-----------------------------------------------------------
Prompts and responses are never written to spans, and the GenAI upload runs in
``NO_CONTENT`` mode (metadata only). User prompts carry business context and
sometimes attached data, so shipping them into trace storage would put user
data somewhere with a different retention and access model than Firestore.

What this costs: a trace tells you *that* a node ran, with what token count and
latency, but not what it said. Reasoning is reconstructed instead from the
causal state the graph records deliberately and shows the user — the model
graph, the parsed query, the decision result and the trace lines — all keyed by
``causal_run_id`` so a turn can be followed end to end.

Span export itself is handled by ``LanggraphAgent(enable_tracing=True)``, which
installs the Agent Engine instrumentor at ``set_up()``. This module only sets
the environment policy that instrumentor reads, so it must run before the agent
is constructed.
"""

from __future__ import annotations

import logging
import os


def setup_telemetry() -> str | None:
    """Configure GenAI prompt/response logging via OpenTelemetry.

    To capture content in a non-production project, set both ``LOGS_BUCKET_NAME``
    and ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``; only do that
    where the bucket's retention and IAM are appropriate for user data.
    """
    os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true")

    bucket = os.environ.get("LOGS_BUCKET_NAME")
    capture_content = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
    )
    if bucket and capture_content != "false":
        logging.info(
            "Prompt-response logging enabled - mode: NO_CONTENT "
            "(metadata only, no prompts/responses)"
        )
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "NO_CONTENT"
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_UPLOAD_FORMAT", "jsonl")
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK", "upload")
        os.environ.setdefault(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        commit_sha = os.environ.get("COMMIT_SHA", "dev")
        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"service.namespace=causaltraceai,service.version={commit_sha}",
        )
        path = os.environ.get("GENAI_TELEMETRY_PATH", "completions")
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH",
            f"gs://{bucket}/{path}",
        )
    else:
        logging.info(
            "Prompt-response logging disabled (set LOGS_BUCKET_NAME=gs://your-bucket "
            "and OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT to enable)"
        )

    return bucket
