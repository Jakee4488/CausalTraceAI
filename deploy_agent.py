#!/usr/bin/env python
"""Deploy the LangGraph agent to Vertex AI Agent Engine.

Replaces the ``agents-cli deploy`` step TracerLensAi used. agents-cli packages
an *ADK* agent directory; this project's agent is a LangGraph runnable wrapped in
``LanggraphAgent``, which Agent Engine takes through the SDK's own
``agent_engines.create``/``update`` instead.

What gets shipped
-----------------
The agent object is pickled and ``src/`` rides along as an extra package, so the
serving side can import ``src.causal.*`` when it unpickles. The object must not
be set up locally: ``set_up()`` builds a Gemini client, and a live client cannot
be serialised.

Usage::

    python deploy_agent.py                 # create, or update the recorded engine
    python deploy_agent.py --create        # force a new engine
    python deploy_agent.py --dry-run       # show what would be sent

Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_REGION (or --project / --region), and
records the resource name in deployment_metadata.json so the next run updates in
place rather than leaving a trail of orphaned engines.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).resolve().parent
METADATA_PATH = REPO_ROOT / "deployment_metadata.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

DISPLAY_NAME = "CausalTraceAI"
DESCRIPTION = (
    "Bayesian causal decision agent: Gemini interprets the question, Pyro "
    "computes the policy, Gemini narrates the result."
)


def read_requirements() -> list[str]:
    """The agent's dependency list, from requirements.txt.

    Parsed rather than duplicated: a second hand-maintained list here is a list
    that drifts, and the failure mode is an engine that imports fine locally and
    dies on its first request in the cloud.
    """
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for line in lines
        if (stripped := line.split("#", 1)[0].strip())
    ]


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except ValueError:
            print(f"! {METADATA_PATH.name} is not valid JSON; ignoring it")
    return {}


def write_metadata(resource_name: str) -> None:
    METADATA_PATH.write_text(
        json.dumps(
            {
                "remote_agent_engine_id": resource_name,
                "agent_framework": "langgraph",
                "deployment_timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--region",
        default=os.environ.get("GOOGLE_CLOUD_REGION", "europe-west2"),
    )
    parser.add_argument(
        "--staging-bucket",
        default=os.environ.get("AGENT_ENGINE_STAGING_BUCKET"),
        help="gs://... bucket for the packaged agent (required by the SDK).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new engine even if one is already recorded.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.project:
        print("GOOGLE_CLOUD_PROJECT is not set (and --project was not given).")
        return 1
    if not args.staging_bucket and not args.dry_run:
        print(
            "A staging bucket is required. Set AGENT_ENGINE_STAGING_BUCKET=gs://... "
            "or pass --staging-bucket."
        )
        return 1

    requirements = read_requirements()
    metadata = load_metadata()
    existing = metadata.get("remote_agent_engine_id")

    print(f"project        : {args.project}")
    print(f"region         : {args.region}")
    print(f"staging bucket : {args.staging_bucket}")
    print(f"requirements   : {len(requirements)} pinned")
    print("extra packages : src/")
    print(f"target         : {'CREATE new engine' if args.create or not existing else existing}")

    if args.dry_run:
        print("\n(dry run — nothing sent)")
        for line in requirements:
            print(f"    {line}")
        return 0

    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    # Imported only now: constructing the agent is inert, but importing
    # src.agent pulls in the telemetry policy, and that should not run during a
    # --dry-run or an argument error.
    from src.agent import build_agent

    agent = build_agent()

    common = dict(
        requirements=requirements,
        extra_packages=["src"],
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
    )

    if existing and not args.create:
        print(f"\nUpdating {existing} ...")
        remote = agent_engines.update(resource_name=existing, agent_engine=agent, **common)
    else:
        print("\nCreating a new Agent Engine ...")
        remote = agent_engines.create(agent_engine=agent, **common)

    resource_name = getattr(remote, "resource_name", None) or str(remote)
    write_metadata(resource_name)

    print(f"\n✅ Deployed: {resource_name}")
    print(
        "\nPoint the proxy at it:\n"
        f"    AGENT_ENGINE_ENDPOINT=https://{args.region}-aiplatform.googleapis.com/v1/"
        f"{resource_name}:query"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
