#!/bin/bash
# =============================================================================
# deploy_to_gcp.sh — One-step deployment of the CausalTraceAI stack
#
# Deploys the full architecture in order:
#   1. agent   → Vertex AI Agent Engine (src/ via deploy_agent.py, in-place update)
#   2. proxy   → Cloud Run service `causaltraceai-app` (proxy/ via Dockerfile.proxy)
#   3. hosting → Firebase Hosting (proxy/static, rewrites /* → Cloud Run)
#
# Usage:
#   ./deploy_to_gcp.sh                  # deploy all three stages
#   ./deploy_to_gcp.sh --only agent     # just the Agent Engine
#   ./deploy_to_gcp.sh --only proxy     # just the Cloud Run proxy
#   ./deploy_to_gcp.sh --only hosting   # just Firebase Hosting
#
#   # A preview copy on its own Cloud Run URL, leaving production alone.
#   # Pair it with --only proxy: the hosting stage publishes the live domain,
#   # and the agent stage updates the one shared engine in place.
#   DEPLOY_SERVICE_NAME=causaltraceai-app-staging APP_URL=https://... \
#     ./deploy_to_gcp.sh --only proxy
#
# Requires: gcloud (authed), python + the agent deps, docker, firebase CLI or npx.
# Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_REGION / AGENT_ENGINE_ENDPOINT
# from .env.
# =============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ONLY="all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --only) ONLY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

case "$ONLY" in
    all|agent|proxy|hosting) ;;
    *) echo "Invalid --only value: $ONLY (expected agent, proxy, or hosting)"; exit 1 ;;
esac

if [ -f ".env" ]; then
    source .env
else
    echo "Error: .env file missing. Needed for GOOGLE_CLOUD_PROJECT."
    exit 1
fi

PROJECT_ID=${GOOGLE_CLOUD_PROJECT}
REGION=${GOOGLE_CLOUD_REGION:-europe-west2}
# DEPLOY_SERVICE_NAME retargets the whole proxy stage at another Cloud Run
# service, which is how the staging workflow gets a full copy of the app on its
# own URL without touching the live one. Deliberately not named SERVICE_NAME or
# IMAGE_NAME: .env already carries an IMAGE_NAME for the GKE path, and this file
# sources .env, so reusing either name would let a local .env silently retarget
# a production deploy.
SERVICE_NAME="${DEPLOY_SERVICE_NAME:-causaltraceai-app}"
# Tagged per service, so a staging build can never be promoted to production by
# a `latest` tag that both services happen to share.
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# ── Stage 1: Agent Engine ────────────────────────────────────────────────────
# deploy_agent.py, not agents-cli: agents-cli packages an ADK agent *directory*,
# and this project's agent is a LangGraph runnable wrapped in LanggraphAgent —
# the SDK's own create/update flow is what accepts that. The script updates the
# engine recorded in deployment_metadata.json, or creates one if there is none,
# and rewrites that file with the resource name either way.
if [ "$ONLY" = "all" ] || [ "$ONLY" = "agent" ]; then
    echo "▶ [1/3] Deploying LangGraph agent (src/) to Vertex AI Agent Engine..."
    : "${AGENT_ENGINE_STAGING_BUCKET:?Set AGENT_ENGINE_STAGING_BUCKET=gs://... in .env (the SDK stages the packaged agent there)}"
    python deploy_agent.py \
        --project "${PROJECT_ID}" \
        --region "${REGION}" \
        --staging-bucket "${AGENT_ENGINE_STAGING_BUCKET}"
fi

# ── Stage 2: Cloud Run proxy ─────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "proxy" ]; then
    echo "▶ [2/3] Deploying proxy (proxy/) to Cloud Run service ${SERVICE_NAME}..."
    gcloud auth configure-docker gcr.io --quiet
    docker build -f Dockerfile.proxy -t "${IMAGE_NAME}" .
    docker push "${IMAGE_NAME}"

    DEPLOY_ARGS=(
        --image "${IMAGE_NAME}"
        --region "${REGION}"
        --project "${PROJECT_ID}"
        --allow-unauthenticated
        # Explicit, not left to default: `gcloud run deploy` only preserves an
        # existing service's service account on a REDEPLOY. The very first
        # deploy of any new SERVICE_NAME (every dev push, every PR preview)
        # falls back to the project's default Compute Engine SA, which has
        # none of agent-app-sa's grants (datastore.user, aiplatform.user) —
        # every Firestore write then fails with a 403 that has nothing to do
        # with the code path that triggered it. Reusing the same runtime SA
        # across all three tiers is safe: its Firestore role is project-scoped
        # (applies to every named database, not just "causaltraceai"), and
        # per-tier isolation already comes from FIRESTORE_DATABASE_ID plus a
        # separate Cloud Run service, not from a separate identity.
        --service-account "agent-app-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    )
    # Point the proxy at the Agent Engine. deployment_metadata.json is kept
    # current by deploy_agent.py on every agent deploy, so it is the source of
    # truth; an explicit AGENT_ENGINE_ENDPOINT env var overrides it.
    #
    # Both key names are read. deploy_agent.py writes "remote_agent_engine_id";
    # files left over from the agents-cli era carry "remote_agent_runtime_id".
    # This used to match only the latter, so against a metadata file written by
    # deploy_agent.py the sed found nothing, AGENT_ENGINE_ENDPOINT stayed unset,
    # and the proxy deployed perfectly — serving proxy/mockdata.py to real
    # users. Nothing in the deploy output or the service logs said so.
    if [ -z "${AGENT_ENGINE_ENDPOINT:-}" ] && [ -f deployment_metadata.json ]; then
        # Two -e expressions rather than one with \| alternation: that is a GNU
        # sed extension and this script also runs on macOS, where BSD sed treats
        # it as a literal and silently matches nothing.
        ENGINE_ID=$(sed -n \
            -e 's/.*"remote_agent_engine_id": *"\([^"]*\)".*/\1/p' \
            -e 's/.*"remote_agent_runtime_id": *"\([^"]*\)".*/\1/p' \
            deployment_metadata.json | head -1)
        if [ -n "${ENGINE_ID}" ]; then
            # v1, matching what deploy_agent.py prints on success. Pinned rather
            # than split across scripts, and overridable for a preview surface.
            AGENT_ENGINE_ENDPOINT="https://${REGION}-aiplatform.googleapis.com/${AGENT_ENGINE_API_VERSION:-v1}/${ENGINE_ID}:query"
        fi
    fi
    # Fail rather than deploy a proxy with no engine: an unset endpoint is not a
    # degraded deploy, it is a silently fake one. proxy/main.py falls back to its
    # offline mock whenever AGENT_ENGINE_ENDPOINT is absent, and that path
    # answers every question with canned numbers.
    if [ -z "${AGENT_ENGINE_ENDPOINT:-}" ]; then
        echo "ERROR: could not determine AGENT_ENGINE_ENDPOINT." >&2
        echo "       deployment_metadata.json is missing or has no engine id, and" >&2
        echo "       AGENT_ENGINE_ENDPOINT is not set. Deploying now would put a" >&2
        echo "       proxy live that serves proxy/mockdata.py to every visitor." >&2
        echo "       Run './deploy_to_gcp.sh --only agent' first, or export" >&2
        echo "       AGENT_ENGINE_ENDPOINT=https://.../reasoningEngines/ID:query" >&2
        exit 1
    fi
    # proxy/main.py builds the streaming URL by substituting ':streamQuery' for
    # ':query', and proxy/admin.py splits on the same suffix for session deletes.
    case "${AGENT_ENGINE_ENDPOINT}" in
        *:query|*:streamQuery) ;;
        *) echo "ERROR: AGENT_ENGINE_ENDPOINT must end in ':query' (got ${AGENT_ENGINE_ENDPOINT})" >&2
           exit 1 ;;
    esac
    DEPLOY_ARGS+=(--update-env-vars "AGENT_ENGINE_ENDPOINT=${AGENT_ENGINE_ENDPOINT}")
    # Cross-origin allow-list, so the app (Firebase Hosting) can call the
    # Cloud Run service directly (e.g. via api.causaltraceai.com) and bypass
    # Hosting's 60s timeout on long causal runs. Empty = same-origin only.
    # Requires a separate `gcloud run domain-mappings create` + DNS record for
    # the api subdomain, and TRACERLENS_API_BASE set in the frontend (see ui/src/lib/api.ts).
    CORS_ORIGINS="${CORS_ORIGINS:-https://causaltraceai.com,https://api.causaltraceai.com}"
    if [ -n "${CORS_ORIGINS}" ]; then
        # ^##^ sets '##' as the delimiter so the commas inside the value are
        # not parsed by gcloud as separate env-var assignments.
        DEPLOY_ARGS+=(--update-env-vars "^##^CORS_ORIGINS=${CORS_ORIGINS}")
    fi

    # ── Access gate (docs/access_control.md) ─────────────────────────────────
    # Only forwarded when present in the environment, so a partial deploy never
    # blanks a value already set on the service. The two secrets are the ones
    # that matter: without ACCESS_SIGNING_SECRET every cold start signs all
    # users out, and without ADMIN_TOKEN the dashboard answers 503.
    # FIRESTORE_DATABASE_ID is here so a staging service can be pointed at its
    # own database; unset, proxy/access.py:get_db() defaults to "tracerlensai"
    # (the name predates the rename and is what the live database is actually
    # called — do not "fix" it in access.py without migrating the data) and
    # staging shares production's records. SMTP_* mirrors RESEND_API_KEY as the
    # alternative mail transport.
    for VAR in ACCESS_SIGNING_SECRET ADMIN_TOKEN RESEND_API_KEY \
               SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD \
               ACCESS_NOTIFY_EMAIL ACCESS_FROM_EMAIL APP_URL \
               FIRESTORE_DATABASE_ID \
               ACCESS_TOKEN_LIMIT ACCESS_TOKEN_GRANT \
               CHAT_RETENTION_HOURS RUN_METRICS_RETENTION_DAYS; do
        if [ -n "${!VAR:-}" ]; then
            DEPLOY_ARGS+=(--update-env-vars "${VAR}=${!VAR}")
        fi
    done
    if [ -z "${ACCESS_SIGNING_SECRET:-}" ]; then
        echo "WARNING: ACCESS_SIGNING_SECRET is not set in this environment." >&2
        echo "         Leaving whatever the service already has. If it has none," >&2
        echo "         every cold start invalidates all sessions." >&2
    fi
    if [ -z "${APP_URL:-}" ]; then
        echo "WARNING: APP_URL is not set in this environment." >&2
        echo "         Leaving whatever the service already has. If it has none," >&2
        echo "         the revision now refuses to start rather than mail out" >&2
        echo "         sign-in links pointing at http://localhost:8080." >&2
        echo "         Set it to the public origin, e.g.:" >&2
        echo "           export APP_URL=https://causaltraceai.com" >&2
    fi

    gcloud run deploy "${SERVICE_NAME}" "${DEPLOY_ARGS[@]}"
fi

# ── Stage 3: Firebase Hosting ────────────────────────────────────────────────
# Builds the React UI, then publishes ui/dist and the rewrite rule
# (firebase.json) that routes /analyze-prompt and every other non-static path
# to the Cloud Run proxy.
if [ "$ONLY" = "all" ] || [ "$ONLY" = "hosting" ]; then
    echo "▶ [3/3] Building UI and deploying Firebase Hosting (ui/dist + rewrites)..."

    # Publishing a rewrite that names a service which does not exist takes the
    # whole site down: Hosting accepts the config and every non-static path then
    # 404s at the edge. That is exactly what `--only hosting` would have done
    # while firebase.json still said "tracerlensai-app" and this script deployed
    # "causaltraceai-app" — two names that were never reconciled. Check first.
    REWRITE_SERVICE=$(sed -n 's/.*"serviceId": *"\([^"]*\)".*/\1/p' firebase.json | head -1)
    if [ -n "${REWRITE_SERVICE}" ]; then
        if gcloud run services describe "${REWRITE_SERVICE}" \
                --region "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
            echo "  firebase.json rewrites → ${REWRITE_SERVICE} (exists)"
        else
            echo "ERROR: firebase.json rewrites to Cloud Run service '${REWRITE_SERVICE}'," >&2
            echo "       which does not exist in ${REGION}. Deploying hosting now would" >&2
            echo "       make every non-static path 404." >&2
            echo "       Either deploy the proxy first (./deploy_to_gcp.sh --only proxy," >&2
            echo "       which creates ${SERVICE_NAME}), or point firebase.json at a" >&2
            echo "       service that already exists:" >&2
            gcloud run services list --project "${PROJECT_ID}" \
                --format='value(metadata.name)' 2>/dev/null | sed 's/^/         - /' >&2
            exit 1
        fi
    fi

    (cd ui && npm ci && npm run build)
    if command -v firebase >/dev/null; then
        firebase deploy --only hosting --project "${PROJECT_ID}" --non-interactive
    else
        npx -y firebase-tools deploy --only hosting --project "${PROJECT_ID}" --non-interactive
    fi
fi

# Report what was actually deployed. A dev or preview run targets a different
# Cloud Run service via DEPLOY_SERVICE_NAME and never touches Firebase
# Hosting, so claiming the live domain unconditionally would tell a preview
# deploy it had just published production.
if [ "$ONLY" = "hosting" ] || { [ "$ONLY" = "all" ] && [ "${SERVICE_NAME}" = "causaltraceai-app" ]; }; then
    echo "✅ Deployment finished. Live at https://causaltraceai.com"
elif [ "${SERVICE_NAME}" = "causaltraceai-app" ]; then
    echo "✅ Deployment finished. Service ${SERVICE_NAME} (${REGION}) — serving https://causaltraceai.com"
else
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region "${REGION}" --project "${PROJECT_ID}" \
        --format='value(status.url)' 2>/dev/null || true)
    echo "✅ Deployment finished. Service ${SERVICE_NAME} (${REGION})${SERVICE_URL:+ — ${SERVICE_URL}}"
    echo "   Production (https://causaltraceai.com) was not touched."
fi
