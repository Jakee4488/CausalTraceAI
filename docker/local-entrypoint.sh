#!/bin/sh
# Local-dev-only entrypoint for the proxy container (docker-compose.yml).
#
# Not used by Dockerfile.proxy's own CMD / the deployed Cloud Run image Ã¢â‚¬â€ this
# is bind-mounted in only for `docker compose up`, so it never touches
# production. It resolves AGENT_ENGINE_ENDPOINT the same way deploy_to_gcp.sh
# does: from deployment_metadata.json, which agents-cli keeps current on every
# agent deploy. That makes this compose file self-updating Ã¢â‚¬â€ redeploy the
# agent, and the next `docker compose up` picks up the new engine automatically.
set -e

if [ "$MODE" = "mock" ]; then
    echo "[local-entrypoint] MODE=mock Ã¢â‚¬â€ AGENT_ENGINE_ENDPOINT left unset; proxy serves its built-in canned responses."
    unset AGENT_ENGINE_ENDPOINT
    # The access gate reads a record on every request, so without an in-memory
    # store mock mode would still need Firestore credentials just to open the
    # chat. Approve yourself from the link printed in these logs.
    export ACCESS_STORE="${ACCESS_STORE:-memory}"
    export ADMIN_TOKEN="${ADMIN_TOKEN:-local-admin}"
    export APP_URL="${APP_URL:-http://localhost:8080}"
    echo "[local-entrypoint] ACCESS_STORE=$ACCESS_STORE — access records are in-memory and reset on restart."
    echo "[local-entrypoint] Emails are printed here, not sent. Admin dashboard: $APP_URL/admin (password: $ADMIN_TOKEN)"
elif [ -z "$AGENT_ENGINE_ENDPOINT" ] && [ -f /app/deployment_metadata.json ]; then
    # Both key names: deploy_agent.py writes "remote_agent_engine_id", while
    # files from the agents-cli era carry "remote_agent_runtime_id". Matching
    # only the latter meant a metadata file written by the current deploy script
    # resolved to nothing and compose came up in mock mode looking healthy.
    ENGINE_ID=$(sed -n \
        -e 's/.*"remote_agent_engine_id": *"\([^"]*\)".*/\1/p' \
        -e 's/.*"remote_agent_runtime_id": *"\([^"]*\)".*/\1/p' \
        /app/deployment_metadata.json | head -1)
    if [ -n "$ENGINE_ID" ]; then
        export AGENT_ENGINE_ENDPOINT="https://${GOOGLE_CLOUD_REGION:-europe-west2}-aiplatform.googleapis.com/${AGENT_ENGINE_API_VERSION:-v1}/${ENGINE_ID}:query"
        echo "[local-entrypoint] Resolved from deployment_metadata.json -> AGENT_ENGINE_ENDPOINT=$AGENT_ENGINE_ENDPOINT"
    else
        echo "[local-entrypoint] deployment_metadata.json present but no remote_agent_engine_id / remote_agent_runtime_id; falling back to mock mode."
    fi
else
    echo "[local-entrypoint] Using AGENT_ENGINE_ENDPOINT from the environment: $AGENT_ENGINE_ENDPOINT"
fi

# Use --reload when the proxy volume is mounted (dev); production CMD skips it.
exec uvicorn proxy.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir /app/proxy
