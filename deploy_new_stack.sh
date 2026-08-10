#!/usr/bin/env bash
# =============================================================================
# deploy_new_stack.sh — stand up a NEW Agent Engine and a NEW Cloud Run proxy
#
# deploy_to_gcp.sh updates the *existing* stack in place: it reuses the engine
# recorded in deployment_metadata.json and redeploys the one production Cloud
# Run service. This script is the other operation — bringing a second, fully
# independent stack into existence beside the current one:
#
#     Agent Engine  : a brand new reasoningEngine (agent_engines.create)
#     Cloud Run     : a brand new service, causaltraceai-app-<stack>
#     Firestore     : its own named database, so records never mix
#     Hosting       : NOT touched unless you pass --promote-hosting
#
# Use it for a v2 rollout, a staging tier, a region migration, or a rebuild
# after the current engine has drifted from what src/ actually contains.
#
# Usage:
#   ./deploy_new_stack.sh --stack v2 --app-url https://v2.causaltraceai.com
#   ./deploy_new_stack.sh --stack v2 --dry-run          # plan only, no writes
#   ./deploy_new_stack.sh --stack v2 --skip-agent       # reuse this stack's engine
#   ./deploy_new_stack.sh --stack v2 --promote-hosting  # then repoint the domain
#
# Everything is namespaced by --stack. Nothing this script does can modify the
# production engine, the production service, or deployment_metadata.json.
#
# Requires: bash, gcloud (authenticated), docker, python (with the agent deps),
# curl. On Windows run it from Git Bash, not PowerShell.
# =============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────────────
STACK=""
DRY_RUN=0
ASSUME_YES=0
SKIP_AGENT=0
SKIP_PROXY=0
RUN_SMOKE=1
PROMOTE_HOSTING=0
# v1, matching what deploy_agent.py prints on success. deploy_to_gcp.sh builds
# a v1beta1 URL for the same resource; both surfaces expose :streamQuery, but
# they are not interchangeable for every field, so this is pinned and
# overridable rather than left to whichever script wrote it last.
API_VERSION="${AGENT_ENGINE_API_VERSION:-v1}"
RUNTIME_SA=""
CPU="2"
MEMORY="2Gi"
TIMEOUT="900"       # causal runs stream for minutes; the 300s default cuts them
CONCURRENCY="40"
MIN_INSTANCES="0"
MAX_INSTANCES="4"

# Prints the header block above, which is the only copy of the usage text.
# The range stops at the closing rule on line 29 — extend it if the header grows.
usage() { sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --stack)            STACK="$2"; shift ;;
        --project)          GOOGLE_CLOUD_PROJECT="$2"; shift ;;
        --region)           GOOGLE_CLOUD_REGION="$2"; shift ;;
        --staging-bucket)   AGENT_ENGINE_STAGING_BUCKET="$2"; shift ;;
        --firestore-db)     FIRESTORE_DATABASE_ID="$2"; shift ;;
        --app-url)          APP_URL="$2"; shift ;;
        --cors-origins)     CORS_ORIGINS="$2"; shift ;;
        --runtime-sa)       RUNTIME_SA="$2"; shift ;;
        --api-version)      API_VERSION="$2"; shift ;;
        --min-instances)    MIN_INSTANCES="$2"; shift ;;
        --max-instances)    MAX_INSTANCES="$2"; shift ;;
        --skip-agent)       SKIP_AGENT=1 ;;
        --skip-proxy)       SKIP_PROXY=1 ;;
        --no-smoke)         RUN_SMOKE=0 ;;
        --promote-hosting)  PROMOTE_HOSTING=1 ;;
        --dry-run)          DRY_RUN=1 ;;
        -y|--yes)           ASSUME_YES=1 ;;
        -h|--help)          usage 0 ;;
        *) echo "Unknown parameter: $1" >&2; usage 1 ;;
    esac
    shift
done

# ── Output helpers ───────────────────────────────────────────────────────────
if [ -t 1 ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
    YLW=$'\033[33m'; RST=$'\033[0m'
else
    B=""; DIM=""; RED=""; GRN=""; YLW=""; RST=""
fi
step() { echo; echo "${B}▶ $*${RST}"; }
ok()   { echo "  ${GRN}✓${RST} $*"; }
warn() { echo "  ${YLW}!${RST} $*" >&2; }
die()  { echo "${RED}✗ $*${RST}" >&2; exit 1; }
note() { echo "  ${DIM}$*${RST}"; }

# Every mutating call goes through this, so --dry-run is enforced in one place
# rather than by remembering to guard each call site.
run() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "  ${DIM}[dry-run]${RST} $*"
        return 0
    fi
    "$@"
}

confirm() {
    [ "$ASSUME_YES" = "1" ] && return 0
    [ "$DRY_RUN" = "1" ] && return 0
    local reply
    read -r -p "  $1 [y/N] " reply </dev/tty
    [[ "$reply" =~ ^[Yy] ]] || die "Aborted."
}

# ── Configuration ────────────────────────────────────────────────────────────
# .env is sourced for convenience but never required: a fresh stack is often
# built from an operator shell that has the values exported directly, and
# demanding a file there just invites a stale one being committed.
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
    note "Loaded .env"
else
    note "No .env — reading configuration from the environment and flags"
fi

[ -n "$STACK" ] || die "--stack is required (e.g. --stack v2). It namespaces the
  Cloud Run service, the image tag and the metadata file, which is what keeps
  this run from colliding with production."
[[ "$STACK" =~ ^[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?$ ]] || die \
    "--stack must be lowercase alphanumeric with hyphens, <=20 chars ('$STACK'
  is not). Cloud Run service names are DNS labels and this becomes a suffix."

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${GOOGLE_CLOUD_REGION:-europe-west2}"
[ -n "$PROJECT_ID" ] || die "GOOGLE_CLOUD_PROJECT is not set (or pass --project)."

SERVICE_NAME="causaltraceai-app-${STACK}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
# Namespaced so `deploy_agent.py --create` can never overwrite the production
# engine's record. deploy_agent.py always writes ./deployment_metadata.json, so
# the agent stage below swaps that file out around the call and restores it.
METADATA_FILE="deployment_metadata.${STACK}.json"
PROD_METADATA="deployment_metadata.json"
# Its own database by default: proxy/access.py reads a record on every request
# and a shared database would let a new stack's bugs corrupt live users' quota
# and history. Override with --firestore-db to deliberately share one.
FIRESTORE_DATABASE_ID="${FIRESTORE_DATABASE_ID:-causaltraceai-${STACK}}"
RUNTIME_SA="${RUNTIME_SA:-agent-app-sa@${PROJECT_ID}.iam.gserviceaccount.com}"

echo
echo "${B}CausalTraceAI — new stack deployment${RST}"
echo "  stack           : ${STACK}"
echo "  project         : ${PROJECT_ID}"
echo "  region          : ${REGION}"
echo "  cloud run       : ${SERVICE_NAME}  ${DIM}(new)${RST}"
echo "  image           : ${IMAGE_NAME}"
echo "  agent engine    : $([ "$SKIP_AGENT" = 1 ] && echo "reuse ${METADATA_FILE}" || echo "CREATE NEW")"
echo "  firestore db    : ${FIRESTORE_DATABASE_ID}"
echo "  runtime SA      : ${RUNTIME_SA}"
echo "  metadata        : ${METADATA_FILE}"
echo "  hosting         : $([ "$PROMOTE_HOSTING" = 1 ] && echo "${YLW}WILL REPOINT the live domain${RST}" || echo "untouched")"
[ "$DRY_RUN" = "1" ] && echo "  ${YLW}mode            : DRY RUN — nothing will be created${RST}"

# =============================================================================
# Phase 1 — Preflight
#
# Every check runs before anything is created, and failures accumulate rather
# than aborting on the first one. A half-built stack (engine created, proxy
# failed on a missing role) is the expensive failure mode: the engine is
# already billing and the next run creates a second one.
# =============================================================================
step "[1/6] Preflight"
FAILURES=()
fail() { FAILURES+=("$1"); echo "  ${RED}✗${RST} $1"; }

for tool in gcloud docker python curl; do
    if command -v "$tool" >/dev/null 2>&1; then ok "$tool present"
    else fail "$tool is not on PATH"; fi
done

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1 || true)"
if [ -n "$ACTIVE_ACCOUNT" ]; then ok "gcloud authenticated as ${ACTIVE_ACCOUNT}"
else fail "gcloud has no active account — run: gcloud auth login"; fi

if gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
    PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
    ok "project ${PROJECT_ID} reachable (number ${PROJECT_NUMBER})"
else
    fail "cannot describe project ${PROJECT_ID} — wrong project id, or no access"
    PROJECT_NUMBER=""
fi

# ADC is what deploy_agent.py and the Vertex SDK authenticate with. `gcloud auth
# login` alone does not create it, and the resulting failure surfaces deep
# inside the SDK as an opaque DefaultCredentialsError.
if python -c "import google.auth, sys; google.auth.default(); sys.exit(0)" >/dev/null 2>&1; then
    ok "application default credentials resolve"
else
    fail "no ADC — run: gcloud auth application-default login"
fi

# Required APIs. artifactregistry is here because gcr.io hostnames are served by
# Artifact Registry now; a project where only container.googleapis.com was ever
# enabled will fail the push with a 403 that reads like an auth problem.
REQUIRED_APIS=(aiplatform.googleapis.com run.googleapis.com
               artifactregistry.googleapis.com firestore.googleapis.com
               storage.googleapis.com iamcredentials.googleapis.com)
if [ -n "$PROJECT_NUMBER" ]; then
    ENABLED="$(gcloud services list --enabled --project "$PROJECT_ID" --format='value(config.name)' 2>/dev/null || true)"
    for api in "${REQUIRED_APIS[@]}"; do
        if grep -qx "$api" <<<"$ENABLED"; then ok "api enabled: ${api}"
        else fail "api not enabled: ${api} — gcloud services enable ${api} --project ${PROJECT_ID}"; fi
    done
fi

# Staging bucket: the SDK uploads the pickled agent here. Only needed when the
# agent stage will actually run.
if [ "$SKIP_AGENT" = "0" ]; then
    if [ -z "${AGENT_ENGINE_STAGING_BUCKET:-}" ]; then
        fail "AGENT_ENGINE_STAGING_BUCKET is not set (or pass --staging-bucket).
      The Vertex SDK stages the packaged agent there; it is not optional."
    elif [[ "$AGENT_ENGINE_STAGING_BUCKET" != gs://* ]]; then
        fail "AGENT_ENGINE_STAGING_BUCKET must start with gs:// (got '${AGENT_ENGINE_STAGING_BUCKET}')"
    elif gcloud storage ls "$AGENT_ENGINE_STAGING_BUCKET" >/dev/null 2>&1; then
        ok "staging bucket ${AGENT_ENGINE_STAGING_BUCKET} reachable"
    else
        fail "staging bucket ${AGENT_ENGINE_STAGING_BUCKET} is missing or unreadable —
      gcloud storage buckets create ${AGENT_ENGINE_STAGING_BUCKET} --location=${REGION}"
    fi
fi

# Runtime service account and its roles. gcloud run deploy only preserves a
# service account across a REDEPLOY; a first deploy of a new service silently
# falls back to the default Compute Engine SA, which holds none of these grants.
if gcloud iam service-accounts describe "$RUNTIME_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
    ok "runtime service account ${RUNTIME_SA} exists"
    if [ -n "$PROJECT_NUMBER" ]; then
        POLICY="$(gcloud projects get-iam-policy "$PROJECT_ID" \
            --flatten='bindings[].members' \
            --filter="bindings.members:serviceAccount:${RUNTIME_SA}" \
            --format='value(bindings.role)' 2>/dev/null || true)"
        # datastore.user is the one that bites: terraform/iam.tf grants
        # aiplatform.user, bigquery.dataEditor and logging.logWriter but NOT
        # this, while proxy/access.py touches Firestore on every single request.
        # Without it the service starts clean and 403s the first time a user
        # signs in — a failure that looks nothing like a missing IAM role.
        for role in roles/aiplatform.user roles/datastore.user roles/logging.logWriter; do
            if grep -qx "$role" <<<"$POLICY"; then ok "  ${RUNTIME_SA} has ${role}"
            else fail "${RUNTIME_SA} is missing ${role} —
      gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
        --member=serviceAccount:${RUNTIME_SA} --role=${role}"; fi
        done
    fi
else
    fail "runtime service account ${RUNTIME_SA} does not exist —
      gcloud iam service-accounts create agent-app-sa --project ${PROJECT_ID}"
fi

# Firestore database. proxy/access.py opens a *named* database and does not
# create one; a wrong name fails per-request, not at boot.
if gcloud firestore databases describe --database="$FIRESTORE_DATABASE_ID" \
        --project "$PROJECT_ID" >/dev/null 2>&1; then
    ok "firestore database '${FIRESTORE_DATABASE_ID}' exists"
else
    fail "firestore database '${FIRESTORE_DATABASE_ID}' does not exist —
      gcloud firestore databases create --database=${FIRESTORE_DATABASE_ID} \\
        --location=${REGION} --type=firestore-native --project=${PROJECT_ID}"
fi

# Secrets and origins. These are warnings, not failures: the service starts
# without them, it just behaves badly in ways worth naming out loud.
if [ -z "${ACCESS_SIGNING_SECRET:-}" ]; then
    warn "ACCESS_SIGNING_SECRET is unset — the proxy will mint an ephemeral
      per-process key and every cold start signs all users out. Generate one:
      export ACCESS_SIGNING_SECRET=\$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
else
    ok "ACCESS_SIGNING_SECRET set (${#ACCESS_SIGNING_SECRET} chars)"
fi
if [ -z "${ADMIN_TOKEN:-}" ]; then
    warn "ADMIN_TOKEN is unset — /admin will answer 503 and nobody can approve
      the first user, which on a new stack means nobody can use it at all."
else
    ok "ADMIN_TOKEN set"
fi
if [ -z "${APP_URL:-}" ]; then
    fail "APP_URL is not set (or pass --app-url). proxy/access.py refuses to
      start on a managed runtime without it, rather than mail sign-in links
      pointing at http://localhost:8080. For a stack with no domain of its own
      use the Cloud Run URL — deploy once with --skip-proxy to learn it, or set
      it to the service URL pattern and redeploy after the first revision."
else
    ok "APP_URL=${APP_URL}"
fi
if [ -z "${RESEND_API_KEY:-}${SMTP_HOST:-}" ]; then
    warn "Neither RESEND_API_KEY nor SMTP_HOST is set — sign-in emails cannot be
      delivered. Approve users from the links printed in Cloud Logging instead."
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo
    die "${#FAILURES[@]} preflight check(s) failed. Nothing was created. Fix the
  items marked ✗ above and re-run — every one of them is a failure that would
  otherwise surface after the engine is already billing."
fi
ok "preflight clean"

[ "$PROMOTE_HOSTING" = "1" ] && confirm \
    "--promote-hosting will repoint https://causaltraceai.com at ${SERVICE_NAME}. Continue?"

# =============================================================================
# Phase 2 — Agent Engine (new reasoningEngine)
#
# deploy_agent.py drives agent_engines.create; this stage only handles the part
# that script does not: keeping the new engine's identity out of production's
# metadata file.
# =============================================================================
if [ "$SKIP_AGENT" = "0" ]; then
    step "[2/6] Creating a new Agent Engine"

    # deploy_agent.py writes ./deployment_metadata.json unconditionally, and
    # with --create it would replace production's recorded engine id with the
    # new one — after which every later `deploy_to_gcp.sh --only agent` would
    # update the NEW engine while believing it was updating production. Move the
    # file aside for the duration and restore it on every exit path.
    RESTORE_PROD_METADATA=0
    if [ -f "$PROD_METADATA" ] && [ "$DRY_RUN" = "0" ]; then
        cp "$PROD_METADATA" "${PROD_METADATA}.bak"
        RESTORE_PROD_METADATA=1
        note "Preserved ${PROD_METADATA} → ${PROD_METADATA}.bak"
    fi
    restore_prod_metadata() {
        if [ "$RESTORE_PROD_METADATA" = "1" ] && [ -f "${PROD_METADATA}.bak" ]; then
            mv -f "${PROD_METADATA}.bak" "$PROD_METADATA"
            note "Restored ${PROD_METADATA}"
        fi
    }
    trap restore_prod_metadata EXIT

    note "python deploy_agent.py --create  (this takes several minutes: the SDK"
    note "pickles the agent, uploads src/, and builds the managed environment"
    note "from requirements.txt — torch and pyro make that a large image)"

    run python deploy_agent.py \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --staging-bucket "${AGENT_ENGINE_STAGING_BUCKET:-}" \
        --create

    if [ "$DRY_RUN" = "0" ]; then
        [ -f "$PROD_METADATA" ] || die "deploy_agent.py did not write ${PROD_METADATA};
  the engine may or may not have been created. Check:
    gcloud ai reasoning-engines list --region=${REGION} --project=${PROJECT_ID}"
        mv -f "$PROD_METADATA" "$METADATA_FILE"
        ok "New engine recorded in ${METADATA_FILE}"
    fi
    restore_prod_metadata
    trap - EXIT
else
    step "[2/6] Agent Engine — skipped (--skip-agent)"
fi

# =============================================================================
# Phase 3 — Resolve the endpoint
#
# This is where deploy_to_gcp.sh has a live defect worth knowing about: it reads
# "remote_agent_runtime_id" from the metadata file, but deploy_agent.py writes
# "remote_agent_engine_id". The sed finds nothing, AGENT_ENGINE_ENDPOINT is
# never set, and the proxy deploys perfectly and serves its offline mock — a
# stack that looks healthy and answers every question with canned data.
# Both key names are accepted here, and a missing endpoint is fatal.
# =============================================================================
step "[3/6] Resolving the Agent Engine endpoint"

if [ -n "${AGENT_ENGINE_ENDPOINT:-}" ]; then
    ok "Using AGENT_ENGINE_ENDPOINT from the environment (overrides ${METADATA_FILE})"
elif [ -f "$METADATA_FILE" ]; then
    ENGINE_ID="$(python - "$METADATA_FILE" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
# deploy_agent.py writes remote_agent_engine_id; older files from the agents-cli
# era used remote_agent_runtime_id. Accept both so a metadata file written by
# either tool resolves.
print(meta.get("remote_agent_engine_id") or meta.get("remote_agent_runtime_id") or "")
PY
)"
    [ -n "$ENGINE_ID" ] || die "${METADATA_FILE} has neither remote_agent_engine_id
  nor remote_agent_runtime_id. Refusing to deploy a proxy with no engine — it
  would start cleanly and serve proxy/mockdata.py to real users."
    AGENT_ENGINE_ENDPOINT="https://${REGION}-aiplatform.googleapis.com/${API_VERSION}/${ENGINE_ID}:query"
    ok "Resolved from ${METADATA_FILE}"
elif [ "$DRY_RUN" = "1" ]; then
    AGENT_ENGINE_ENDPOINT="https://${REGION}-aiplatform.googleapis.com/${API_VERSION}/projects/${PROJECT_ID}/locations/${REGION}/reasoningEngines/DRY-RUN:query"
    warn "No metadata file yet — using a placeholder endpoint for the plan"
else
    die "No ${METADATA_FILE} and no AGENT_ENGINE_ENDPOINT. Run without
  --skip-agent to create an engine, or export AGENT_ENGINE_ENDPOINT."
fi

# proxy/main.py derives the streaming URL as endpoint.replace(":query",
# ":streamQuery"), and proxy/admin.py splits on the same suffix to build the
# session-delete URL. An endpoint without it produces a POST to the wrong
# method and a broken 24-hour retention sweep.
[[ "$AGENT_ENGINE_ENDPOINT" == *:query || "$AGENT_ENGINE_ENDPOINT" == *:streamQuery ]] || die \
    "AGENT_ENGINE_ENDPOINT must end in ':query' — proxy/main.py builds the
  streaming URL by substituting ':streamQuery' for it.
  Got: ${AGENT_ENGINE_ENDPOINT}"
note "${AGENT_ENGINE_ENDPOINT}"

# Recover the bare resource name from the endpoint when it did not come from a
# metadata file, so the rollback instructions at the end print a real id rather
# than a placeholder the operator has to go and look up mid-incident.
if [ -z "${ENGINE_ID:-}" ]; then
    ENGINE_ID="${AGENT_ENGINE_ENDPOINT#*/${API_VERSION}/}"
    ENGINE_ID="${ENGINE_ID%:query}"
    ENGINE_ID="${ENGINE_ID%:streamQuery}"
fi

# Prove the engine answers before building an image that points at it. This
# calls the real thing, so it costs two Gemini calls and a Monte Carlo run.
if [ "$RUN_SMOKE" = "1" ] && [ "$DRY_RUN" = "0" ] && [ "$SKIP_AGENT" = "0" ]; then
    note "Probing the engine directly (bypasses the proxy and the access gate)…"
    PROBE_URL="${AGENT_ENGINE_ENDPOINT/:query/:streamQuery}"
    PROBE_BODY='{"class_method":"stream_query","input":{"input":{"user_input":"Demand is 13 units. Which policy should I choose?"},"stream_mode":"updates"}}'
    PROBE_OUT="$(mktemp)"
    PROBE_CODE="$(curl -sS -o "$PROBE_OUT" -w '%{http_code}' -X POST "$PROBE_URL" \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json" \
        --max-time 600 -d "$PROBE_BODY" || echo 000)"
    if [ "$PROBE_CODE" = "200" ]; then
        ok "Engine responded 200 ($(wc -c <"$PROBE_OUT" | tr -d ' ') bytes streamed)"
        grep -q 'causal_steps\|explain_result' "$PROBE_OUT" \
            && ok "Response carries causal trace lines — the graph ran end to end" \
            || warn "200, but no causal_steps/explain_result in the body. Inspect: ${PROBE_OUT}"
    else
        warn "Engine probe returned HTTP ${PROBE_CODE}. First 400 bytes:"
        head -c 400 "$PROBE_OUT" >&2; echo >&2
        confirm "Continue and deploy the proxy against it anyway?"
    fi
    rm -f "$PROBE_OUT"
fi

# =============================================================================
# Phase 4 — Build and push the proxy image
# =============================================================================
if [ "$SKIP_PROXY" = "0" ]; then
    step "[4/6] Building and pushing the proxy image"
    note "Dockerfile.proxy ships proxy/ and the built UI only — never src/ — so"
    note "this image carries none of torch/pyro. Three stages: pip deps, npm"
    note "build of ui/, then a slim runtime as a non-root user."
    run gcloud auth configure-docker gcr.io --quiet
    run docker build -f Dockerfile.proxy -t "$IMAGE_NAME" .
    run docker push "$IMAGE_NAME"
    ok "Pushed ${IMAGE_NAME}"
else
    step "[4/6] Proxy image — skipped (--skip-proxy)"
fi

# =============================================================================
# Phase 5 — Deploy the new Cloud Run service
#
# --set-env-vars, not --update-env-vars: this service is new, so the full
# environment is declared in one place and the resulting revision is exactly
# what this script specifies. --update-env-vars would leave the door open to
# inheriting something from a service that happened to already exist under
# this name.
# =============================================================================
if [ "$SKIP_PROXY" = "0" ]; then
    step "[5/6] Deploying Cloud Run service ${SERVICE_NAME}"

    if gcloud run services describe "$SERVICE_NAME" --region "$REGION" \
            --project "$PROJECT_ID" >/dev/null 2>&1; then
        warn "Service ${SERVICE_NAME} already exists — this will add a revision
      and shift 100% of traffic to it, not create a second service."
        confirm "Proceed?"
    fi

    ENV_PAIRS=(
        "AGENT_ENGINE_ENDPOINT=${AGENT_ENGINE_ENDPOINT}"
        "FIRESTORE_DATABASE_ID=${FIRESTORE_DATABASE_ID}"
        # Same-origin by default. A new stack is served from its own Cloud Run
        # URL, and listing production's domains here would let the live site
        # call this service cross-origin — exactly the traffic mixing a
        # separate stack exists to prevent.
        "CORS_ORIGINS=${CORS_ORIGINS:-}"
    )
    # Forwarded only when present, so a partial run never blanks a value the
    # service already carries.
    for VAR in ACCESS_SIGNING_SECRET ADMIN_TOKEN RESEND_API_KEY \
               SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD \
               ACCESS_NOTIFY_EMAIL ACCESS_FROM_EMAIL APP_URL \
               ACCESS_TOKEN_LIMIT ACCESS_TOKEN_GRANT \
               CAUSAL_MODEL CAUSAL_NUM_SAMPLES CAUSAL_SEED \
               CHAT_RETENTION_HOURS RUN_METRICS_RETENTION_DAYS; do
        [ -n "${!VAR:-}" ] && ENV_PAIRS+=("${VAR}=${!VAR}")
    done
    # '^##^' makes '##' the separator, so the commas inside CORS_ORIGINS are not
    # read by gcloud as the start of another assignment. Joined by hand rather
    # than with IFS: IFS is a set of single characters, so IFS='##' would join
    # on one '#' and every value after the first would be misparsed.
    ENV_ARG="^##^${ENV_PAIRS[0]}"
    for pair in "${ENV_PAIRS[@]:1}"; do ENV_ARG+="##${pair}"; done

    run gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --platform managed \
        --allow-unauthenticated \
        --service-account "$RUNTIME_SA" \
        --cpu "$CPU" --memory "$MEMORY" \
        --timeout "$TIMEOUT" \
        --concurrency "$CONCURRENCY" \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --set-env-vars "$ENV_ARG"

    if [ "$DRY_RUN" = "0" ]; then
        SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
            --region "$REGION" --project "$PROJECT_ID" --format='value(status.url)')"
        ok "Deployed ${SERVICE_URL}"
    else
        SERVICE_URL="https://${SERVICE_NAME}-<hash>.${REGION}.run.app"
    fi
else
    step "[5/6] Cloud Run — skipped (--skip-proxy)"
    SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" \
        --project "$PROJECT_ID" --format='value(status.url)' 2>/dev/null || echo '')"
fi

# =============================================================================
# Phase 6 — Verify, and optionally repoint hosting
# =============================================================================
step "[6/6] Verifying"

if [ "$RUN_SMOKE" = "1" ] && [ "$DRY_RUN" = "0" ] && [ -n "$SERVICE_URL" ]; then
    HEALTH="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 "${SERVICE_URL}/health" || echo 000)"
    [ "$HEALTH" = "200" ] && ok "GET /health → 200" || warn "GET /health → ${HEALTH}"

    # The proxy answers /health identically whether or not it has an engine, so
    # health alone cannot distinguish a working stack from one silently serving
    # proxy/mockdata.py. Read the env off the live revision instead.
    LIVE_ENDPOINT="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" \
        --project "$PROJECT_ID" \
        --format='value(spec.template.spec.containers[0].env.filter("name:AGENT_ENGINE_ENDPOINT").extract("value"))' 2>/dev/null || true)"
    if [[ "$LIVE_ENDPOINT" == *reasoningEngines* ]]; then
        ok "Live revision carries AGENT_ENGINE_ENDPOINT — not in mock mode"
    else
        warn "The live revision has no usable AGENT_ENGINE_ENDPOINT. This stack
      will answer every question from proxy/mockdata.py. Got: '${LIVE_ENDPOINT}'"
    fi
fi

if [ "$PROMOTE_HOSTING" = "1" ]; then
    step "Repointing Firebase Hosting at ${SERVICE_NAME}"
    # firebase.json currently rewrites to serviceId "tracerlensai-app", a name
    # left over from the rename that no longer matches anything deploy_to_gcp.sh
    # creates. Rewriting it is what makes the new stack the live site.
    run python - "$SERVICE_NAME" "$REGION" <<'PY'
import json, sys, pathlib
service, region = sys.argv[1], sys.argv[2]
path = pathlib.Path("firebase.json")
cfg = json.loads(path.read_text(encoding="utf-8"))
for rw in cfg["hosting"]["rewrites"]:
    if "run" in rw:
        print(f"  {rw['run']['serviceId']} -> {service}")
        rw["run"]["serviceId"] = service
        rw["run"]["region"] = region
path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY
    run bash -c 'cd ui && npm ci && npm run build'
    if command -v firebase >/dev/null 2>&1; then
        run firebase deploy --only hosting --project "$PROJECT_ID" --non-interactive
    else
        run npx -y firebase-tools deploy --only hosting --project "$PROJECT_ID" --non-interactive
    fi
    ok "Hosting now routes to ${SERVICE_NAME}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "${B}${GRN}✅ Stack '${STACK}' deployed${RST}"
echo
echo "  Cloud Run    : ${SERVICE_URL:-<unknown>}"
echo "  Engine       : ${AGENT_ENGINE_ENDPOINT}"
echo "  Metadata     : ${METADATA_FILE}"
echo "  Firestore    : ${FIRESTORE_DATABASE_ID}"
[ "$PROMOTE_HOSTING" = "1" ] \
    && echo "  Hosting      : repointed — https://causaltraceai.com now serves this stack" \
    || echo "  Hosting      : untouched — production still serves the old stack"
echo
echo "${B}Next${RST}"
echo "  1. Approve yourself:  open ${SERVICE_URL:-<url>}/admin  (password: \$ADMIN_TOKEN)"
echo "  2. Ask a real question and confirm the numbers move when you change the prompt."
echo "     Identical numbers across different questions means mock mode."
echo "  3. Logs:   gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"
echo
echo "${B}Rollback${RST}"
echo "  Hosting :  git checkout firebase.json && firebase deploy --only hosting --project ${PROJECT_ID}"
echo "  Proxy   :  gcloud run services delete ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"
# Not `gcloud ai reasoning-engines delete`: that command group needs the beta
# component installed, which the core CLI does not have. REST always works.
echo "  Engine  :  curl -sX DELETE -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\"
echo "               https://${REGION}-aiplatform.googleapis.com/${API_VERSION}/${ENGINE_ID:-<resource-name>}?force=true"
echo "  ${DIM}Production's engine and service were never modified by this script.${RST}"
