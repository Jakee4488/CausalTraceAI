# Service Account for the GKE Application pods
resource "google_service_account" "app_sa" {
  account_id   = "agent-app-sa"
  display_name = "Agent Orchestrator Application SA"
}

# Roles required by the application
resource "google_project_iam_member" "app_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

resource "google_project_iam_member" "app_bigquery_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

resource "google_project_iam_member" "app_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

# Firestore. This SA is the Cloud Run runtime identity, and proxy/access.py
# reads an access record on EVERY request — the gate runs before anything else
# does. Without this role the service starts clean and 403s the first time
# anyone signs in, which reads like an application bug rather than a missing
# grant. deploy_to_gcp.sh already assumed this binding existed.
#
# Project-scoped, so it covers every named database (tracerlensai, and the
# per-stack databases deploy_new_stack.sh expects). Per-tier isolation comes
# from FIRESTORE_DATABASE_ID plus a separate Cloud Run service, not from a
# separate identity.
resource "google_project_iam_member" "app_datastore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

# Workload Identity binding for GKE Kubernetes Service Account to GCP Service Account
resource "google_service_account_iam_member" "workload_identity_binding" {
  service_account_id = google_service_account.app_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/agent-ksa]"
}

# (Optional) Workload Identity Federation for GitHub Actions
resource "google_iam_workload_identity_pool" "github_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions-pool-v3"
  display_name              = "GitHub Actions Pool"
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions-provider-v3"
  display_name                       = "GitHub Actions Provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Service Account for CI/CD
resource "google_service_account" "cicd_sa" {
  account_id   = "github-actions-sa"
  display_name = "GitHub Actions CI/CD SA"
}

resource "google_project_iam_member" "cicd_owner" {
  project = var.project_id
  role    = "roles/owner" # WARNING: High privilege. Should scope down in production.
  member  = "serviceAccount:${google_service_account.cicd_sa.email}"
}

resource "google_service_account_iam_member" "github_sa_binding" {
  service_account_id = google_service_account.cicd_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}

data "google_project" "project" {}

resource "google_project_iam_member" "gke_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}
