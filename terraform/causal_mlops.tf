# Enable Required APIs
resource "google_project_service" "artifactregistry_api" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudfunctions_api" {
  project = var.project_id
  service = "cloudfunctions.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudrun_api" {
  project = var.project_id
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "aiplatform_api" {
  project = var.project_id
  service = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry
resource "google_artifact_registry_repository" "causal_repo" {
  location      = var.region
  repository_id = var.causal_artifact_repo_name
  description   = "Docker repository for Causal MLOps components"
  format        = "DOCKER"
  depends_on    = [google_project_service.artifactregistry_api]
}



# GCS Bucket for Artifacts
resource "google_storage_bucket" "causal_artifacts" {
  name                        = "${var.causal_artifacts_bucket}-${var.project_id}"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

