resource "google_artifact_registry_repository" "agent_repo" {
  location      = var.region
  repository_id = "agent-docker-repo"
  description   = "Docker repository for Enterprise Agent containers"
  format        = "DOCKER"
}

resource "google_storage_bucket" "agent_cache" {
  name          = "${var.project_id}-agent-cache"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}

resource "google_bigquery_dataset" "agent_logs" {
  dataset_id                  = "agent_orchestrator_logs"
  friendly_name               = "Agent Orchestrator Logs"
  description                 = "Stores evaluation metrics and interaction logs from the Agent Orchestrator"
  location                    = var.region
  default_table_expiration_ms = 31536000000 # 365 days
}
