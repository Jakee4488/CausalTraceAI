variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "icarus-agent-26"
}

variable "region" {
  description = "The default GCP region"
  type        = string
  default     = "europe-west2"
}

variable "causal_artifact_repo_name" {
  description = "Artifact Registry name for Causal MLOps"
  type        = string
  default     = "causal-mlops-repo"
}

variable "causal_artifacts_bucket" {
  description = "GCS bucket name for Causal MLOps artifacts"
  type        = string
  default     = "tracerlens-causal-artifacts"
}

variable "github_repo" {
  description = "GitHub repository for Workload Identity Federation (format: owner/repo)"
  type        = string
  default     = "Jakee4488/TracerLensAi"
}
