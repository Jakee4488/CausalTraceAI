# Bootstrap stub only. It provisions the service with a hello-world image and
# ignore_changes on that image; the real container is deployed imperatively by
# deploy_to_gcp.sh / deploy_new_stack.sh via `gcloud run deploy`. Terraform never
# owns what actually runs here.
#
# ⚠ APPLYING THE NAME CHANGE BELOW REPLACES THE SERVICE.
# `name` is not an updatable attribute — Terraform destroys and recreates, which
# deletes the live service and its URL. The name was corrected from
# "tracerlensai-app" (a leftover from the TracerLensAi rename) to the name both
# deploy scripts actually use, but do NOT apply it during normal operation. Run
# `terraform plan`, confirm the replacement is what you want, and do it at
# cutover only. Better still: drop this resource from Terraform entirely and let
# the deploy scripts own the service outright — a stub that fights the tool
# actually managing the resource earns nothing.
resource "google_cloud_run_service" "causaltraceai_app" {
  name     = "causaltraceai-app"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.app_sa.email
      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello" # Dummy image for initial provisioning
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GOOGLE_CLOUD_REGION"
          value = var.region
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
    ]
  }
}

# Keeps the state entry attached across the resource *address* rename, so that
# part is a no-op instead of a second destroy/create. The `name` attribute change
# above still forces a replacement — see the warning there.
moved {
  from = google_cloud_run_service.tracerlensai_app
  to   = google_cloud_run_service.causaltraceai_app
}

resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.causaltraceai_app.location
  project  = google_cloud_run_service.causaltraceai_app.project
  service  = google_cloud_run_service.causaltraceai_app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
