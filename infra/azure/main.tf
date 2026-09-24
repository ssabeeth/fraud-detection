# A slice of the system on Azure: the scoring API as a Container App that scales to zero.
# Nothing here costs money while idle: the Consumption plan bills per request-second
# above a monthly free grant, the environment has no fixed fee, no Log Analytics
# workspace is created (so no ingestion charges), and the image is pulled from a public
# GHCR repository (no registry to pay for). A budget alert watches the resource group.

resource "azurerm_resource_group" "this" {
  name     = "rg-${var.name}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_consumption_budget_resource_group" "this" {
  name              = "budget-${var.name}"
  resource_group_id = azurerm_resource_group.this.id
  amount            = var.budget_amount
  time_grain        = "Monthly"

  time_period {
    start_date = var.budget_start_date
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = 50
    threshold_type = "Actual"
    contact_emails = var.budget_emails
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Forecasted"
    contact_emails = var.budget_emails
  }
}

resource "azurerm_container_app_environment" "this" {
  name                = "cae-${var.name}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  tags                = local.tags
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${var.name}-api"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = azurerm_resource_group.this.name
  revision_mode                = "Single"
  tags                         = local.tags

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "api"
      image  = var.image
      cpu    = 0.25
      memory = "0.5Gi"

      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
    }

    http_scale_rule {
      name                = "http"
      concurrent_requests = "20"
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}

locals {
  tags = {
    project = "fraud-detection"
    purpose = "portfolio"
  }
}
