terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.40"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  # A new subscription has Container Apps (Microsoft.App) switched off, and the
  # provider's default "core" set does not include it. Registering it is free.
  resource_providers_to_register = ["Microsoft.App"]
}
