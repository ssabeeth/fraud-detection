variable "subscription_id" {
  description = "Azure subscription to deploy into (from `az account show --query id -o tsv`)."
  type        = string
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "uksouth"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "fraud"
}

variable "image" {
  description = "Scoring API image on GitHub Container Registry (public, so no pull secret)."
  type        = string
  default     = "ghcr.io/ssabeeth/fraud-api:latest"
}

variable "budget_amount" {
  description = "Monthly budget for the resource group, in the billing account's currency."
  type        = number
  default     = 5
}

variable "budget_emails" {
  description = "Who gets the budget alerts."
  type        = list(string)
}

variable "budget_start_date" {
  description = "First day of the budget period (the first of a month, RFC 3339)."
  type        = string
  default     = "2026-10-01T00:00:00Z"
}
