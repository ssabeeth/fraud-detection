output "api_url" {
  description = "Public URL of the scoring API (the first request after idle wakes it up)."
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "resource_group" {
  value = azurerm_resource_group.this.name
}
