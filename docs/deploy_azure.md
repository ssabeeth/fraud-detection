# Deploying the scoring API to Azure

A slice of the system on Azure, defined in Terraform (`infra/azure/`): a resource group,
a Container Apps environment, the scoring API as a Container App that **scales to zero**,
and a **budget alert**. Nothing costs money while idle: Container Apps on the Consumption
plan bills per vCPU-second and GiB-second actually used, above a monthly free grant; the
environment has no fixed fee; no Log Analytics workspace is created; the image comes from
a public GitHub Container Registry package.

## What the public image contains

CI builds `ghcr.io/ssabeeth/fraud-api` with a model trained on the **synthetic CI
fixtures**, not on the competition data, because the competition's rules forbid sharing
the data and a public image is a form of sharing. The frozen policy (a threshold and the
cost assumptions) is the real one from `reports/policy_frozen.json`. So the deployed API
shows the real code path (features from supplied history, score, SHAP reasons, action),
while its scores come from a demonstration model. To serve the real model, build the image
locally with `scripts/build_image.sh data/models/lightgbm` and push it to a *private*
registry; the Container App then needs a registry credential.

## Prerequisites (owner)

- An Azure account (a card is needed to verify identity; nothing here is billed while idle).
- `brew install azure-cli terraform`
- `az login`, then note the subscription id: `az account show --query id -o tsv`.
- In GitHub, make the `fraud-api` package public (Packages → fraud-api → Package settings →
  Change visibility), so Azure can pull it without a secret.

## Apply: budget first

```bash
cd infra/azure
cp terraform.tfvars.example terraform.tfvars      # fill in subscription_id and budget_emails
terraform init
# 1. the resource group and its budget alert, before anything that could cost money
terraform apply -target=azurerm_resource_group.this -target=azurerm_consumption_budget_resource_group.this
# 2. everything else
terraform apply
curl "$(terraform output -raw api_url)/health"
```

The first request after a quiet period takes a few seconds while a replica starts.

## Destroy

```bash
cd infra/azure
terraform destroy
az group show --name rg-fraud 2>/dev/null || echo "resource group gone"
```

`terraform.tfvars` and the state files are git-ignored; keep the state until `destroy`
has run.

## CI

Every push runs `terraform fmt -check`, `terraform init -backend=false`,
`terraform validate` and `tflint` (with the azurerm ruleset). None of these needs Azure
credentials; `apply` only ever runs from the owner's machine.
