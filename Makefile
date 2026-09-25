.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the virtualenv with every extra and install pre-commit hooks
	$(UV) sync --all-extras
	$(UV) run pre-commit install

.PHONY: lint
lint: ## Ruff lint and format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: fmt
fmt: ## Auto-fix lint and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: test
test: ## Unit tests (no broker needed)
	$(UV) run pytest -m "not kafka"

.PHONY: check-data
check-data: ## Fail if any tracked file looks like competition rows
	git ls-files -z | xargs -0 $(UV) run python scripts/check_no_raw_data.py

.PHONY: data
data: ## Download from Kaggle and build bronze (deletes the CSVs), silver, gold, profile
	$(UV) run fraud download
	$(UV) run fraud bronze
	$(MAKE) lakehouse

.PHONY: lakehouse
lakehouse: ## Rebuild silver, gold and the data profile from bronze
	$(UV) run fraud silver
	$(UV) run fraud gold
	$(UV) run fraud profile

.PHONY: features
features: ## Point-in-time features, the point-in-time check and online/offline parity
	$(UV) run fraud features
	$(UV) run fraud check-pit --sample 3000
	$(UV) run fraud check-parity

.PHONY: train
train: ## Fit the rules baseline, logistic regression and LightGBM (tuned on validation)
	$(UV) run fraud train

.PHONY: evaluate
evaluate: ## Score the frozen models on validation and, once, on test; reports/model.md
	$(UV) run fraud evaluate

.PHONY: policy
policy: ## Choose the policy on validation, freeze it, report it on test; reports/policy.md
	$(UV) run fraud policy

.PHONY: compare
compare: ## LightGBM vs XGBoost vs CatBoost on Feb, Mar and Apr (never reads May); reports/model_comparison.md
	$(UV) run fraud compare-models

.PHONY: dashboard
dashboard: ## Rebuild site/index.html (GitHub Pages) from the daily export and the policy report
	$(UV) run fraud dashboard

.PHONY: explain
explain: ## Explainable-only model, segment checks, SHAP; then the model and data cards
	$(UV) run fraud explain
	$(UV) run fraud cards

.PHONY: stream
stream: up ## Replay the test month through Redpanda, the processor and the Spark sink
	$(UV) run fraud stream

.PHONY: monitor
monitor: ## Drift and delayed-label performance on the replay; reports/monitoring.md
	$(UV) run fraud monitor

.PHONY: serve
serve: ## Run the scoring API on localhost:8000
	$(UV) run uvicorn fraud.serve.app:app --port 8000

.PHONY: fixtures
fixtures: ## Regenerate the synthetic CI fixtures
	$(UV) run python scripts/make_fixtures.py

.PHONY: up
up: ## Start Redpanda (Kafka API on localhost:19092)
	docker compose up -d --wait redpanda

.PHONY: down
down: ## Stop Redpanda and remove its volume
	docker compose down -v

.PHONY: databricks-deploy
databricks-deploy: ## Validate and deploy the Databricks Asset Bundle (needs databricks auth login)
	databricks bundle validate -t free
	databricks bundle deploy -t free

.PHONY: databricks-upload
databricks-upload: ## Download the CSVs from Kaggle, upload them to the raw volume, delete local copies
	$(UV) run fraud download
	databricks fs cp data/raw/train_transaction.csv dbfs:/Volumes/workspace/fraud/raw/ --overwrite
	databricks fs cp data/raw/train_identity.csv dbfs:/Volumes/workspace/fraud/raw/ --overwrite
	rm -f data/raw/train_transaction.csv data/raw/train_identity.csv

.PHONY: databricks-run
databricks-run: ## Run phases 2-5 as a Databricks Job and wait for it
	databricks bundle run fraud_pipeline -t free

.PHONY: tf-check
tf-check: ## terraform fmt, validate and tflint for infra/azure (no credentials needed)
	cd infra/azure && terraform fmt -check -recursive && terraform init -backend=false -input=false >/dev/null && terraform validate
	docker run --rm -v "$(CURDIR)/infra/azure":/data -w /data --entrypoint sh ghcr.io/terraform-linters/tflint:v0.64.0 -c "tflint --init >/dev/null && tflint"

.PHONY: image
image: ## Build the scoring API image with the local LightGBM bundle
	scripts/build_image.sh data/models/lightgbm fraud-api:local
