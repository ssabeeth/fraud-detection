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

.PHONY: fixtures
fixtures: ## Regenerate the synthetic CI fixtures
	$(UV) run python scripts/make_fixtures.py

.PHONY: up
up: ## Start Redpanda (Kafka API on localhost:19092)
	docker compose up -d --wait redpanda

.PHONY: down
down: ## Stop Redpanda and remove its volume
	docker compose down -v
