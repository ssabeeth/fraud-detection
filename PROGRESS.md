# Progress

**Status: phase 1 (scaffold) in progress.**

## Phase status

| Phase | Status | Tag |
|---|---|---|
| 1. Scaffold | in progress | |
| 2. Data and lakehouse | not started | |
| 3. Features | not started | |
| 4. Modelling | not started | |
| 5. Decision policy and money | not started | |
| 6. Explainability and governance | not started | |
| 7. Streaming | not started | |
| 8. Serving and monitoring | not started | |
| 9. Databricks | not started | |
| 10. Cloud slice with Terraform | not started | |
| 11. Business dashboard | not started | |
| 12. README | not started | |

## Log

### Phase 1 — Scaffold (2026-09-24)

Done:
- `uv` project (Python 3.12), `ruff`, `pre-commit`, `pytest`, a Makefile and a CI
  workflow (lint, tests, Compose validation).
- `configs/base.yaml` holds the time anchor, the split boundaries and the label delay.
  `fraud.config` validates them (contiguous, ordered, non-empty) and converts dates to
  the `TransactionDT` scale.
- `docker-compose.yml` runs Redpanda.
- A guard against committing competition rows (`scripts/check_no_raw_data.py`), run by
  pre-commit and CI.
- Read the competition rules; what they allow is in DECISIONS.md.

Tools installed on this machine: `openjdk@17` (Homebrew, for PySpark).
