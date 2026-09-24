# Progress

**Status (2026-09-24): phases 1-8, 11 and 12 done; phases 9 and 10 written and checked
without credentials, waiting for the owner's accounts. Nothing has been pushed: the
GitHub repository does not exist yet.** Every result is from the real data and is in
`reports/`; the headline is in the README.

## Stopped for the owner: exact situation and next steps

1. **Create the GitHub repository and push.** Creating a public repository was blocked
   by this session's permission settings, so all work is committed locally on `main`
   with tags `v0.1-scaffold` to `v0.12-readme`. From this folder:

   ```bash
   gh repo create ssabeeth/fraud-detection --public --source . --remote origin
   git push origin main --tags
   ```

   Then check the CI run (the `image` job publishes `ghcr.io/ssabeeth/fraud-api` with a
   model trained on synthetic data) and make that package public in GitHub (Packages →
   fraud-api → settings), which the Azure step needs.
2. **Expire the Kaggle API token** that was pasted into the chat (kaggle.com → Settings
   → API). The download used `kaggle auth login` instead; the pasted token was never
   stored or used.
3. **Databricks (phase 9).** Create a Free Edition workspace, then
   `databricks auth login --host https://<workspace>.cloud.databricks.com`, then
   `make databricks-deploy databricks-upload databricks-run` (see `docs/databricks.md`).
4. **Azure (phase 10).** Create an account, `brew install azure-cli`, `az login`, copy
   `infra/azure/terraform.tfvars.example` to `terraform.tfvars` and fill it in, then apply
   the budget first and the rest second, exactly as in `docs/deploy_azure.md`. Destroy
   with `terraform destroy` when done.
5. **Tableau (phase 11).** Build and publish the dashboard from
   `exports/daily_policy_results.csv` following `docs/tableau.md`; add the link to the
   README.
6. When 3-5 are done, update this file and the README and tag `v1.0`.

**Local machine state.** Installed with Homebrew: `openjdk@17`, `terraform` 1.16.4,
the Databricks CLI (v1.17.0); Colima and Docker were already there (Colima is stopped
at the end of the session). `data/` holds the only copy of the data (the Delta lake,
745 MB; 1.1 GB with the models, the MLflow store and the monitoring HTML); the Kaggle CSVs were
deleted after bronze. Kaggle credentials are in `~/.kaggle/credentials.json` from
`kaggle auth login`.

## Phase status

| Phase | Status | Tag |
|---|---|---|
| 1. Scaffold | done | `v0.1-scaffold` |
| 2. Data and lakehouse | done | `v0.2-lakehouse` |
| 3. Features | done | `v0.3-features` |
| 4. Modelling | done | `v0.4-modelling` |
| 5. Decision policy and money | done | `v0.5-policy` |
| Fix: depth-limited LightGBM, phases 4-5 re-reported | done | `v0.5.1-depth-limit` |
| 6. Explainability and governance | done | `v0.6-explainability` |
| 7. Streaming | done | `v0.7-streaming` |
| 8. Serving and monitoring | done | `v0.8` |
| 9. Databricks | ready; needs the owner's workspace login to deploy | `v0.9-databricks-ready` |
| 10. Cloud slice with Terraform | ready; needs the owner's Azure login to apply | `v0.10-azure-ready` |
| 11. Business dashboard | export and guide done; the owner builds and publishes | `v0.11-dashboard` |
| 12. README | done (`v1.0` waits for 9-11's owner steps) | `v0.12-readme` |

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

### Phase 2 — Data and lakehouse (2026-09-24)

Done:
- Kaggle access: the owner signed in with `kaggle auth login` (browser OAuth; the CLI
  stores its own credentials in `~/.kaggle`). Competition rules accepted by the owner.
- `fraud download` fetched `train_transaction.csv` (683 MB) and `train_identity.csv`
  (27 MB); `fraud bronze` converted them to Delta and deleted them. The lake is the only
  copy: bronze 91 MB, silver 95 MB, gold 102 MB.
- Silver: typed with `try_cast`, deduplicated, joined, anchored to the calendar, with
  data-quality checks that fail the job. All pass on the real data (590,540 rows).
- Gold: split column from config and the entity keys (pseudo-card, device, email).
- `reports/data.md` (generated): split sizes and fraud rates, months, the time-anchor
  check and the pseudo-card key stability. Anchor supported by a 20-24 December peak and
  by the D1 day boundary matching the anchor's midnight (offset search, below).
- Synthetic fixtures (`scripts/make_fixtures.py`, 4,009 rows, IDs from 1) with same-second
  ties, shared card1 values and D1 off-by-one noise; tests for bronze, silver, gold, the
  failing checks and key parity.

Numbers: train 417,559 rows (3.53% fraud, $2.16M fraud), valid 83,655 (3.38%,
$450k), test 89,326 (3.49%, $477k). 24.4% of transactions have an identity record.

Day-boundary offset search (train months, keys with addr1 and D1): distinct card keys
by hour offset h of the day boundary were 152,471 at h=0 and rose monotonically to
175,614 at h=17 before falling back to 160,649 at h=23; one-day-apart pairs likewise
had their minimum (28,742) at h=0. The anchor's midnight is the boundary D1 used.

Tools installed: `openjdk@17` (Homebrew). Spark 4.2.0 with Delta 4.4.0 locally.

### Phase 3 — Features (2026-09-24)

Done:
- One definition of every feature (`features/definitions.py`): 17 point-in-time history
  aggregates over the card, device and email keys, and 19 readable transaction fields.
- Three implementations: Spark window functions (`offline.py`), per-key streaming state
  (`online.py`), and a naive pandas recomputation from raw history (`reference.py`).
- Tests: the point-in-time test on every fixture row, a canary (a window that includes
  the current row) that the test must catch, online/offline parity on every fixture row,
  and unit tests for ties, window edges, new-value flags and out-of-order events.
- Real data: `gold/features` (590,540 rows, 225 MB). Point-in-time check: 76,415 values,
  0 differences. Parity: 14.2 million values, 0 mismatches; the online code computes
  about 112,000 events per second in one process.
- `docs/features.md`, generated from the definitions.

Fixed on the way: the first feature build ran for minutes on one core because rows with
no device key shared one window partition, and the sliding count over `gmail.com`
(228,355 rows) was quadratic. See DECISIONS.md.

### Phase 4 — Modelling (2026-09-24)

Done:
- Rules baseline (points, 432-combination grid on round thresholds), logistic
  regression (8 settings) and LightGBM (8 settings, early stopping), all tuned on April
  only, in that order. Weighting, not oversampling. Calibration chosen by a time split
  inside April (none for logistic regression, Platt for LightGBM).
- MLflow: local sqlite store under `data/mlflow`, experiment `fraud-models`, registered
  models `fraud-logreg` and `fraud-lightgbm`.
- Bundles (`data/models/<name>/`) carry preprocessing, model and calibration; a test
  checks that a saved and reloaded bundle scores identically, one row or many.
- `reports/model.md` (generated): test PR-AUC 0.561 for LightGBM against 0.113 for
  logistic regression and 0.047 for the rules; recall 47.6% at a 1% false-positive rate;
  ECE 0.004. Every read of the test month is logged (3 so far).

Kept failure: logistic regression collapses on May because one legitimate pseudo-card
with 1,393 transactions fills its alert list (see DECISIONS.md).

### Phase 5 — Decision policy and money (2026-09-24)

Done:
- Cost model in `configs/costs.yaml` (seven assumptions, each with a rationale and a
  range), with hand-checked tests.
- Policies decided in arrival order within 50 reviews a day: expected loss (one tuned
  threshold), probability cut-offs, rules levels, approve-all. All tuned on April.
- The chosen policy frozen to `reports/policy_frozen.json` before the test month was read;
  `reports/policy.md` (generated) with the test month, validation, the ranking
  comparison and a 14-row sensitivity table; `exports/daily_policy_results.csv` for
  the dashboard (daily aggregates only).

Headline: on May 2018 the chosen policy catches 60.4% of fraud value at $270,554,
against $474,219 for the rules baseline (16.1% caught).

Fixed on the way: the ranking comparison measured the saving against the wrong
alternative (see DECISIONS.md); fixed on validation only, no extra test read.

### Fix — depth-limited LightGBM (2026-09-24)

Building phase 7 showed the phase 4 LightGBM needed 162 ms per decision for its SHAP
reasons (trees up to depth 57). Trees are now capped at depth 8 and LightGBM re-tuned on
April: 5.4 ms per decision with reasons (p99 5.7 ms). Test PR-AUC 0.547 (was 0.561); the
policy costs $278,535 on May against $474,219 for the rules (was $270,554). Phases 4 and
5 reports regenerated; the test month was read twice more for that, logged. Details in
DECISIONS.md.

### Phase 6 — Explainability and governance (2026-09-24)

Done:
- Reason codes: top three positive TreeSHAP contributions per decision, with plain
  names (reviewer names for engineered and readable fields, Vesta's family for masked
  columns).
- The explainable-only experiment: $92,513 (33%) more on May than all features.
- Segment checks by product, card network, card type, device type and email domain,
  with the statement that this is not a fairness audit and what one would need.
- `docs/model_card.md` and `docs/data_card.md`, generated from the report JSON
  (`fraud cards`).
- Found and fixed on the way: reason values that were Python integers were shown as
  strings.

### Phase 7 — Streaming (2026-09-24)

Done:
- Replay producer (event time × speed-up, labels 30 days late on their own topic),
  stream processor (per-key state warmed from the lake, features → score → TreeSHAP
  reasons → frozen policy with the daily review capacity), Spark Structured Streaming
  sink from Kafka to Delta bronze.
- The parity test through the stream on the real test month: 89,326 transactions ×
  20 values read back from Delta bronze, **0 mismatches**; every action equal to the
  frozen policy applied offline; largest score difference 1.2e-14.
- Latency and throughput (`reports/stream.md`): 5.4 ms per decision at the median as
  fast as possible (183 decisions a second, one process); paced at 1,800× real time,
  15.5 ms end to end at the median and 56 ms at p99.
- Tests: the scorer against offline features, scores and policy; message encoding; the
  label timeline; the whole stream on the fixtures with Redpanda (CI job `stream`),
  including a second replay that must land the same rows.

Fixed on the way: the first LightGBM needed 162 ms per decision for its SHAP reasons,
so trees are now capped at depth 8 (see the fix entry above); the sink checkpoint would
have skipped a second replay's messages; the producer held a month of 430-field dicts in
memory at once.

### Phase 8 — Serving and monitoring (2026-09-24)

Done:
- FastAPI service (`fraud.serve.app`): `/score` computes the point-in-time features
  from the history supplied with the request (same online code as the processor),
  scores, returns the top three reasons and the recommended action; `/health`;
  `/metrics` (Prometheus: requests, actions, latency, score histogram). History after the
  transaction is rejected with 422.
- Docker image (`docker/Dockerfile`, `make image`): the model and frozen policy baked in;
  base dependencies slimmed so the Python environment is 416 MB. Smoke-tested locally
  with the real model (not published); CI builds and publishes it with a fixture-trained
  model only.
- Monitoring (`fraud monitor`, `reports/monitoring.md`): daily Evidently drift on inputs
  and PSI on scores over 7 days, a daily feed-health check, and per-week performance
  from labels as they arrive 30 days late; thresholds and the retrain rule in
  `configs/monitoring.yaml`. Demonstrated on the streamed replay and on a stress
  scenario (identity feed silent from 22 May).

Results: the drift monitor warned on the velocity features for 16 of 25 days (the busy
pseudo-card), scores stayed stable (PSI at most 0.011), no retrain triggered, and the
June cohorts confirmed performance held. The feed-health check caught the stress
scenario's silent identity feed on its first full day. Fixed on the way: the first
version only noticed the silent feed after ten days (see DECISIONS.md).

### Phase 9 — Databricks (2026-09-24): ready, not deployed

Done: Asset Bundle with schema, volumes and a job running phases 2-5 from the wheel on
serverless; `make databricks-deploy`, `make databricks-upload`, `make databricks-run`;
`docs/databricks.md`; a test that every task is a valid CLI command in pipeline order.
The wheel was built and run outside the repository (packaged configs, `--data-dir`).
Blocked on: the owner's Free Edition workspace and `databricks auth login` (none on this
machine). The Databricks CLI (v1.17.0) is installed.

### Phase 10 — Cloud slice with Terraform (2026-09-24): ready, not applied

Done: `infra/azure/` (resource group, budget alert, Container Apps environment, a
scale-to-zero Container App running `ghcr.io/ssabeeth/fraud-api`), `terraform.tfvars.example`,
`docs/deploy_azure.md` (budget-first apply, test, destroy), CI job `terraform` (fmt,
validate, tflint; no credentials), `make tf-check`. Terraform 1.16.4 installed via
Homebrew; tflint runs from its official Docker image locally (no Homebrew formula).
Blocked on: the owner's Azure account and `az login` (Azure CLI not installed), and the
GitHub repository existing so CI can publish the image.

### Phase 11 — Business dashboard (2026-09-24)

Done: `exports/daily_policy_results.csv` (daily aggregates for every policy on the test
month; a test ties its totals to `reports/policy.md`) and `docs/tableau.md` (step-by-step
build guide). Owner's step: build and publish in Tableau Public, then add the link to the
README.

### Phase 12 — README (2026-09-24)

Done: README with the headline, results tables, the correctness checks, architecture
(Mermaid), how it works, quickstart, decisions, failures kept, known limitations, what
production would add, the Kaggle-leaderboard caveat and the data licence. Every number
checked against the generated reports; one claim corrected on the way (four of the
*five* busiest days fall 20-24 December; 2 March is the second busiest).
