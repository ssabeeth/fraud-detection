# Real-Time Card Fraud Detection — Build Instructions

You are building a portfolio project end to end, working autonomously. The owner is not
available to answer questions during the build. Make reasonable decisions, record them,
and keep moving. Only stop for the items listed under "Stop and hand back".

## Goal

A production-style fraud system that:
1. Scores card transactions as they stream in, in milliseconds, with a reason for every
   decision.
2. Chooses what to block or send for review by **expected money lost**, not by a
   probability cut-off, and reports every policy in money.
3. Is trained and evaluated only on the past, with features that could have been computed
   at the moment of the transaction.
4. Runs on a lakehouse, is monitored for drift under realistic label delay, and deploys a
   slice to the cloud with Terraform.

The audience is hiring managers for data scientist, ML engineer, data engineer and
analytics engineer roles in UK finance. Correctness and clarity beat feature count. This
project sits beside the owner's other repos (`electricity`, `Groundwork`, `geosight`) and
should read like them: numbers produced by code, decisions written down, failures kept.

**The headline claim to earn:** "On the most recent month, the chosen review policy
catches X% of fraud value at a cost of $Y, against $Z for a rules baseline." Every word of
that must come from a run.

## Stack

- Python 3.12, managed with `uv`; `ruff`, `pytest`, `pre-commit`
- PySpark with Delta Lake (bronze, silver, gold), run locally; the same jobs on Databricks
  Free Edition in phase 9
- Kafka API via Redpanda in Docker, for the transaction stream
- LightGBM for the main model; logistic regression and a hand-written rules baseline first
- MLflow for tracking and the registry
- SHAP for per-decision explanations
- FastAPI for synchronous scoring; a Python stream processor for the online path
- Evidently for drift reports
- Terraform for an Azure slice (phase 10)
- Tableau Public for the business dashboard (phase 11; the owner publishes)
- GitHub Actions: lint, tests, Spark jobs on fixture data, `terraform validate`, image build

No Airflow here (the `electricity` repo already shows it). A Makefile and GitHub Actions
are enough; on Databricks, use Jobs defined in an Asset Bundle.

## Data

**IEEE-CIS Fraud Detection** (Vesta), from Kaggle: `train_transaction.csv` and
`train_identity.csv`, joined on `TransactionID`. About 590,000 transactions over roughly six
months, 3.5% fraud. The Kaggle test files have no labels; do not use them.

- **Licence.** Downloading needs the owner's Kaggle account and acceptance of the
  competition rules. Read the rules on the competition page before building, and record
  in `DECISIONS.md` what they allow. Assume redistribution is not allowed: **never commit
  raw rows**, not even samples. Only aggregates (metrics, daily totals) leave the machine.
- **CI fixtures** come from a small synthetic generator that mimics the schema
  (`scripts/make_fixtures.py`), committed under `tests/fixtures/`.
- **Time.** `TransactionDT` is seconds from an undisclosed reference point. Choose an
  anchor date, document it, and treat all time as relative to it.
- **Currency.** `TransactionAmt` is in US dollars. Report money in USD and say so. Do not
  relabel it as pounds.
- **Anonymised features.** Most columns (`V1`–`V339`, `C*`, `D*`, `M*`) are masked. Phase
  6 measures what it costs to leave them out in favour of features a reviewer can explain.
- **Card identity.** There is no card ID. Build a pseudo-card key from card and address
  fields (the usual `card1` + `addr1` + a derived first-seen day), document how it was
  derived, and measure how stable it is.
- **Kaggle leaderboard.** Do not compare results to it as if they were like-for-like. Top
  solutions exploited the competition's split and client identification. This project's
  time split and point-in-time features are stricter, and the README says so.

## Non-negotiable correctness rules

1. **Time-based splits only.** Months 1–4 train, month 5 validation (all tuning, the
   threshold and the policy), month 6 test, touched once per reported result. No random
   splits, no shuffling across time.
2. **Point-in-time features.** Every aggregate for a transaction (velocity, amount
   z-scores, time since the card's previous transaction, device and email counts) uses
   only transactions strictly before it. Enforce this with a test that recomputes features
   for sampled rows from raw history and fails on any difference.
3. **One feature definition, two implementations, tested equal.** The Spark (offline) and
   stream-processor (online) feature code must agree. A parity test replays a period
   through the stream and compares every feature with the offline value. Training/serving
   skew is the bug this project exists to rule out.
4. **Label delay.** Fraud labels (chargebacks) arrive late. The stream simulation releases
   each label a configurable delay after its transaction (default 30 days, documented).
   Monitoring and any retraining may only use labels that would have arrived by then.
5. **The baseline comes first.** A rules baseline (amount and velocity thresholds, tuned on
   validation) and logistic regression are measured before LightGBM, on the same split,
   and everything is reported against them.
6. **Metrics that fit imbalance.** Report PR-AUC, recall at fixed false-positive rates, and
   money at the chosen policy. ROC-AUC alone is not a result. Check calibration.
7. **Decisions in money.** The cost model is a config file:
   - a missed fraud loses the transaction amount plus a chargeback fee;
   - a review costs analyst time, and a declined or delayed legitimate customer costs a
     friction amount.
   Every value is an assumption with a written source or rationale. Report results under
   a sensitivity range, not just the defaults.
8. **Rank by expected loss.** With a daily review capacity (a config value), send the
   transactions with the highest `P(fraud) × amount − review cost` to review, not the
   highest probabilities. Compare the two rankings in money on the validation month.
9. **Every decision is explained.** The scoring path returns the top reasons (SHAP) in
   plain names alongside the score.

## Build phases

Complete each phase fully, commit, then move on. Develop on a small time slice, then run
the full data once the pipeline works.

1. **Scaffold.**
   - `uv` project, `pre-commit`, `ruff`, CI skeleton, `.env.example`, README stub,
     `DECISIONS.md`, `PROGRESS.md`, `docker-compose.yml` with Redpanda.
   - `.gitignore` covers `.env`, `data/`, `mlruns/`, model artefacts and Kaggle files
     from the first commit.
2. **Data and lakehouse.**
   - Download via the Kaggle API; convert once to Delta bronze and delete the CSVs.
   - Silver: typed, joined, deduplicated, with the anchor date applied and data-quality
     checks (keys, nulls, ranges) that fail the job.
   - Gold: the modelling table. Split boundaries live in config.
3. **Features.**
   - Point-in-time aggregates over the pseudo-card, device and email keys, as Spark
     window functions ordered by time with an exclusive upper bound.
   - The point-in-time test (rule 2). A feature catalogue in `docs/features.md`: name,
     definition, window, and what a reviewer would call it.
4. **Modelling.**
   - Rules baseline, logistic regression, LightGBM. Imbalance handled by weighting;
     record why oversampling was or was not used.
   - Tune on validation only. Log everything to MLflow. Calibrate if needed.
   - Commit `reports/model.md` with metrics and plots.
5. **Decision policy and money.**
   - The cost model, the capacity-constrained review policy, and the expected-loss
     ranking against probability ranking (rule 8).
   - Choose the policy on validation, freeze it, and report it once on test against the
     rules baseline, with a sensitivity table.
   - Commit `reports/policy.md`.
6. **Explainability and governance.**
   - SHAP reason codes per decision, with readable names.
   - An experiment: the same model on explainable features only, against all features,
     with the difference in money.
   - Segment checks: alert rate, precision and recall by product, card network, card
     type, device type and email domain. State plainly that the data has no protected
     characteristics, so this is not a fairness audit, and say what one would need.
   - `docs/model_card.md` and `docs/data_card.md`.
7. **Streaming.**
   - A replay producer publishes month 6 to Redpanda in time order, with a speed-up
     factor.
   - A stream processor keeps per-key state, computes online features, scores, and
     publishes decisions with reasons. Labels arrive on their own topic after the delay.
   - Kafka → Spark Structured Streaming → Delta bronze, so the lakehouse fills from the
     stream as well.
   - The parity test (rule 3). Measure scoring latency (p50 and p99) and throughput.
8. **Serving and monitoring.**
   - FastAPI: `/score` (features computed from supplied history, reasons returned),
     `/health`, `/metrics`.
   - Evidently drift reports on input features and scores over sliding windows, and
     performance once labels arrive. Define alert thresholds and what would trigger a
     retrain, and demonstrate them on the replay.
   - Tag `v0.8`. Everything so far runs locally with no accounts beyond Kaggle.
9. **Databricks.**
   - A Databricks Asset Bundle that runs phases 2–5 as Jobs on Databricks Free Edition,
     with the data uploaded to a Unity Catalog volume (Free Edition blocks most outbound
     internet, so do not download from Kaggle inside it).
   - Needs the owner's workspace and CLI login. If they are not set up, stop and hand back.
10. **Cloud slice with Terraform.**
    - Azure: resource group, Container Apps environment and a scale-to-zero Container App
      running the scoring API from an image on GitHub Container Registry, and a budget
      alert. Nothing that costs money while idle.
    - CI runs `terraform fmt -check`, `validate` and `tflint` with no credentials.
      `apply` only with the owner's `az login`; document `destroy`.
    - If the owner has no Azure account yet, stop and hand back.
11. **Business dashboard.**
    - Export aggregated daily results (transactions, fraud caught, reviews, false
      alarms, money under each policy) as a small CSV. Nothing row-level.
    - Write `docs/tableau.md`: a step-by-step build guide for Tableau Public (sheets,
      calculated fields, layout). The owner builds and publishes it.
12. **README.**
    - Architecture diagram in Mermaid, quickstart, results, a "Decisions" section, known
      limitations, and "What production would add" (feature store, case management,
      model risk review).

## Working rules

- **Decide, don't ask.** When something is ambiguous, pick the sensible option. Log it in
  `DECISIONS.md` with a date, the options considered and the reason.
- **Keep a log.** Maintain `PROGRESS.md` with what is done, what is next and any
  blockers. Update it at the end of every phase.
- **Commit often.** Small commits with conventional messages. Never commit secrets, API
  keys, Kaggle files or raw data.
- **Test as you go.** Every phase needs passing tests before it is marked done: the
  point-in-time test, the parity test, the cost model and the policy each get their own.
- **Keep scope tight.** No Kubernetes, no agents, no LLMs. Depth beats breadth: if a phase
  threatens the quality of an earlier one, cut the later one and say so.
- **Disk is limited.** The machine has about 25 GB free. Keep one copy of the data in
  Delta, delete intermediate CSVs, and prune Docker images you create.
- **Tools.** Docker and Colima are installed. Install what else is needed with Homebrew or
  `uv` (Java 17 for PySpark, Terraform, the Databricks and Azure CLIs) and record it in
  `PROGRESS.md`.

## Git and GitHub

- **Remote.** Run `git remote -v` first. If there is no remote, create a public repository
  `ssabeeth/fraud-detection` with `gh` (it is authenticated) and push.
- **Branches.** One branch per phase, e.g. `phase-3-features`; push regularly.
- **Merging.** When a phase is complete and green: merge into `main` with a merge commit,
  push, and tag it (`v0.3-features`, and so on; `v1.0` after phase 12).
- **Keep `main` green.** Never push a failing build to `main`. Never force-push or rewrite
  pushed history.
- **Before every push**, check the diff for secrets, data rows and files over 5 MB.

## What the owner provides, and when

| When | What | Notes |
|---|---|---|
| Before phase 2 | Kaggle API token at `~/.kaggle/kaggle.json`, and the competition rules accepted | Never ask for the token in chat |
| Phase 9 | Databricks Free Edition workspace, logged in with `databricks auth login` | Free, no card |
| Phase 10 | Azure account, logged in with `az login` | Card needed for identity; budget alert first |
| Phase 11 | Tableau Public account and desktop app | The owner builds and publishes |

Credentials live in the owner's own login sessions or files outside the repository. Never
ask for a secret in chat, never print one, never write one to a file in the repository.

## Stop and hand back only for

- A missing account or login from the table above.
- The data turning out not to be usable as planned (licence or content), where the
  fallback would change the headline claim.
- Anything that costs money.

When stopping, write the exact situation and the recommended next step at the top of
`PROGRESS.md`.
