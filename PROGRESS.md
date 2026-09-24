# Progress

**Status: phase 5 (decision policy and money) done; phase 6 (explainability) next.**

The GitHub remote does not exist yet: creating the public repository was blocked by
the session's permission settings, so all work is committed locally. See "What only
the owner can do".

## What only the owner can do

1. **Create the GitHub repository and push.** Run `gh repo create ssabeeth/fraud-detection --public --source . --push`
   from this folder (or allow Claude to), then `git push origin --all && git push origin --tags`.
2. **Expire the Kaggle API token** that was pasted into the chat on 2026-09-24
   (kaggle.com → Settings → API). The download used `kaggle auth login` instead, so the
   token was never stored or used.

## Phase status

| Phase | Status | Tag |
|---|---|---|
| 1. Scaffold | done | `v0.1-scaffold` |
| 2. Data and lakehouse | done | `v0.2-lakehouse` |
| 3. Features | done | `v0.3-features` |
| 4. Modelling | done | `v0.4-modelling` |
| 5. Decision policy and money | done | `v0.5-policy` |
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
