# Progress

**Status: phase 2 (data and lakehouse) done; phase 3 (features) next.**

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
