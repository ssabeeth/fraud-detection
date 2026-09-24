# Decisions log

Each entry: date, the decision, options considered, and why. Newest last.

---

## 2026-09-24 — Data: IEEE-CIS Fraud Detection, used under its competition rules

**What the rules allow.** Read on the competition page
(`kaggle.com/competitions/ieee-fraud-detection/rules`) on 2026-09-24. Section 7A,
*Data Access and Use*, allows use of the Competition Data "for non-commercial purposes
only, including for participating in the Competition and on Kaggle.com forums, and for
academic research and education". Section 7B, *Data Security*, forbids transmitting,
duplicating, publishing or redistributing the data to anyone who has not agreed to the
rules.

**Decision.** This is a non-commercial portfolio project, which falls under education.
The repository never contains a raw row, not even a sample:

- `.gitignore` excludes `data/`, every CSV and zip, and the Kaggle file names from the
  first commit;
- `scripts/check_no_raw_data.py` runs as a pre-commit hook and in CI over every tracked
  file. It rejects zips and any file with a `TransactionID` column outside
  `tests/fixtures/`, and fixture files whose IDs reach the real range (real IDs start
  at 2,987,000; the synthetic generator numbers rows from 1);
- only aggregates (metrics, daily totals, plots of aggregates) are committed;
- CI runs on synthetic fixtures from `scripts/make_fixtures.py`.

**Consequence.** Anyone reproducing the results needs their own Kaggle account and must
accept the rules. The README says so.

## 2026-09-24 — Time anchor: TransactionDT = 0 is 2017-11-30 00:00 UTC

**Options:** (a) keep relative days only; (b) 2017-11-30, the anchor most public
analyses use, which puts the first transaction on 2017-12-01; (c) 2017-12-01.

**Decision:** (b). A calendar makes months, weekdays and hours readable in reports, and
the split boundaries become ordinary dates. Vesta never published the anchor, so it is
an assumption, checked against the data in phase 2 (`reports/data.md`):

- the data then runs from 1 December 2017 to 31 May 2018 23:58 (the last
  `TransactionDT` is 15,811,131 s), six whole calendar months;
- the four busiest days are 20, 22, 23 and 24 December, a pre-Christmas peak;
- Sunday is the quietest weekday and Friday the busiest;
- `D1` ("days since the card was first seen") was counted on the same day boundary as
  the anchor's midnight. Shifting the boundary by any whole hour from 1 to 23 raises the
  number of pseudo-card keys that split one card across two first-seen days, and the
  minimum is at zero (train months only; script in the phase 2 log).

**Consequence.** All times are naive UTC relative to the anchor. The hour of day is
relative to the anchor's midnight; it is the day boundary Vesta used, but its offset
from any customer's local time is unknown.

## 2026-09-24 — Split boundaries: calendar months, train Dec-Mar, valid Apr, test May

**Decision:** train is `[2017-12-01, 2018-04-01)` (417,559 rows), validation
`[2018-04-01, 2018-05-01)` (83,655) and test `[2018-05-01, 2018-06-01)` (89,326). The
boundaries live in `configs/base.yaml`; a test rejects overlapping, gapped or empty
intervals, and a gold data-quality check fails if any row falls outside every split.

## 2026-09-24 — Streaming broker: Redpanda in dev-container mode, pinned version

**Decision:** `redpandadata/redpanda:v26.2.3`, one core, 1 GB, with the Kafka API on
`localhost:19092`. Redpanda is a single binary with no ZooKeeper or KRaft controller to
run, which keeps the local stack to one container. The Console UI is behind a Compose
profile, so it is not pulled unless asked for.

## 2026-09-24 — Bronze keeps strings; silver casts and counts what does not parse

**Options:** (a) infer a schema when reading the CSV; (b) read with an explicit typed
schema; (c) read every column as a string in bronze and cast in silver.

**Decision:** (c). Bronze is the only copy of the data once the CSVs are deleted, so it
keeps exactly what was received. Silver casts with `try_cast` and counts, per column,
values that were present but did not parse; any such value fails the job instead of
silently becoming null (Spark 4 runs in ANSI mode, where a plain cast would throw on the
first bad value without saying how many there are). Bronze also refuses a CSV whose
header differs from the 394 + 41 expected columns.

## 2026-09-24 — Silver data-quality checks fail the job

Checks, each counted in one pass and written to `data/lake/_quality/silver.json`:
unparseable values; `TransactionID` unique in each table and never null; every identity
row matching a transaction; `isFraud` in {0, 1}; `TransactionDT` ≥ 0;
`TransactionAmt` present and > 0; `card1` and `ProductCD` present; `ProductCD`,
`card4`, `card6` and `DeviceType` within their known values; every event inside the
configured date range. Exact duplicate rows are dropped and counted; two different rows
with the same `TransactionID` fail. On the real data every check passes with zero
failing rows and zero duplicates (590,540 transactions, 144,233 with identity).

## 2026-09-24 — Pseudo-card key: card1 + addr1 + first-seen day, kept as is

**Definition:** `card_key = card1 _ addr1 _ (floor(TransactionDT / 86400) − D1)`, with
`NA` for a missing `addr1` or `D1`. Every part is a field of the transaction itself, so
the key is known at authorisation time. The Spark and Python implementations are tested
equal on every fixture row.

**Stability on the real data** (`reports/data.md`): 217,850 keys; 78.8% of rows belong
to a key with at least one other row; 96.6% of repeat keys have a single network, type
and two issuer codes, as one card should. The one-day-apart key pairs that would signal
D1 rounding splitting a card (44,053) are barely above the pairs two to five days apart
(about 42,700), which can only be different cards sharing `card1` and `addr1`: the excess
is about 1,400 pairs, under 1% of keys. 11.3% of rows have `addr1` or `D1` missing and
fall into coarser keys.

**Options considered:** (a) drop rows with a missing part from card features; (b) merge
keys one day apart; (c) keep the key as is. **Decision:** (c). (a) would give a sixth of
fraud-relevant rows no history at all; (b) would merge far more distinct cards than it
repairs, since adjacent pairs are mostly different cards. The coarse keys are a known
weakness and are listed as a limitation.

## 2026-09-24 — Device and email keys

**Decision:** the device key is `DeviceType|DeviceInfo|id_30|id_31|id_33` (type, model
or OS string, OS version, browser, screen), and is missing when `DeviceInfo` is. The
email key is `P_emaildomain`. Neither is a person: a domain like `gmail.com` covers most
customers, so email features measure how common a domain has been lately rather than
tracking an individual. A missing key gives missing (not zero) features, so rows
without an identity record are not pooled into one giant "device".

## 2026-09-24 — pandas below 3

PySpark 4.2 warns that it does not yet support pandas 3, so `pandas<3` is pinned until it
does.
