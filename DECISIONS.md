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
the split boundaries become ordinary dates. The anchor is an assumption: Vesta never
published it. With this anchor the data runs from 1 December 2017 to 1 June 2018, and the
six calendar months line up with the six months the brief asks for. It is verified in
phase 2 against the daily volume curve (a holiday peak should fall in late December),
and the result is recorded below.

**Consequence.** All times are naive UTC relative to the anchor. Hour-of-day features are
relative to the anchor's midnight and may be offset from local time by an unknown amount.

## 2026-09-24 — Split boundaries: calendar months, train Dec-Mar, valid Apr, test May

**Decision:** train is `[2017-12-01, 2018-04-01)`, validation `[2018-04-01, 2018-05-01)`
and test `[2018-05-01, 2018-06-02)`. The test interval ends a day late so that the handful
of transactions in the first minutes of 1 June (the last TransactionDT is about 183 days
and 38 minutes after the anchor) are not silently dropped. The boundaries live in
`configs/base.yaml` and a test rejects overlapping, gapped or empty intervals.

## 2026-09-24 — Streaming broker: Redpanda in dev-container mode, pinned version

**Decision:** `redpandadata/redpanda:v26.2.3`, one core, 1 GB, with the Kafka API on
`localhost:19092`. Redpanda is a single binary with no ZooKeeper or KRaft controller to
run, which keeps the local stack to one container. The Console UI is behind a Compose
profile, so it is not pulled unless asked for.
