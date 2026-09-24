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

## 2026-09-24 — Point-in-time features: strictly before, same second invisible

**Decision:** every aggregate for a transaction at `t` uses transactions of the same
entity with `t_e < t`; a window of `W` keeps `t - W <= t_e < t`. In Spark this is a
*range* frame `rangeBetween(-W, -1)` on the integer `TransactionDT` (a row frame would
let same-second rows see each other depending on their order). The online state holds
same-second events as *pending* and folds them in only when a later timestamp arrives.
No feature uses a label, so label delay cannot leak into features.

**Options considered for ties:** (a) order ties by `TransactionID` and let the later see
the earlier; (b) make them invisible to each other. (a) depends on an ordering the
stream does not guarantee, so the two implementations could disagree; (b) is what a
scorer can promise. 1.1% of fixture rows are same-second ties by construction; the
tests cover them.

## 2026-09-24 — Feature set: 17 history aggregates plus readable transaction fields

The aggregates (velocity by card over 1 h / 24 h / 7 d / 30 d / all history, spend over
24 h / 7 d, time since the previous transaction, amount z-score and ratio against the
card's history, distinct devices and email domains, new-device and new-email flags,
device velocity and cards per device, email-domain volume) are the ones a fraud
analyst reasons with. Windows are fixed, round and few, so each reason code reads
naturally. Features are defined once in `features/definitions.py`; `docs/features.md` is
generated from it and a test fails when it is stale.

## 2026-09-24 — Spark performance: private partitions for missing keys, counts by difference

Two changes made the feature job linear, with the results unchanged (the point-in-time
and parity tests pass before and after):

- rows with no device (76%) or no email domain were one window partition each; they now
  get a partition per row, since their features are null anyway;
- windowed counts are `count(before t) − count(before t − W)`. Spark recomputes a
  sliding frame's aggregate for every row, which is quadratic on a hot key
  (`gmail.com` has 228,355 rows); running counts are incremental and integers subtract
  exactly. Sums keep the sliding frame, whose partitions (cards) are small, so floating
  point matches the online path's order of addition.

## 2026-09-24 — Point-in-time and parity checks on the real data

`fraud check-pit` recomputed 17 aggregates for 4,495 sampled rows (a random 3,000 plus
fraud rows and rows of the busiest cards; 899 are fraud) from silver history with a
naive pandas implementation that shares no code with Spark: 76,415 values, **0
differences** (`reports/pit_check.json`). `fraud check-parity` replayed all 590,540
transactions through the online feature code and compared 24 values per row with the
Spark table: 14.2 million values, **0 mismatches** (`reports/parity_check.json`).

## 2026-09-24 — Imbalance: weighting, tuned; no oversampling

**Options:** (a) no correction; (b) class weights (`class_weight`, `scale_pos_weight`);
(c) random oversampling of fraud; (d) SMOTE.

**Decision:** (b), with the weight treated as a hyperparameter and chosen on validation
alongside the others, so "no weighting" wins if it is better. Oversampling was not used:
duplicated rows give the same loss as a weight while making every epoch slower, and SMOTE
invents transactions by interpolating between frauds, whose history features (counts,
time since the last transaction, new-device flags) would then describe card histories
that never existed. Weighting inflates scores, so every model is recalibrated on the
validation month (next entry) before its probabilities meet the cost model.

## 2026-09-24 — Tuning protocol and the test month

All hyperparameters, the rules' thresholds, the calibration method and (in phase 5) the
policy are chosen on the validation month. LightGBM early-stops on validation PR-AUC.
The models are trained on months 1-4 only and are **not** refitted on months 1-5 before
the test month, so the test result measures exactly the model that was tuned. Every read
of the test month goes through `load_frame(..., test_purpose=...)`, which appends the
purpose to `reports/test_touches.jsonl`; the reports print the count.

## 2026-09-24 — Calibration chosen by a time split inside validation

The policy multiplies probabilities by amounts, so they must be calibrated. Fitting a
calibrator and measuring it on the same month would flatter it, so the method (none,
Platt on the log-odds, isotonic) is chosen by a two-fold time split inside April (fit on
the first half, Brier score on the second, and the reverse), then refitted on all of
April. Test calibration is reported in `reports/model.md`.

## 2026-09-24 — Label delay and the offline split

With a 30-day chargeback delay, April's labels would not all be known until the end of
May, so a policy tuned on April could not really be frozen on 1 May. The offline split
ignores this, as backtests usually do: it measures how good the choice is, not when it
could have been made. Phase 7 and 8 honour the delay where it binds in operation:
monitoring and any retraining trigger see a label only 30 days after its transaction.
No feature uses a label, so the delay cannot leak into features.

## 2026-09-24 — Rules baseline: points on round thresholds, tuned like the models

**Decision:** five rules a fraud team would write first (large amount, burst in the last
hour, many in 24 hours, high 24-hour spend, new device with a large amount), with fixed
points and ties broken by amount. The thresholds come from a grid of round values
(432 combinations) and are chosen on validation PR-AUC, the same criterion as the
models, and in phase 5 its decline and review levels are tuned for money on the same
month as every other policy. The baseline is simple on purpose, but it gets the same
tuning budget in money as the model, so the comparison is fair.

## 2026-09-24 — Finding: one busy pseudo-card breaks the linear baseline in May

**What happened.** On the test month logistic regression's PR-AUC fell 43% (0.196 to
0.113) while LightGBM's fell 12% (0.637 to 0.561). One pseudo-card key has 1,393
transactions in May, none fraud (a business account, or several cards sharing card1,
addr1 and first-seen day), so 1.5% of May's transactions have more than 100 card
transactions in the previous 30 days, against 0.01% in April. Logistic regression
extends its (log) velocity terms linearly and 809 of its top 893 test scores come from
that key; LightGBM's trees stop at the largest split they learnt, and none of its top
1% do (89% of them are fraud). The all-history counts also creep up month by month
because the data starts on 1 December (a card's history is censored at the start).

**Decision:** keep the features and the models as tuned, and report the failure
(`reports/model.md`, generated). Capping the velocity features or dropping the key after
seeing the test month would be tuning on the test month. The monitoring phase monitors
`card_txn_24h` and `card_txn_prior`, where this shift should show up before it costs
money; a production system would also cap or winsorise velocity inputs to a linear
model, and treat very high-volume keys (merchants' own cards, corporate accounts) as a
separate population.

**Consequence:** the test month was read three times in phase 4 (the metrics, then the
diagnostic, then the metrics and diagnostic together in one final run; the models did
not change between reads). `fraud evaluate --report-only` now re-renders the report from
saved metrics, so wording fixes no longer need a test read.

## 2026-09-24 — Cost model: three actions, seven assumptions, all in USD

**Decision:** approve, decline or review, costed as in `configs/costs.yaml`: a missed
fraud loses the amount plus a $20 chargeback fee; a review costs $7 of analyst time,
stops 90% of the fraud it sees and costs a good customer $2 of delay; a false decline
costs 25% of the amount (lost margin) plus $10 (the relationship). Each value has its
rationale in the file and a range that the sensitivity table sweeps.

**Options considered for a false decline:** (a) the full amount; (b) a fixed amount;
(c) margin share plus a fixed amount. (a) overstates it (the merchant keeps the goods);
(b) ignores that declining a $2,000 order loses more than a $20 one. (c) keeps both.

## 2026-09-24 — Review capacity: 50 a day, allocated in arrival order

**Decision:** one analyst's day (50 reviews of about 10 minutes). The policies decide
one transaction at a time, as a stream would: a case qualifies for review if its value
clears the policy's threshold, and it gets a slot only if the day's queue has room. No
policy sees the rest of the day. The threshold is what saves slots for large cases, and
it is tuned on validation. A "top 50 of the day" version with hindsight is used only to
compare the two rankings on equal terms (rule 8), never as a result.

## 2026-09-24 — The expected-loss policy and its review ranking

**Decision:** decline when the expected cost of declining is lower than that of
approving; review when the expected saving of a review over the better of the two is
above a threshold. The saving is `min(p × (amount + fee), (1 − p) × friction) − (review
cost + p × (1 − catch) × (amount + fee) + (1 − p) × delay)`; with no fee, a perfect
analyst and no delay cost it is exactly `p × amount − review cost`, the brief's formula
(a test checks this). Only one number is tuned: the review threshold.

**Alternatives kept as comparisons:** probability cut-offs for decline and review (two
thresholds tuned on validation for money), and the rules baseline with a decline level and
a review level (tuned the same way). All are tuned on April with the same cost model and
capacity, so the comparison is between decision methods, not tuning effort.

## 2026-09-24 — Freezing the policy before the test month

The chosen policy (cheapest on validation among the model policies) and the rules
baseline's settings are written to `reports/policy_frozen.json` before the test month is
read. The streaming processor and the API load the policy from that file, so the online
path runs exactly what was chosen.

## 2026-09-24 — Result: the headline, and a bug in the ranking comparison

**Headline (test month, May 2018, generated in `reports/policy.md`):** the frozen
expected-loss policy on LightGBM catches 60.4% of fraud value at a total cost of
$270,554, against $474,219 for the rules baseline, which catches 16.1%: $203,664 (43%)
less. Approving everything would have cost $539,636. The policy is cheaper than the
rules at both ends of every cost assumption's range (savings of 39% to 48%).

**Bug found and fixed.** The first version of the rule 8 comparison (review the daily
top 50, approve the rest) ranked by the saving over the *better of approving and
declining*, although declining was not allowed in that comparison. That undervalues
reviewing a likely fraud that a decline would have handled, and the probability ranking
appeared to win ($348,854 against $395,104). Measured against approving, which is the
only alternative there, the expected-saving ranking costs $287,562 against $348,854. The
fix touched the validation-only comparison; the frozen policy was not affected (it can
decline, so its saving is correctly measured against the better of the two), and
`fraud policy --rerank` re-tuned it on validation, checked it matched the frozen file,
and re-rendered the report without reading the test month again.

**Also noted:** the rules baseline's tuned setting never declines; with a decline costing
a quarter of the amount plus $10, its points are not sharp enough for a decline to pay.
Logistic regression's policy declines 2,544 transactions on May, 2,187 of them
legitimate: the busy pseudo-card again (phase 4 finding).

## 2026-09-24 — Failure: the first LightGBM was too slow to explain, so trees are capped at depth 8

**What happened.** Building the streaming path (phase 7) showed that the LightGBM chosen
in phase 4 (255 leaves, no depth limit, 3,175 trees) scores a transaction in 0.96 ms
but needs 162 ms to compute its exact TreeSHAP reasons. LightGBM grows trees leaf-wise,
and these reached a mean depth of 32 (maximum 57); TreeSHAP's cost is proportional to
trees × leaves × depth². Every decision must carry its reasons, so a 162 ms explanation
breaks the "milliseconds" requirement.

**Options:** (a) keep the model and explain only declines and reviews; (b) explain
asynchronously after the decision; (c) an approximate attribution (Saabas paths);
(d) constrain the trees and re-tune. (a) and (b) break "every decision is explained";
(c) is not SHAP and would need a compiled implementation to be fast. **Decision:** (d),
`max_depth = 8` as a fixed design constraint, the grid otherwise unchanged, re-tuned on
April only.

**What it cost.** The first model's results are kept here: validation PR-AUC 0.637, test
PR-AUC 0.561 (ROC-AUC 0.903); its frozen policy cost $270,554 on May and caught 60.4% of
fraud value, against $474,219 for the rules. The depth-limited model's results replace
them in the reports, and the test month was read again to report the
new model (logged in `reports/test_touches.jsonl` with its purpose). The v0.4 and v0.5
tags point at the first model's reports.

**The depth-limited model:** 63 leaves, no class weight, platt calibration,
validation PR-AUC 0.616, test PR-AUC 0.547 (ROC-AUC
0.902). One decision takes 0.21 ms to score and
5.4 ms (p99 5.7 ms) with its reasons. Its frozen policy cost
$278,535 on May and caught
59.8% of fraud value, against
$474,219 for the rules: explaining every decision in milliseconds
cost $7,981 over the month relative to the first model.
