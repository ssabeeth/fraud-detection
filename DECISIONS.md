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
- four of the five busiest days are 20, 22, 23 and 24 December, a pre-Christmas peak
  (the other is 2 March);
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

## 2026-09-24 — Reason codes from LightGBM's own TreeSHAP

**Options:** (a) the `shap` package's `TreeExplainer`; (b) LightGBM's `pred_contrib`,
which implements the same TreeSHAP algorithm inside the booster.

**Decision:** (b). It gives exact SHAP values for the tree ensemble, needs no extra
dependency in the scoring image (the `shap` package pulls in numba and llvmlite), and one
call returns both the probability (the contributions plus the bias sum to the raw
log-odds; a test checks this) and the reasons. The reasons are the three features with
the largest positive contributions, i.e. the ones that pushed this transaction towards
fraud. SHAP values are in log-odds before calibration; calibration is monotone, so it
does not change which features pushed the score up.

## 2026-09-24 — Naming masked features by the family Vesta published

A reviewer cannot be told what `V258` measures, because Vesta did not say. Reasons name
masked columns by family ("V258 — Vesta risk signal (masked)", "C13 — count of linked
entities (masked)"), which is honest about what is known. How often a masked column is
the top reason, and what it costs to drop them, are both reported, since that is the
governance question: a model whose reasons cannot be explained to a customer or a
regulator carries model risk that the money must justify.

## 2026-09-24 — The explainable-only experiment uses the same protocol

The explainable model gets the same LightGBM grid, early stopping, calibration choice and
expected-loss policy tuned on validation as the main model; only the feature set
differs (36 features against 452). The difference in money on the test month is the
price of explainability under this cost model.

## 2026-09-24 — Segment checks, and why they are not a fairness audit

Alert rate, precision, recall and fraud value caught are reported by product, card
network, card type, device type and email domain (top eight, the rest pooled). The data
has no protected characteristics, so fairness in the legal sense cannot be assessed; the
report says so and lists what an audit would need (the characteristics or a lawful proxy,
a stated criterion such as equal false-decline rates, and outcomes for declined
customers).

## 2026-09-24 — Model and data cards generated from the reports

`docs/model_card.md` and `docs/data_card.md` are written by `fraud cards` from the report
JSON files, so every number in them comes from a run and they cannot drift from the
results.

## 2026-09-24 — Result: explainability costs $92,513 a month; Vesta's columns carry the history

Generated in `reports/explainability.md`. On May 2018 the explainable-only LightGBM
(36 features, same protocol) costs $371,047 against $278,535 for the all-features model:
$92,513 (33%) more, and it catches 46.9% of fraud value against 59.8%. 63% of the
all-features model's alerts have a masked column as their top reason (C13 alone 18%).

This project's own point-in-time aggregates account for only 5.7% of the all-features
model's mean |SHAP|, while in the explainable model the time since the card's last
transaction, the history length and 7-day spend are among its top 20 features. The
likely reading is that Vesta's masked C (linked-entity counts) and D (time deltas)
columns already encode the entity history, computed on Vesta's side with its real card
identifiers; the aggregates matter when those columns are removed. It also means the
all-features model leans on columns whose point-in-time correctness cannot be checked
here, which is recorded as a limitation.

Segment checks show the burden tracking the fraud rate: credit cards are alerted at
10.2% (fraud rate 6.4%) and debit at 2.8% (2.6%); product W, 78% of volume, has the
lowest precision (21%) and recall (32%).

## 2026-09-24 — Streaming: one partition, in-memory state warmed from the lake

**Decision:** each topic has one partition, and the processor keeps per-key state in
memory, warmed at start-up with every earlier transaction from the lake. One partition
preserves a single event-time order, which the point-in-time rule relies on for card,
device and email state at once.

**What scaling out would change:** partitioning by card key keeps card state local, but
device and email features aggregate across cards, so they would need their own keyed
stages (repartition by device key, by email domain) whose outputs are joined back, or a
shared low-latency state store. Warm-up from the lake would become a snapshot of that
store. Both are what a feature store provides; see "What production would add".

## 2026-09-24 — Latency is measured per event, including SHAP

The processor scores one event at a time (no micro-batching) and computes TreeSHAP for
every decision, since every decision must carry reasons. The single-event path builds the
model input as a NumPy row from the fitted category levels (no pandas), and one
`pred_contrib` call gives both the probability and the reasons. A test checks that
single-event scores equal the batch scores to 1e-10.

## 2026-09-24 — Labels on their own topic, 30 days late, and the clock runs on

Each label is published at `TransactionDT + 30 days` on the simulated clock. Labels of
April transactions fall due during the May replay and are published then; after the last
May transaction the clock runs on for 30 more days so May's own labels arrive, as they
would in June. The monitor reads labels only from this topic.

## 2026-09-24 — Stream sink: Spark Structured Streaming to Delta bronze, raw JSON

Each topic lands in its own bronze table with the raw message value and Kafka metadata
(topic, partition, offset, timestamp), with checkpoints so a re-run only appends new
messages. Parsing happens downstream (the parity test and the monitor read the JSON),
as with the batch bronze tables.

## 2026-09-24 — Two replays: as fast as possible, and paced

The test month is replayed twice. As fast as possible, the decisions per second are the
processor's capacity and every feature is compared with the offline table. Paced at
3,600× real time (a month in about 12 minutes), the processor keeps up with arrivals, so
end-to-end latency (producer send to decision published) is what a transaction would
see; replayed flat out, end-to-end latency is mostly time spent queued behind the
backlog the producer builds, which says nothing about the system.

## 2026-09-24 — Stream landing zone reset per replay

Each replay recreates its topics, so Kafka offsets start again at zero. The Spark sink's
checkpoint from the previous replay would then skip every new message below the old
offsets. `fraud stream` therefore empties the stream bronze tables and their checkpoints
before each replay; a test replays twice and checks the second lands the same rows.

## 2026-09-24 — Result: the stream agrees with the offline system

On the real test month through Redpanda (`reports/stream.md`): 89,326 transactions ×
20 values read back from Delta bronze, 0 mismatches with the offline feature table;
every action equal to the frozen policy applied offline to the same rows; the largest
difference in P(fraud) between the online (NumPy row) and offline (pandas batch) paths
1.2e-14; no decision without reasons. One process makes 183 decisions a second (5.4 ms
each at the median, with SHAP); paced at 1,800× real time the end-to-end latency is
15.5 ms at the median and 56 ms at the 99th percentile.

## 2026-09-24 — Scoring API: stateless, history in the request

`/score` takes the transaction and the earlier transactions of the same card, device and
email domain, and computes the features with the same online code as the processor. A
stateless API is easy to scale and test, and it cannot drift from the processor's
state; the price is that the caller must supply the history, which in production would
come from a feature store. History at or after the transaction's time is rejected (422).
The daily review capacity belongs to the queue, so the API returns
`review_recommended` and leaves capacity to the processor.

## 2026-09-24 — Monitoring thresholds

- **Score drift:** PSI of P(fraud) against validation deciles; warn at 0.1, alert at 0.2,
  the conventional bands for PSI.
- **Input drift:** Evidently's per-column tests (normalised Wasserstein distance for
  numbers, Jensen-Shannon for categories, each at its 0.1 default), plus a missing-values
  check (a column's missing share moving by 0.2 or more) because a silent feed shows up as
  missing values that a distribution test on the remaining values cannot see. Warn when 30%
  of monitored inputs drift.
- **Feed health:** on the latest day alone, a field present on at least 5% of
  reference rows that is present on less than half its usual share raises a feed
  alert. Added after the first monitoring run: the 7-day drift window caught the stress
  scenario's silent identity feed only ten days in, and an absolute change in the missing
  share misses fields that are usually missing. A feed alert means fix the feed; it never
  triggers a retrain, which would learn from broken inputs.
- **Alert rate:** warn when declines plus reviews move by half against validation.
- **Performance:** per weekly cohort, once its labels are all in; alert if PR-AUC falls
  below 80% of validation's or fraud value caught drops by 10 points.
- **Retrain:** on a performance alert, or on 7 consecutive days of score-drift alerts
  backed by an input-drift warning. A single noisy day never triggers it.

The reference is the validation month, the data the policy was chosen on. The windows
are seven days, long enough to smooth the weekly cycle.

## 2026-09-24 — Stress scenario: the identity feed goes silent

To show the monitors firing, the test month is also replayed in-process through the same
`OnlineScorer` with every device and identity field blank from 22 May, as when an
upstream fingerprinting service fails. This is a synthetic change applied to real
transactions, reported separately from the true replay and never mixed into any result.

## 2026-09-24 — Scoring image: small, model baked in, never the real model in public

**Decision:** `docker/Dockerfile` installs only the base dependencies and the `serve`
extra (FastAPI, LightGBM, scikit-learn); Spark, Arrow and matplotlib moved out of the
base set into the extras that use them, which cut the image's Python environment from
658 MB to 416 MB. The model bundle and the frozen policy are copied in at build time, so
a container needs no volume or network to start. CI builds the image with a model trained
on the synthetic fixtures and publishes that one; a model trained on the competition data
is only ever built locally (`make image`), because a public image is a form of sharing and
the rules forbid sharing the data.

## 2026-09-24 — Result: what the monitors saw on the replay

Generated in `reports/monitoring.md`. On the streamed test month the input-drift monitor
warned on 16 of 25 days, driven by the velocity and history features (`card_txn_24h`,
`card_txn_prior`, `email_txn_7d`): the same shift from the busy pseudo-card and the
growing card histories that broke logistic regression in phase 4. The score PSI never
passed 0.011 and the alert rate moved from 4.0% to at most 5.3%, so no retrain was
triggered; the complete weekly cohorts, known only in June, confirm the model held
(PR-AUC 0.49 to 0.59 against 0.62 on validation, fraud value caught 56% to 65%). In the
stress scenario the feed-health check raised an alert on 23 May, the first full day of
the silent identity feed, and not once on the real replay; the model's own performance
barely moved, because identity fields are missing for three quarters of transactions
anyway.

## 2026-09-24 — Databricks: the local CLI as Jobs from a wheel, on serverless

**Decision:** an Asset Bundle (`databricks.yml`, `databricks/resources.yml`) creates a
schema, a `raw` volume for the owner's upload, a `lake` volume, and one job whose tasks
are the same `fraud` CLI commands as the local pipeline, run from the project's wheel
with `--data-dir /Volumes/workspace/fraud/lake`. The wheel carries `configs/` (packaged as
`fraud/_configs`), so it runs without the repository; MLflow switches to the workspace
tracking server and the Unity Catalog registry when it detects Databricks. Nothing is
downloaded inside the workspace, which Free Edition would block anyway. A test parses
every task's parameters with the real CLI parser, so the bundle cannot drift from it.

**Status:** not deployed. It needs the owner's Free Edition workspace and
`databricks auth login`, which the brief lists as the owner's. `databricks bundle
validate` also needs that login, so the bundle is checked offline only: the wheel was
built, installed into a clean environment and run outside the repository.

## 2026-09-24 — Merging phases that are ready but not deployed

Phases 9 and 10 need the owner's accounts to finish (a Databricks run, a Terraform
apply). Their code is complete and checked without credentials, so they are merged into
`main` with tags that say so (`v0.9-databricks-ready`, `v0.10-azure-ready`), and phases
11 and 12, which need no accounts, go ahead. `v1.0` is left for when the owner has run
the Databricks job, applied the Terraform and published the dashboard.

## 2026-09-24 — Azure slice: Container Apps on the Consumption plan, budget first

**Options:** Azure Container Apps (Consumption), Azure Container Instances, App Service,
Azure Functions. **Decision:** Container Apps with `min_replicas = 0`: it runs the same
image as local Docker, scales to zero when idle (billed per vCPU- and GiB-second above a
monthly free grant), and its environment has no fixed fee. Container Instances bill while
running, App Service plans bill while idle, and Functions would need the API rewritten.
No Log Analytics workspace (ingestion is billed) and no container registry (the image is
public on GitHub Container Registry). A resource-group budget with e-mail alerts at 50%
of actual and 100% of forecast spend is applied first, on its own
(`terraform apply -target=...`), before anything that could cost money.

**Status:** applied 2026-09-25 with the owner's `az login` (see below). CI runs
`terraform fmt -check`, `init -backend=false`, `validate` and `tflint` (azurerm ruleset)
with no credentials. The provider lock file covers linux_amd64, darwin_arm64 and
darwin_amd64 so CI verifies the same provider build.

## 2026-09-24 — Dashboard: one daily CSV, built by the owner in Tableau Public

**Decision:** `fraud policy` writes `exports/daily_policy_results.csv`, one row per day
of the test month and policy (155 rows, 16 KB): transactions, fraud, fraud value and
value caught, declines and false declines, reviews and reviews that were fraud, false
alarms, and the three costs and their total. Nothing is row-level, so it can be published
with the workbook. A test checks that its monthly totals equal the policy report to the
dollar. `docs/tableau.md` is the build guide (connection, calculated fields, five sheets,
layout, the check before publishing); the owner builds and publishes, as the brief says,
since Tableau Public needs the owner's account and publishing is public.

## 2026-09-25 — Azure: what a new subscription needed

The first full apply stopped at the Container Apps environment: a new subscription does
not have the `Microsoft.App` resource provider registered, and the azurerm provider's
default "core" registrations leave it out. **Decision:** the provider block lists it in
`resource_providers_to_register`, so `apply` registers it (free), rather than a manual
`az provider register` that the next person would have to know about. The second plan
then wanted to remove a `Consumption` workload profile that Azure adds to every new
environment, and Azure put it back after each apply. **Decision:** declare that profile
(pay per use, no fixed fee) on the environment and the app; the plan now reports no
changes. The live API answered `/health` and a synthetic `/score` (52 s for the first
request from zero, 28 ms inside the API after).

## 2026-09-25 — Databricks: what the first real run changed

Four problems appeared only on the workspace, each fixed in code with a test where one
fits:

1. **Development mode renamed the schema** to `dev_<user>_fraud`, while the job's volume
   paths and `make databricks-upload` use `/Volumes/workspace/fraud`. **Decision:** no
   `mode` on the target (the job has no schedule to pause and runs as the deploying user
   either way); a test ties the paths to the schema. The empty misnamed schema and
   volumes were replaced with the owner's approval.
2. **Serverless environment 3 ships pandas 1.5 and numpy 1.26**, and serverless refuses a
   wheel that upgrades core packages. Loosening the wheel's minimums would run the code
   on versions it was never tested with. **Decision:** environment version 6 (pandas
   2.3.3, numpy 2.3.4, scikit-learn 1.7.2, lightgbm 4.6.0), which meets every minimum.
   The wheel's version carries a build stamp so a redeploy is never served a cached
   install.
3. **Unity Catalog model names** have three levels and no hyphens, and registration
   needs a signature. **Decision:** `registered_name()` gives `fraud-<model>` locally and
   `<catalog>.<schema>.fraud_<model>` in Unity Catalog; both models log a signature
   everywhere, and the test registers LightGBM and checks both signatures. skops joins
   the job's libraries for the logistic regression's format.
4. **The point-in-time check ran for 57 minutes** before it was cancelled (25 s on the
   laptop). It split the whole history into one frame per key (217,850 card keys) to
   recompute 4,495 rows. **Decision:** group only the keys the sample needs, each with
   its full history, so the check is as independent as before and gives the same
   result; each stage now logs its time. It then took 13.2 minutes on serverless.

The job's evaluate and policy tasks read the test month again. That is a reproduction of
the reported run, not a new result, so `--test-note` labels those reads in the workspace's
test-read log. The run was repaired from `check_pit` onward rather than restarted,
because bronze deletes the uploaded CSVs once they are converted.

## 2026-09-25 — Dashboard: a page built by code on GitHub Pages, not Tableau Public

The owner decided not to use Tableau. **Options:** a Streamlit app (as in the owner's
other repositories), a hosted notebook, or a static page. **Decision:** a static page,
`site/index.html`, written by `fraud dashboard` from the same daily export and the policy
report, and published by a GitHub Pages workflow. It needs no account beyond GitHub and
costs nothing, and like every other number in the repository it comes from code: a test
rebuilds the page and fails if the committed copy differs, and checks the headline figures
against `reports/policy_results.json`. It has the five views the Tableau guide planned
(month totals, cost per day, cumulative saving against the rules, cost breakdown, the
review queue against capacity) and the sensitivity table, drawn as inline SVG with hover
values, no JavaScript and no external files, in light and dark themes. Streamlit would
need a running server and the owner's Streamlit account for a page that never changes.
`docs/tableau.md` is removed; the CSV stays, for any BI tool.
