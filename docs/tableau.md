# Building the business dashboard in Tableau Public

The dashboard shows, day by day over the test month (May 2018), what each decision policy
cost and how much fraud it stopped. It is built from one small aggregated file,
`exports/daily_policy_results.csv` (written by `fraud policy`): one row per day and policy,
no transaction-level data, so it can be published.

Tableau Public is free: install **Tableau Public** (desktop) from
<https://www.tableau.com/products/public/download> and sign in with a Tableau Public
account. Workbooks saved to Tableau Public are public.

## The data

| Column | Meaning |
|---|---|
| `date` | Day (2018-05-01 to 2018-05-31) |
| `policy` | `approve_all`, `rules`, `logreg_expected_loss`, `lightgbm_cutoffs`, `lightgbm_expected_loss` |
| `transactions` | Transactions that day |
| `fraud_transactions` | Of which fraud |
| `fraud_value_usd` | Fraud dollars that day |
| `fraud_value_caught_usd` | Fraud dollars stopped (declined, or reviewed × analyst catch rate) |
| `declines`, `false_declines` | Declines, and those that were legitimate |
| `reviews`, `reviews_fraud` | Reviews, and those that were fraud |
| `false_alarms` | Legitimate transactions declined or reviewed |
| `missed_fraud_cost_usd` | Fraud let through, plus chargeback fees |
| `decline_friction_usd` | Cost of declining good customers |
| `review_cost_usd` | Analyst time plus the delay cost of reviewed good orders |
| `total_cost_usd` | The sum of the three costs above |

## 1. Connect

1. Open Tableau Public → **Connect → To a File → Text file** → pick
   `exports/daily_policy_results.csv`.
2. On the data source page check the types: `date` is a Date (click the icon above the
   column if it shows as a string), every `_usd` and count column is a Number (whole or
   decimal), `policy` is a String.
3. Click **Sheet 1**.

## 2. Calculated fields

Create each with **Analysis → Create Calculated Field**.

- **Policy name**
  ```
  CASE [policy]
    WHEN "approve_all" THEN "Approve everything"
    WHEN "rules" THEN "Rules baseline"
    WHEN "logreg_expected_loss" THEN "Logistic regression, expected loss"
    WHEN "lightgbm_cutoffs" THEN "LightGBM, probability cut-offs"
    WHEN "lightgbm_expected_loss" THEN "LightGBM, expected loss (chosen)"
  END
  ```
- **Fraud value caught %**: `SUM([fraud_value_caught_usd]) / SUM([fraud_value_usd])`
  (format as percentage, one decimal).
- **Cost per 1,000 transactions**: `SUM([total_cost_usd]) / SUM([transactions]) * 1000`
- **Rules cost (same day)**:
  `{ FIXED [date] : SUM(IF [policy] = "rules" THEN [total_cost_usd] END) }`
- **Saving vs rules**: `MIN([Rules cost (same day)]) - SUM([total_cost_usd])`. Used with
  one policy and `date` in the view, it is that day's rules cost minus the policy's cost
  (`MIN` because the fixed value repeats on every row of the day).
- **Cumulative saving vs rules**: `RUNNING_SUM([Saving vs rules])` (a table calculation;
  compute using `date`).

## 3. Sheets

1. **KPIs.** Filter `policy` to `lightgbm_expected_loss` and `rules`. Put `Policy name` on
   Rows; `SUM(total_cost_usd)`, `Fraud value caught %`, `SUM(reviews)` and
   `SUM(false_declines)` on Text via Measure Names/Measure Values. Format money as currency
   (USD, no decimals). Name the sheet "Month totals".
2. **Daily cost.** `date` (exact date, continuous) on Columns, `SUM(total_cost_usd)` on
   Rows, `Policy name` on Colour. Filter out `approve_all` (it dwarfs the others; mention
   its total in the caption). Line chart. Name it "Cost per day".
3. **Cumulative saving.** Filter `policy` to `lightgbm_expected_loss`. `date` on Columns,
   `Cumulative saving vs rules` on Rows, area chart. Name it "Saving against rules".
4. **Where the money goes.** Filter to the chosen policy and the rules. `Policy name` on
   Rows; `SUM(missed_fraud_cost_usd)`, `SUM(decline_friction_usd)`, `SUM(review_cost_usd)`
   as a stacked bar (Measure Values on Columns, Measure Names on Colour). Name it "Cost
   breakdown".
5. **Workload.** Chosen policy only: `date` on Columns, `SUM(reviews)` and
   `SUM(reviews_fraud)` as dual bars. Add a reference line at 50 (the daily review
   capacity). Name it "Review queue".

## 4. Dashboard

1. **New Dashboard**, size **Fixed 1200 × 850**.
2. Title (Text object): *Card fraud: what each decision policy cost, May 2018 (USD)*.
3. Top row: "Month totals". Middle: "Cost per day" (left, two thirds) and "Cost breakdown"
   (right). Bottom: "Saving against rules" (left) and "Review queue" (right).
4. Use the `Policy name` colour legend once, top right; make every sheet use the same
   colours (Rules baseline grey, LightGBM expected loss dark blue).
5. Caption (Text object, bottom): *Policies were tuned and chosen on April 2018 and
   frozen; May is out of sample. Costs follow the assumptions in configs/costs.yaml
   (chargeback fee, analyst time, lost margin on false declines). Data: IEEE-CIS Fraud
   Detection (Vesta, Kaggle), aggregated. Code: github.com/ssabeeth/fraud-detection.*

## 5. Check the numbers, then publish

Before publishing, the "Month totals" sheet must match `reports/policy.md` (test month
table): the same total cost and fraud value caught for the rules baseline and the chosen
policy, to the dollar. Then **File → Save to Tableau Public**, name it *Card fraud
decision policies*, and add the link to the README.
