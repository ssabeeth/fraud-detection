# Feature catalogue

Generated from `src/fraud/features/definitions.py` by `fraud catalogue`; a test fails if
this file is out of date. Amounts are USD.

**Point-in-time rule.** Every aggregate for a transaction at time `t` uses only
transactions of the same entity with event time strictly before `t`; a window of `W`
keeps those with `t - W <= t_e < t`. Transactions in the same second are not visible to
each other. No feature uses a label. The offline (Spark) and online (stream) code both
implement this list, and three tests hold them to it: the point-in-time test against a
naive recomputation from raw history, a canary that must fail when the window includes
the current row, and the online/offline parity test.

**Entities.** `card_key` is the pseudo-card (`card1` + `addr1` + first-seen day, see
`reports/data.md` for how stable it is); `device_key` is a coarse device fingerprint
(`DeviceType`, `DeviceInfo`, OS `id_30`, browser `id_31`, screen `id_33`), missing
without an identity record; `email_key` is the purchaser's email domain.

## History aggregates

| Feature | Reviewer would call it | Entity | Window | Definition |
|---|---|---|---|---|
| `card_txn_1h` | Card transactions, last hour | `card_key` | 1 hour | Number of transactions by this card in the previous 1 hour. |
| `card_txn_24h` | Card transactions, last 24 hours | `card_key` | 24 hours | Number of transactions by this card in the previous 24 hours. |
| `card_txn_7d` | Card transactions, last 7 days | `card_key` | 7 days | Number of transactions by this card in the previous 7 days. |
| `card_txn_30d` | Card transactions, last 30 days | `card_key` | 30 days | Number of transactions by this card in the previous 30 days. |
| `card_txn_prior` | Card's transaction history length | `card_key` | all prior history | Number of transactions by this card ever before. |
| `card_amt_24h` | Card spend, last 24 hours | `card_key` | 24 hours | Total USD spent by this card in the previous 24 hours. |
| `card_amt_7d` | Card spend, last 7 days | `card_key` | 7 days | Total USD spent by this card in the previous 7 days. |
| `card_secs_since_prev` | Time since the card's last transaction | `card_key` | all prior history | Seconds since this card's previous transaction (null if none). |
| `card_amt_zscore` | Amount unusual for this card | `card_key` | all prior history | Standard deviations between this amount and this card's previous amounts (null with fewer than two previous transactions). |
| `card_amt_ratio` | Amount vs the card's usual spend | `card_key` | all prior history | This amount divided by the mean of this card's previous amounts (null if none). |
| `card_devices_30d` | Devices used by the card, last 30 days | `card_key` | 30 days | Distinct devices used by this card in the previous 30 days. |
| `card_emails_30d` | Email domains used by the card, last 30 days | `card_key` | 30 days | Distinct email domains used by this card in the previous 30 days. |
| `card_new_device` | New device for this card | `card_key` | all prior history | 1 if this device has never been seen for this card before, else 0 (null when unknown). |
| `card_new_email` | New email domain for this card | `card_key` | all prior history | 1 if this email domain has never been seen for this card before, else 0 (null when unknown). |
| `device_txn_24h` | Device transactions, last 24 hours | `device_key` | 24 hours | Number of transactions by this device in the previous 24 hours. |
| `device_cards_7d` | Cards seen on this device, last 7 days | `device_key` | 7 days | Distinct cards used by this device in the previous 7 days. |
| `email_txn_7d` | How common this email domain is lately | `email_key` | 7 days | Number of transactions by this email domain in the previous 7 days. |

## Transaction fields

Computed from the transaction itself; no history involved.

| Feature | Reviewer would call it | Type | Definition |
|---|---|---|---|
| `TransactionAmt` | Amount (USD) | numeric | Transaction amount in US dollars. |
| `amt_cents` | Cents part of the amount | numeric | TransactionAmt minus its whole-dollar part. |
| `hour` | Hour of day | numeric | Hour of day relative to the time anchor (0-23, UTC-relative). |
| `weekday` | Day of week | numeric | Day of week relative to the anchor (0 = Thursday 30 Nov). |
| `ProductCD` | Product code | categorical | Vesta product code (W, H, C, S, R). |
| `card4` | Card network | categorical | visa, mastercard, american express, discover. |
| `card6` | Card type | categorical | debit, credit, charge card. |
| `card1` | Card issuer code | numeric | Masked card attribute (issuer/BIN-like). |
| `card2` | Card code 2 | numeric | Masked card attribute. |
| `card3` | Card country code | numeric | Masked card attribute (country-like). |
| `card5` | Card code 5 | numeric | Masked card attribute. |
| `addr1` | Billing region | numeric | Masked billing region code. |
| `addr2` | Billing country | numeric | Masked billing country code. |
| `dist1` | Distance 1 | numeric | Masked distance between addresses. |
| `P_emaildomain` | Purchaser email domain | categorical | Email domain of the purchaser. |
| `R_emaildomain` | Recipient email domain | categorical | Email domain of the recipient. |
| `email_match` | Purchaser and recipient domains match | numeric | 1 if both email domains are present and equal, 0 if both present and different. |
| `DeviceType` | Device type | categorical | mobile or desktop (identity records only). |
| `has_identity` | Has identity record | numeric | 1 if the transaction has an identity row. |

## Anonymised fields

`C1`-`C14`, `D1`-`D15`, `M1`-`M9`, `V1`-`V339` and the identity fields `id_01`-`id_38`
and `DeviceInfo` are masked by Vesta. They are used as supplied in the *all features*
model; phase 6 measures what it costs in money to leave them out. Vesta computed them
and we cannot verify that each was available at transaction time; that is recorded as
a limitation.
