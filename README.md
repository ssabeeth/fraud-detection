# Real-Time Card Fraud Detection

[![CI](https://github.com/ssabeeth/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ssabeeth/fraud-detection/actions/workflows/ci.yml)

A production-style fraud system that scores card transactions as they stream in, gives
a reason for every decision, and decides what to block or send for review by
**expected money lost** rather than by a probability cut-off. It is trained and
evaluated only on the past, with features that could have been computed at the moment
of each transaction.

> **Status: under construction.** See [PROGRESS.md](PROGRESS.md) for what is done and
> [DECISIONS.md](DECISIONS.md) for why.

## Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Vesta), about 590,000 card-not-present transactions over six months, 3.5% fraud.
Amounts are in **US dollars**. The competition rules allow non-commercial use for
education and research and forbid redistribution, so **this repository contains no raw
rows**. To reproduce the results you need a Kaggle account that has accepted the rules.
CI runs on synthetic data with the same schema.

## Quickstart

```bash
make install     # uv virtualenv and pre-commit hooks
make test        # unit tests
make up          # Redpanda on localhost:19092
```
