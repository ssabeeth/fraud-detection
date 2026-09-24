"""Generate synthetic IEEE-CIS lookalike files for CI and tests.

The competition data may not be redistributed, so CI runs on this instead. The files
have every column of ``train_transaction.csv`` and ``train_identity.csv`` with
plausible types and values, and enough structure to exercise the pipeline:

- cards with a stable card1 + addr1 + first-seen day, so the pseudo-card key works,
  with several cards sharing a card1 and a small share of D1 values off by one day
  (the instability the real key has);
- repeat transactions within minutes, hours and days, so velocity windows fill;
- a few transactions at exactly the same second as the card's previous one, so the
  "strictly before" rule in the point-in-time features is tested on ties;
- compromised cards whose later transactions are a fraud burst (larger amounts, often a
  new device or email), and a few cards that are fraudulent from the start;
- identity rows for about a quarter of transactions, more often for fraud.

IDs start at 1. Real IDs start at 2,987,000, and ``check_no_raw_data.py`` uses that
gap to tell synthetic fixtures from real rows.

Usage: ``uv run python scripts/make_fixtures.py [--out tests/fixtures] [--cards 400]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.lakehouse.schema import IDENTITY_COLUMNS, TRANSACTION_COLUMNS

DAY = 86_400
N_DAYS = 183
EMAILS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "outlook.com",
    "icloud.com",
    None,
]
FRAUD_EMAILS = ["gmail.com", "hotmail.com", "outlook.com", "protonmail.com", "mail.com"]
DEVICES = [
    ("desktop", "Windows"),
    ("desktop", "MacOS"),
    ("mobile", "iOS Device"),
    ("mobile", "SM-G960U Build/R16NW"),
    ("mobile", "Moto G (5) Build/NPP25.137-93"),
    (None, None),
]
NETWORKS = ["visa", "mastercard", "american express", "discover"]
NETWORK_P = [0.65, 0.30, 0.03, 0.02]
CARD_TYPES = ["debit", "credit", "charge card", "debit or credit"]
CARD_TYPE_P = [0.72, 0.27, 0.005, 0.005]
PRODUCTS = ["W", "H", "C", "S", "R"]
PRODUCT_P = [0.72, 0.06, 0.12, 0.03, 0.07]


def _cards(rng: np.random.Generator, n: int) -> pd.DataFrame:
    card1_pool = rng.integers(1000, 18397, size=max(n // 3, 10))
    addr_pool = rng.integers(100, 541, size=40).astype(float)
    device_idx = rng.integers(0, len(DEVICES), size=n)
    return pd.DataFrame(
        {
            "card_id": np.arange(n),
            "card1": rng.choice(card1_pool, size=n),
            "card2": rng.integers(100, 601, size=n).astype(float),
            "card3": rng.choice([150.0, 185.0, 144.0], size=n, p=[0.9, 0.07, 0.03]),
            "card4": rng.choice(NETWORKS, size=n, p=NETWORK_P),
            "card5": rng.choice([226.0, 224.0, 166.0, 102.0, 117.0], size=n),
            "card6": rng.choice(CARD_TYPES, size=n, p=CARD_TYPE_P),
            "addr1": np.where(rng.random(n) < 0.05, np.nan, rng.choice(addr_pool, size=n)),
            "addr2": np.where(rng.random(n) < 0.05, np.nan, 87.0),
            "first_seen_day": rng.integers(-200, N_DAYS - 10, size=n),
            "email": rng.choice(np.array(EMAILS, dtype=object), size=n),
            "device_type": [DEVICES[i][0] for i in device_idx],
            "device_info": [DEVICES[i][1] for i in device_idx],
            "product": rng.choice(PRODUCTS, size=n, p=PRODUCT_P),
            "rate": rng.lognormal(mean=-3.7, sigma=0.9, size=n),
            "compromised": rng.random(n) < 0.15,
            "fraud_only": rng.random(n) < 0.03,
        }
    )


def _events(rng: np.random.Generator, card: pd.Series) -> list[dict]:
    start = max(int(card.first_seen_day), 1)
    span = N_DAYS - start
    if span <= 0:
        return []
    n = max(1, rng.poisson(card.rate * span) + 1)
    # Sessions: each is a day, and each holds a small burst of transactions.
    days = np.sort(rng.integers(start, N_DAYS, size=n))
    times: list[int] = []
    for d in days:
        t = d * DAY + int(rng.integers(0, DAY))
        times.append(t)
        for _ in range(rng.poisson(0.4)):
            t = t + int(rng.exponential(900)) + 1
            times.append(t)
    times.sort()
    events = [
        {"t": t, "fraud": bool(card.fraud_only), "email": card.email, "device": card.device_info}
        for t in times
    ]
    if card.compromised and not card.fraud_only and len(events) >= 2:
        t0 = events[int(rng.integers(len(events) // 2, len(events)))]["t"] + int(
            rng.integers(3600, 5 * DAY)
        )
        new_email = rng.random() < 0.5
        new_device = rng.random() < 0.5
        t = t0
        for _ in range(int(rng.integers(3, 9))):
            events.append(
                {
                    "t": t,
                    "fraud": True,
                    "email": rng.choice(FRAUD_EMAILS) if new_email else card.email,
                    "device": DEVICES[int(rng.integers(0, 5))][1] if new_device else None,
                }
            )
            t += int(rng.exponential(1800)) + 1
    return [e for e in events if DAY <= e["t"] < N_DAYS * DAY]


def generate(n_cards: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cards = _cards(rng, n_cards)
    rows = []
    for card in cards.itertuples(index=False):
        for e in _events(rng, card):
            rows.append({**card._asdict(), **e})
    tx = pd.DataFrame(rows).sort_values(["t", "card_id"], kind="stable").reset_index(drop=True)

    # Same-second ties: copy the time of the card's previous transaction.
    prev_same_card = tx.groupby("card_id")["t"].shift(1)
    tie_rows = rng.choice(tx.index[prev_same_card.notna()], size=25, replace=False)
    tx.loc[tie_rows, "t"] = prev_same_card.loc[tie_rows].astype(int)
    tx = tx.sort_values(["t", "card_id"], kind="stable").reset_index(drop=True)

    n = len(tx)
    amount = np.where(
        tx["fraud"],
        rng.lognormal(mean=4.6, sigma=1.0, size=n),
        rng.lognormal(mean=3.8, sigma=0.9, size=n),
    )
    day_int = tx["t"] // DAY
    d1 = (day_int - tx["first_seen_day"]).astype(float)
    off_by_one = rng.random(n) < 0.02
    d1 = d1 + np.where(off_by_one, rng.choice([-1, 1], size=n), 0)
    d1 = d1.clip(lower=0)

    out = pd.DataFrame(index=tx.index, columns=TRANSACTION_COLUMNS, dtype=object)
    out["TransactionID"] = np.arange(1, n + 1)
    out["isFraud"] = tx["fraud"].astype(int)
    out["TransactionDT"] = tx["t"].astype(int)
    out["TransactionAmt"] = np.round(amount, 3)
    out["ProductCD"] = tx["product"]
    for c in ("card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"):
        out[c] = tx[c]
    out["dist1"] = np.where(rng.random(n) < 0.6, np.nan, np.round(rng.exponential(60, n)))
    out["P_emaildomain"] = tx["email"]
    out["R_emaildomain"] = np.where(rng.random(n) < 0.75, None, tx["email"])
    for i in range(1, 15):
        out[f"C{i}"] = rng.poisson(1.5, n).astype(float)
    out["D1"] = d1
    out["D2"] = np.where(d1 > 0, d1, np.nan)
    for i in range(3, 16):
        out[f"D{i}"] = np.where(rng.random(n) < 0.5, np.nan, rng.integers(0, 300, n))
    for i in (1, 2, 3, 5, 6, 7, 8, 9):
        out[f"M{i}"] = rng.choice(np.array(["T", "F", None], dtype=object), n)
    out["M4"] = rng.choice(np.array(["M0", "M1", "M2", None], dtype=object), n)
    for i in range(1, 21):
        out[f"V{i}"] = rng.poisson(0.5, n).astype(float)

    has_identity = rng.random(n) < np.where(tx["fraud"], 0.5, 0.2)
    idx = np.flatnonzero(has_identity)
    m = len(idx)
    ident = pd.DataFrame(index=range(m), columns=IDENTITY_COLUMNS, dtype=object)
    ident["TransactionID"] = out.loc[idx, "TransactionID"].to_numpy()
    ident["id_01"] = rng.choice([0.0, -5.0, -10.0, -20.0], m)
    ident["id_02"] = np.round(rng.lognormal(11.5, 1.0, m))
    ident["id_05"] = rng.integers(-5, 10, m).astype(float)
    ident["id_06"] = -rng.integers(0, 30, m).astype(float)
    ident["id_11"] = 100.0
    ident["id_12"] = rng.choice(np.array(["Found", "NotFound"], dtype=object), m)
    ident["id_13"] = rng.choice([49.0, 52.0, 64.0], m)
    ident["id_15"] = rng.choice(np.array(["New", "Found", "Unknown"], dtype=object), m)
    ident["id_17"] = rng.choice([166.0, 225.0], m)
    ident["id_19"] = rng.integers(100, 671, m).astype(float)
    ident["id_20"] = rng.integers(100, 661, m).astype(float)
    ident["id_28"] = rng.choice(np.array(["New", "Found"], dtype=object), m)
    ident["id_29"] = rng.choice(np.array(["Found", "NotFound"], dtype=object), m)
    ident["id_30"] = rng.choice(np.array(["Windows 10", "iOS 11.2.1", "Android 7.0"]), m)
    ident["id_31"] = rng.choice(np.array(["chrome 63.0", "mobile safari 11.0", "edge 16.0"]), m)
    ident["id_32"] = rng.choice([24.0, 32.0], m)
    ident["id_33"] = rng.choice(np.array(["1920x1080", "2208x1242", "1366x768"]), m)
    ident["id_34"] = rng.choice(np.array(["match_status:2", "match_status:1"]), m)
    for c in ("id_35", "id_36", "id_37", "id_38"):
        ident[c] = rng.choice(np.array(["T", "F"], dtype=object), m)
    info = tx.loc[idx, "device"].to_numpy()
    ident["DeviceInfo"] = info
    ident["DeviceType"] = [
        next((k for k, v in DEVICES if v == d), None) if d is not None else None for d in info
    ]
    return out, ident


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--cards", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    tx, ident = generate(args.cards, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tx.to_csv(args.out / "train_transaction.csv", index=False)
    ident.to_csv(args.out / "train_identity.csv", index=False)
    print(
        f"wrote {len(tx):,} transactions ({tx['isFraud'].mean():.1%} fraud) "
        f"and {len(ident):,} identity rows to {args.out}"
    )


if __name__ == "__main__":
    main()
