"""
generate_sample_data.py – Creates synthetic campaign & complaint data for testing.

Run from project root:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = BASE_DIR / "data" / "campaign"
COMPLAINTS_DIR = BASE_DIR / "data" / "complaints"

CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
COMPLAINTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Campaign data (2 000 rows) ─────────────────────────────────────────────────
def generate_campaign_csv(n: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.integers(20_000, 150_000, n),
            "num_campaigns": rng.integers(1, 10, n),
            "num_clicks": rng.integers(0, 50, n),
            "num_opens": rng.integers(0, 100, n),
            "recency_days": rng.integers(1, 180, n),
            "tenure_days": rng.integers(30, 1825, n),
            "gender": rng.choice(["M", "F", "Other"], n),
            "channel": rng.choice(["email", "sms", "push", "direct_mail"], n),
            "product_category": rng.choice(["electronics", "fashion", "home", "food", "travel"], n),
            "region": rng.choice(["north", "south", "east", "west", "central"], n),
        }
    )
    # Logistic probability as ground truth label
    score = (
        0.03 * df["num_clicks"]
        + 0.01 * df["num_opens"]
        - 0.003 * df["recency_days"]
        + 0.0001 * df["income"]
        - 1.5
    )
    prob = 1 / (1 + np.exp(-score))
    df["converted"] = (rng.random(n) < prob).astype(int)

    out = CAMPAIGN_DIR / "campaign_data.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Campaign data saved to {out}  (n={n}, conversion rate={df['converted'].mean():.2%})")


# ── Complaint documents ────────────────────────────────────────────────────────
COMPLAINT_TEMPLATES = [
    "Customer #{id}: I was charged twice on {date} for my {product} subscription. "
    "I called support but was put on hold for {wait} minutes without resolution.",

    "Complaint #{id}: My {product} order was delivered on {date} but the item was damaged. "
    "I requested a refund and was told it would take {wait} business days.",

    "Feedback #{id}: The {channel} service went down on {date} for {wait} hours with no notification. "
    "This caused significant disruption to my work.",

    "Issue #{id}: I tried to cancel my {product} subscription on {date} "
    "but was charged again the following month. Customer service denied the error.",

    "Report #{id}: I received an incorrect bill on {date} showing charges for {product} services "
    "I never signed up for. Total overcharge: ${overcharge}.",

    "Complaint #{id}: After upgrading my {product} plan on {date}, "
    "I lost access to features I had before. Support has not responded in {wait} days.",

    "Review #{id}: I was promised a {discount}% discount on {date} but it was never applied. "
    "I have email proof but the {channel} team is unresponsive.",
]

PRODUCTS = ["Premium", "Basic", "Pro", "Enterprise", "Starter"]
CHANNELS = ["email", "phone", "chat", "social media"]
DATES = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in [1, 5, 10, 15, 20, 25]]


def generate_complaints(n: int = 100) -> None:
    random.seed(0)
    for i in range(1, n + 1):
        template = random.choice(COMPLAINT_TEMPLATES)
        text = template.format(
            id=1000 + i,
            date=random.choice(DATES),
            product=random.choice(PRODUCTS),
            channel=random.choice(CHANNELS),
            wait=random.randint(2, 72),
            overcharge=random.randint(10, 500),
            discount=random.choice([10, 15, 20, 25, 30]),
        )
        # Add a bit more padding to make meaningful chunks
        text += " " + textwrap.fill(
            "The customer expressed strong dissatisfaction and requested escalation "
            "to a senior manager. They also mentioned they would consider switching providers "
            "if this issue is not resolved promptly. A follow-up call was scheduled.",
            width=120,
        )
        out = COMPLAINTS_DIR / f"complaint_{i:04d}.txt"
        out.write_text(text, encoding="utf-8")

    print(f"[OK] {n} complaint documents saved to {COMPLAINTS_DIR}")


if __name__ == "__main__":
    generate_campaign_csv()
    generate_complaints()
    print("\nSample data generation complete!")
    print("Next step: run  python services/conversion/train.py  to train the model.")
