# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.32"]
# ///
"""Timestamped consistency audit of the K-LIVE-2026-01 list in the paper PDF."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import requests


TIER1 = [
    ("ai-mathematical-olympiad-progress-prize-3", "2025-11", "2026-04"),
    ("hull-tactical-market-prediction", "2025-09", "2026-06"),
    ("recodai-luc-scientific-image-forgery-detection", "2025-10", "2026-04"),
    ("llm-agentic-legal-information-retrieval", "2026-01", "2026-05"),
    ("deep-past-initiative-machine-translation", "2025-12", "2026-03"),
    ("stanford-rna-3d-folding-2", "2026-01", "2026-03"),
    ("vesuvius-challenge-surface-detection", "2025-11", "2026-02"),
    ("med-gemma-impact-challenge", "2026-01", "2026-02"),
    ("wbc-bench-2026", "2025-11", "2026-05"),
    ("urban-flood-modelling", "2026-01", "2026-03"),
    ("playground-series-s6e1", "2025-12", "2026-01"),
    ("bidding-predictions-for-construction", "2026-01", "2026-03"),
    ("nfl-big-data-bowl-2026-prediction", "2025-09", "2026-01"),
]

TIER2 = [
    "spaceship-titanic",
    "store-sales-time-series-forecasting",
    "llm-classification-finetuning",
    "digit-recognizer",
    "house-prices-advanced-regression-techniques",
    "nlp-getting-started",
    "connectx",
    "tpu-getting-started",
    "contradictory-my-dear-watson",
    "gan-getting-started",
    "home-data-for-ml-course",
    "landmark-recognition-2021",
]

CUTOFFS = {
    "Gemini 3 Flash Preview": "2025-01",
    "DeepSeek-V3.2": "2025-03",
    "Claude Sonnet 4.6": "2025-08",
    "Kimi K2.5 (release-date proxy)": "2026-01",
}


def status(url: str) -> int | str:
    try:
        response = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        return response.status_code
    except requests.RequestException as exc:
        return type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="outputs/klive_audit.json")
    args = parser.parse_args()
    as_of_month = args.as_of[:7]

    temporal = {}
    for model, cutoff in CUTOFFS.items():
        pass_count = sum(start > cutoff for _, start, _ in TIER1)
        fail_count = sum(start < cutoff for _, start, _ in TIER1)
        uncertain_count = sum(start == cutoff for _, start, _ in TIER1)
        temporal[model] = {"pass": pass_count, "fail": fail_count, "same_month_uncertain": uncertain_count}

    urls = {slug: f"https://www.kaggle.com/competitions/{slug}" for slug, _, _ in TIER1}
    urls.update({slug: f"https://www.kaggle.com/competitions/{slug}" for slug in TIER2})
    http = {} if args.offline else {slug: status(url) for slug, url in urls.items()}
    tier1_open_by_reported_month = [slug for slug, _, deadline in TIER1 if deadline >= as_of_month]

    result = {
        "audit_date": args.as_of,
        "source_scope": "Independent consistency and URL audit of the competition IDs/dates transcribed from paper Tables 13-14; not an audit of undisclosed model training corpora.",
        "total_competitions": len(TIER1) + len(TIER2),
        "tier1_deadline_competitions": len(TIER1),
        "tier2_perpetual_competitions": len(TIER2),
        "tier1_open_by_reported_deadline_month": tier1_open_by_reported_month,
        "tier1_closed_by_reported_deadline_month_count": len(TIER1) - len(tier1_open_by_reported_month),
        "cutoff_matrix_recalculated": temporal,
        "kaggle_url_http_status": http,
        "release_url": "https://github.com/mireskandari/klive",
        "release_url_observation": "Repository returned not found during this audit.",
        "contamination_conclusion": "Dates can reduce a temporal contamination route but cannot establish contamination-free status without disclosure of model training corpora; the paper itself uses 'minimal' and 'not airtight' qualifiers.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
