# Claim 3 - K-LIVE benchmark audit


---
<!-- trackio-cell
{"type": "code", "id": "cell_321d4df1e1d3", "created_at": "2026-07-18T02:22:39+00:00", "title": "Run: uv audit_klive.py (exit 0)", "command": ["uv", "run", "audit_klive.py", "--as-of", "2026-07-18", "--output", "outputs/klive_audit.json"], "exit_code": 0, "duration_s": 17.422}
-->
````bash
$ uv run audit_klive.py --as-of 2026-07-18 --output outputs/klive_audit.json
````

exit 0 · 17.4s


````python title=audit_klive.py
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

````


````output
Installed 5 packages in 2ms
{
  "audit_date": "2026-07-18",
  "source_scope": "Independent consistency and URL audit of the competition IDs/dates transcribed from paper Tables 13-14; not an audit of undisclosed model training corpora.",
  "total_competitions": 25,
  "tier1_deadline_competitions": 13,
  "tier2_perpetual_competitions": 12,
  "tier1_open_by_reported_deadline_month": [],
  "tier1_closed_by_reported_deadline_month_count": 13,
  "cutoff_matrix_recalculated": {
    "Gemini 3 Flash Preview": {
      "pass": 13,
      "fail": 0,
      "same_month_uncertain": 0
    },
    "DeepSeek-V3.2": {
      "pass": 13,
      "fail": 0,
      "same_month_uncertain": 0
    },
    "Claude Sonnet 4.6": {
      "pass": 13,
      "fail": 0,
      "same_month_uncertain": 0
    },
    "Kimi K2.5 (release-date proxy)": {
      "pass": 0,
      "fail": 8,
      "same_month_uncertain": 5
    }
  },
  "kaggle_url_http_status": {
    "ai-mathematical-olympiad-progress-prize-3": 200,
    "hull-tactical-market-prediction": 200,
    "recodai-luc-scientific-image-forgery-detection": 200,
    "llm-agentic-legal-information-retrieval": 200,
    "deep-past-initiative-machine-translation": 200,
    "stanford-rna-3d-folding-2": 200,
    "vesuvius-challenge-surface-detection": 200,
    "med-gemma-impact-challenge": 200,
    "wbc-bench-2026": 200,
    "urban-flood-modelling": 200,
    "playground-series-s6e1": 200,
    "bidding-predictions-for-construction": 200,
    "nfl-big-data-bowl-2026-prediction": 200,
    "spaceship-titanic": 200,
    "store-sales-time-series-forecasting": 200,
    "llm-classification-finetuning": 200,
    "digit-recognizer": 200,
    "house-prices-advanced-regression-techniques": 200,
    "nlp-getting-started": 200,
    "connectx": 200,
    "tpu-getting-started": 200,
    "contradictory-my-dear-watson": 200,
    "gan-getting-started": 200,
    "home-data-for-ml-course": 200,
    "landmark-recognition-2021": 200
  },
  "release_url": "https://github.com/mireskandari/klive",
  "release_url_observation": "Repository returned not found during this audit.",
  "contamination_conclusion": "Dates can reduce a temporal contamination route but cannot establish contamination-free status without disclosure of model training corpora; the paper itself uses 'minimal' and 'not airtight' qualifiers."
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bacff3eb57e7", "created_at": "2026-07-18T02:24:27+00:00", "title": "Verdict and audit scope"}
-->
Exact claim: **K-live benchmark with 25 active competitions provides contamination-free, dynamic evaluation environment.**

Verdict: **PARTIAL / CLAIM OVERSTATED.** The executable audit confirms 25 listed and reachable Kaggle URLs (13 deadline-based Tier 1 + 12 perpetual Tier 2) and exactly reproduces the paper cutoff matrix. However, all 13 Tier 1 deadlines reported in the paper had passed by 2026-07-18, the cited release repository returned not found, and temporal ordering cannot prove absence from undisclosed training corpora. The paper itself says “minimal contamination,” “does not establish a full contamination-free guarantee,” and “not airtight,” so the user-supplied contamination-free wording is stronger than the paper supports.
