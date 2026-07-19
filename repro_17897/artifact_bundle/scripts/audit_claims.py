#!/usr/bin/env python3
"""Audit ML-Agent claims against arXiv v1/v2 and the released repository."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "source" / "ml_agent_arxiv_2505.23723v1.txt"
V2 = ROOT / "source" / "ml_agent_arxiv_2505.23723v2.txt"
REPO = ROOT / "official_repo"
OUT = ROOT / "outputs"


CLAIMS = [
    {
        "claim": 1,
        "short": "7B ML-Agent outperforms DeepSeek-R1, GPT-4o, and GPT-4o-mini agents",
        "v1_needles": ["671B-sized DeepSeek-R1", "GPT-4o-mini", "GPT-4o", "Table 1"],
        "v2_needles": ["DeepSeek-R1", "GPT-5", "Table 1"],
        "verdict": "reported_v1_not_independently_reproduced",
        "blocker": "No ML-Agent checkpoint or runnable evaluation code is released.",
    },
    {
        "claim": 2,
        "short": "Training uses 9 tasks and evaluation uses 10 held-out multimodal tasks",
        "v1_needles": ["4 tasks from MLAgentBench", "5 from MLE-bench", "10 held-out"],
        "v2_needles": ["4 tasks from MLAgentBench", "5 from MLE-bench", "10 held-out"],
        "verdict": "task_split_source_verified_generalization_not_reproduced",
        "blocker": "Task membership is auditable, but trained weights and evaluation trajectories are absent.",
    },
    {
        "claim": 3,
        "short": "CIFAR-10 average 68.88% and best 81.45%",
        "v1_needles": ["68.88", "81.45"],
        "v2_needles": ["33.80 ± 11.27"],
        "verdict": "reported_v1_metric_replaced_in_v2_not_reproduced",
        "blocker": "v2 reports relative gain rather than the v1 absolute accuracy, and no checkpoint/evaluator is released.",
    },
    {
        "claim": 4,
        "short": "Generate at least 100 ideas and retain 10 behaviorally distant ideas",
        "v1_needles": ["at least 100 candidate ideas", "select the 10 most diverse ideas"],
        "v2_needles": ["farthest-point sampling", "three semantic axes"],
        "verdict": "reported_and_mechanism_independently_testable",
        "blocker": "Original candidate pools and embeddings are unreleased; reproduction uses a disclosed synthetic corpus.",
    },
    {
        "claim": 5,
        "short": "v1 reward is 0 for errors, 0.5 for corner cases, sigmoid-scaled for successful edits",
        "v1_needles": ["σ(−∞) = 0", "σ(0) = 0.5", "where σ(·) is the sigmoid function"],
        "v2_needles": ["receive -1", "receive 0", "scaled metric improvement"],
        "verdict": "v1_formula_verified_and_executable_v2_materially_changed",
        "blocker": "The reward implementation is unreleased; equations can nevertheless be implemented exactly.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = " ".join(V1.read_text(errors="replace").split())
    v2 = " ".join(V2.read_text(errors="replace").split())

    repo_checks = {
        "commit": "15932e7525deb99d59f7416bbe8c75077cff3690",
        "checkpoint_released": False,
        "evaluation_code_released": (REPO / "scripts" / "eval.sh").read_text().strip() != 'echo "To be released."',
        "rl_code_released": (REPO / "scripts" / "train_step2.sh").read_text().strip() != 'echo "To be released."',
        "training_dataset_paths_configured": 'dataset_dict=(\n    ["data_9task"]=""\n)' not in (REPO / "scripts" / "train_step1.sh").read_text(),
    }

    rows = []
    for item in CLAIMS:
        row = {
            **item,
            "v1_source_supported": all(needle in v1 for needle in item["v1_needles"]),
            "v2_source_supported": all(needle in v2 for needle in item["v2_needles"]),
        }
        rows.append(row)

    audit = {
        "paper": {
            "title": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
            "arxiv_id": "2505.23723",
            "openreview_id": "kcPPWaoegr",
            "v1_sha256": sha256(ROOT / "source" / "ml_agent_arxiv_2505.23723v1.pdf"),
            "v2_sha256": sha256(ROOT / "source" / "ml_agent_arxiv_2505.23723v2.pdf"),
        },
        "official_repo": repo_checks,
        "claims": rows,
    }
    (OUT / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    with (OUT / "claim_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["claim", "short", "v1_source_supported", "v2_source_supported", "verdict", "blocker"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
