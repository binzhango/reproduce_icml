# Source & Reproducibility Audit


---
<!-- trackio-cell
{"type": "code", "id": "cell_833a24b9f7ab", "created_at": "2026-07-16T01:08:51+00:00", "title": "Run: uv audit_claims.py (exit 2)", "command": ["uv", "run", "--with", "numpy", "--with", "matplotlib", "scripts/audit_claims.py"], "exit_code": 2, "duration_s": 0.011}
-->
````bash
$ uv run --with numpy --with matplotlib scripts/audit_claims.py
````

exit 2 · 0.0s


````python title=audit_claims.py
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
    v1 = V1.read_text(errors="replace")
    v2 = V2.read_text(errors="replace")

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

````


````output
error: Failed to initialize cache at `/Users/binzhang/.cache/uv`
  Caused by: failed to open file `/Users/binzhang/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_c1736696998e", "created_at": "2026-07-16T01:09:12+00:00", "title": "Run: uv audit_claims.py (exit 0)", "command": ["uv", "run", "--with", "numpy", "--with", "matplotlib", "scripts/audit_claims.py"], "exit_code": 0, "duration_s": 9.653}
-->
````bash
$ uv run --with numpy --with matplotlib scripts/audit_claims.py
````

exit 0 · 9.7s


````python title=audit_claims.py
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
    v1 = V1.read_text(errors="replace")
    v2 = V2.read_text(errors="replace")

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

````


````output
Downloading matplotlib (8.8MiB)
Downloading fonttools (2.7MiB)
 Downloaded fonttools
 Downloaded matplotlib
Installed 11 packages in 54ms
{
  "paper": {
    "title": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
    "arxiv_id": "2505.23723",
    "openreview_id": "kcPPWaoegr",
    "v1_sha256": "fd30694dc801f789dbe735f901c49249b8712214d083e537c8563dc34eb3b811",
    "v2_sha256": "ebf1a9abcd04211b3d1ab7dc3e54b5306b639323cfcd82415c1ba7a061711a97"
  },
  "official_repo": {
    "commit": "15932e7525deb99d59f7416bbe8c75077cff3690",
    "checkpoint_released": false,
    "evaluation_code_released": false,
    "rl_code_released": false,
    "training_dataset_paths_configured": false
  },
  "claims": [
    {
      "claim": 1,
      "short": "7B ML-Agent outperforms DeepSeek-R1, GPT-4o, and GPT-4o-mini agents",
      "v1_needles": [
        "671B-sized DeepSeek-R1",
        "GPT-4o-mini",
        "GPT-4o",
        "Table 1"
      ],
      "v2_needles": [
        "DeepSeek-R1",
        "GPT-5",
        "Table 1"
      ],
      "verdict": "reported_v1_not_independently_reproduced",
      "blocker": "No ML-Agent checkpoint or runnable evaluation code is released.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 2,
      "short": "Training uses 9 tasks and evaluation uses 10 held-out multimodal tasks",
      "v1_needles": [
        "4 tasks from MLAgentBench",
        "5 from MLE-bench",
        "10 held-out"
      ],
      "v2_needles": [
        "4 tasks from MLAgentBench",
        "5 from MLE-bench",
        "10 held-out"
      ],
      "verdict": "task_split_source_verified_generalization_not_reproduced",
      "blocker": "Task membership is auditable, but trained weights and evaluation trajectories are absent.",
      "v1_source_supported": false,
      "v2_source_supported": true
    },
    {
      "claim": 3,
      "short": "CIFAR-10 average 68.88% and best 81.45%",
      "v1_needles": [
        "68.88",
        "81.45"
      ],
      "v2_needles": [
        "33.80 \u00b1 11.27"
      ],
      "verdict": "reported_v1_metric_replaced_in_v2_not_reproduced",
      "blocker": "v2 reports relative gain rather than the v1 absolute accuracy, and no checkpoint/evaluator is released.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 4,
      "short": "Generate at least 100 ideas and retain 10 behaviorally distant ideas",
      "v1_needles": [
        "at least 100 candidate ideas",
        "select the 10 most diverse ideas"
      ],
      "v2_needles": [
        "farthest-point sampling",
        "three semantic axes"
      ],
      "verdict": "reported_and_mechanism_independently_testable",
      "blocker": "Original candidate pools and embeddings are unreleased; reproduction uses a disclosed synthetic corpus.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 5,
      "short": "v1 reward is 0 for errors, 0.5 for corner cases, sigmoid-scaled for successful edits",
      "v1_needles": [
        "\u03c3(\u2212\u221e) = 0",
        "\u03c3(0) = 0.5",
        "where \u03c3(\u00b7) is the sigmoid function"
      ],
      "v2_needles": [
        "receive -1",
        "receive 0",
        "scaled metric improvement"
      ],
      "verdict": "v1_formula_verified_and_executable_v2_materially_changed",
      "blocker": "The reward implementation is unreleased; equations can nevertheless be implemented exactly.",
      "v1_source_supported": true,
      "v2_source_supported": true
    }
  ]
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_e9e1adbb9bb2", "created_at": "2026-07-16T01:09:12+00:00", "title": "Artifact: claim_matrix.csv", "path": "outputs/claim_matrix.csv", "size": 1180, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim_matrix.csv` · dataset · 1.2 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-artifacts#logbook-files/outputs/claim_matrix.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_7b3206d00e71", "created_at": "2026-07-16T01:09:42+00:00", "title": "Run: uv audit_claims.py (exit 0)", "command": ["uv", "run", "--with", "numpy", "--with", "matplotlib", "scripts/audit_claims.py"], "exit_code": 0, "duration_s": 0.048}
-->
````bash
$ uv run --with numpy --with matplotlib scripts/audit_claims.py
````

exit 0 · 0.0s


````python title=audit_claims.py
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

````


````output
{
  "paper": {
    "title": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
    "arxiv_id": "2505.23723",
    "openreview_id": "kcPPWaoegr",
    "v1_sha256": "fd30694dc801f789dbe735f901c49249b8712214d083e537c8563dc34eb3b811",
    "v2_sha256": "ebf1a9abcd04211b3d1ab7dc3e54b5306b639323cfcd82415c1ba7a061711a97"
  },
  "official_repo": {
    "commit": "15932e7525deb99d59f7416bbe8c75077cff3690",
    "checkpoint_released": false,
    "evaluation_code_released": false,
    "rl_code_released": false,
    "training_dataset_paths_configured": false
  },
  "claims": [
    {
      "claim": 1,
      "short": "7B ML-Agent outperforms DeepSeek-R1, GPT-4o, and GPT-4o-mini agents",
      "v1_needles": [
        "671B-sized DeepSeek-R1",
        "GPT-4o-mini",
        "GPT-4o",
        "Table 1"
      ],
      "v2_needles": [
        "DeepSeek-R1",
        "GPT-5",
        "Table 1"
      ],
      "verdict": "reported_v1_not_independently_reproduced",
      "blocker": "No ML-Agent checkpoint or runnable evaluation code is released.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 2,
      "short": "Training uses 9 tasks and evaluation uses 10 held-out multimodal tasks",
      "v1_needles": [
        "4 tasks from MLAgentBench",
        "5 from MLE-bench",
        "10 held-out"
      ],
      "v2_needles": [
        "4 tasks from MLAgentBench",
        "5 from MLE-bench",
        "10 held-out"
      ],
      "verdict": "task_split_source_verified_generalization_not_reproduced",
      "blocker": "Task membership is auditable, but trained weights and evaluation trajectories are absent.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 3,
      "short": "CIFAR-10 average 68.88% and best 81.45%",
      "v1_needles": [
        "68.88",
        "81.45"
      ],
      "v2_needles": [
        "33.80 \u00b1 11.27"
      ],
      "verdict": "reported_v1_metric_replaced_in_v2_not_reproduced",
      "blocker": "v2 reports relative gain rather than the v1 absolute accuracy, and no checkpoint/evaluator is released.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 4,
      "short": "Generate at least 100 ideas and retain 10 behaviorally distant ideas",
      "v1_needles": [
        "at least 100 candidate ideas",
        "select the 10 most diverse ideas"
      ],
      "v2_needles": [
        "farthest-point sampling",
        "three semantic axes"
      ],
      "verdict": "reported_and_mechanism_independently_testable",
      "blocker": "Original candidate pools and embeddings are unreleased; reproduction uses a disclosed synthetic corpus.",
      "v1_source_supported": true,
      "v2_source_supported": true
    },
    {
      "claim": 5,
      "short": "v1 reward is 0 for errors, 0.5 for corner cases, sigmoid-scaled for successful edits",
      "v1_needles": [
        "\u03c3(\u2212\u221e) = 0",
        "\u03c3(0) = 0.5",
        "where \u03c3(\u00b7) is the sigmoid function"
      ],
      "v2_needles": [
        "receive -1",
        "receive 0",
        "scaled metric improvement"
      ],
      "verdict": "v1_formula_verified_and_executable_v2_materially_changed",
      "blocker": "The reward implementation is unreleased; equations can nevertheless be implemented exactly.",
      "v1_source_supported": true,
      "v2_source_supported": true
    }
  ]
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_34e4c644eea1", "created_at": "2026-07-16T01:09:42+00:00", "title": "Artifact: claim_matrix.csv", "path": "outputs/claim_matrix.csv", "size": 1179, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim_matrix.csv` · dataset · 1.2 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-artifacts#logbook-files/outputs/claim_matrix.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_911e051da6bc", "created_at": "2026-07-16T01:12:09+00:00", "title": "Evidence policy and release audit"}
-->
Outcome: this is a partial reproduction with source verification and mechanism proxies, not a full benchmark replication. The canonical sources are arXiv v1 (https://arxiv.org/pdf/2505.23723v1), arXiv v2 (https://arxiv.org/pdf/2505.23723v2), OpenReview kcPPWaoegr (https://openreview.net/forum?id=kcPPWaoegr), and the official repository (https://github.com/MASWorks/ML-Agent). At official commit 15932e7, the checkpoint, evaluation code, RL code, and training data remain unreleased; train_step2.sh and eval.sh only print “To be released.” Therefore Claims 1–3 cannot be independently rerun. Source-supported means the paper reports the statement; reproduced means an executable independent check was completed.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d44bd98db8ae", "created_at": "2026-07-16T01:23:08+00:00", "title": "Compute and failure audit"}
-->
Four T4-small Jobs were launched while hardening the remote script: one failed before a billed running interval because dependencies were not installed; one failed after 38 running seconds because stdin execution has no __file__; one completed the experiment after 52 running seconds but failed only when a read-scoped token could not create a dataset repository; the final 48-second run completed cleanly. At USD 0.40/hour, total observed GPU-running time was 138 seconds, or about USD 0.0153; using all 172 wall-clock seconds as a conservative upper bound gives less than USD 0.02. Failed attempts are retained as provenance rather than hidden.
