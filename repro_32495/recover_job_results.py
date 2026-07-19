"""Recover raw trial rows printed before the HF Job's denied final upload."""

import csv
import json
from pathlib import Path


log_path = Path("outputs/gpu_job.log")
out_dir = Path("outputs/gpu")
out_dir.mkdir(parents=True, exist_ok=True)
trials = []
summary = None
for line in log_path.read_text(errors="replace").splitlines():
    if line.startswith("[{"):
        trials.extend(json.loads(line))
    elif line.startswith("FINAL_SUMMARY="):
        summary = json.loads(line.removeprefix("FINAL_SUMMARY="))
if len(trials) != 27 or summary is None:
    raise RuntimeError(f"Expected 27 trials and one summary, got {len(trials)} and {summary is not None}")
with (out_dir / "trials.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["task", "seed", "condition", "score", "algorithm"])
    writer.writeheader()
    writer.writerows(trials)
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
(out_dir / "provenance.json").write_text(
    json.dumps(
        {
            "job_url": "https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a",
            "script_revision": "67259e2f0261b7e2d315c9f4a12de79fd183f941",
            "recovery": "Rows and FINAL_SUMMARY parsed verbatim from immutable Job stdout after final direct upload returned 403.",
            "trial_count": len(trials),
        },
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
print(f"Recovered {len(trials)} trial rows into {out_dir}")
