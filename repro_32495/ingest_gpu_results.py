# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "trackio>=0.10.0"]
# ///
"""Import immutable GPU Job outputs into the local Trackio campaign."""

import argparse
import json
from pathlib import Path

import pandas as pd
import trackio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/gpu")
    p.add_argument("--job-url", required=True)
    args = p.parse_args()
    root = Path(args.results)
    trials = pd.read_csv(root / "trials.csv")
    summary = json.loads((root / "summary.json").read_text())
    run = trackio.init(
        project="icml-32495-component-ablation",
        name="qwen-1.5b-gpu-proxy-imported",
        config={"job_url": args.job_url, "source": "immutable HF Job outputs", **summary},
    )
    condition_id = {name: idx for idx, name in enumerate(sorted(trials["condition"].unique()))}
    task_id = {name: idx for idx, name in enumerate(sorted(trials["task"].unique()))}
    for step, row in trials.iterrows():
        trackio.log(
            {
                "balanced_accuracy": float(row["score"]),
                "condition_id": condition_id[row["condition"]],
                "task_id": task_id[row["task"]],
                "seed": int(row["seed"]),
            },
            step=int(step),
        )
    trackio.log(
        {
            "iteration_gain_mean": summary["iteration_gain_mean"],
            "coordination_delta_mean": summary["coordination_delta_mean"],
            "multi_agent_helps": summary["multi_agent_helps"],
            "multi_agent_hurts": summary["multi_agent_hurts"],
        }
    )
    trackio.log_artifact(root, name="gpu-proxy-raw-results", type="dataset", aliases=["gpu", "latest"])
    trackio.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
