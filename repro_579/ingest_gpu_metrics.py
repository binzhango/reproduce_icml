#!/usr/bin/env python3
"""Ingest the completed GPU Job metrics into local Trackio for logbook publication."""

import json
from pathlib import Path

import trackio

path = Path(__file__).parent / "outputs" / "gpu_job_metrics.json"
data = json.loads(path.read_text())
trackio.init(
    project="paper579-squirrel-toy",
    name="gpu-Qwen2.5-Coder-7B-Instruct-recovered",
    config={
        "job_id": data["job_id"],
        "model": data["model"],
        "scope": "toy",
        "hardware": data["hardware"],
    },
)
for index, row in enumerate(data["task_metrics"]):
    trackio.log({
        "task_index": index,
        "gm_proxy": row["gm_proxy"],
        "modify_better_proxy": row["modify_better_proxy"],
        "prediction_parseable": row["prediction_parseable"],
        "latency_seconds": row["latency_seconds"],
    })
trackio.finish()
print(json.dumps({k: data[k] for k in ("job_url", "tasks", "all_gm_proxy_pct", "mean_latency_seconds", "running_seconds", "estimated_cost_usd")}, indent=2))
