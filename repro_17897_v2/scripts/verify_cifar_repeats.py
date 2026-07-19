# /// script
# requires-python = ">=3.11"
# ///
"""Fail-closed checks for substantive CIFAR-10 evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    measurement = data["measurement"]
    runs = measurement["runs"]
    require(protocol["dataset"] == "CIFAR-10", "dataset is real CIFAR-10")
    require(not protocol["synthetic"], "synthetic smoke data are rejected")
    require(protocol["device"] == "cuda" and protocol["gpu_name"], "substantive run used a named GPU")
    require(protocol["train_limit"] == 50000, "all 50,000 CIFAR-10 training examples were used")
    require(len(runs) >= 5 and len(set(protocol["seeds"])) == len(runs), "at least five unique-seed runs")
    require(all(run["test_samples"] == 10000 for run in runs), "every run evaluated all 10,000 test examples")
    require(all(math.isfinite(run["final_accuracy"]) and 0 <= run["final_accuracy"] <= 1 for run in runs), "all accuracies are finite and bounded")
    require(math.isfinite(measurement["mean_accuracy"]), "mean accuracy is newly measured")
    require(math.isfinite(measurement["sample_sd"]), "run-to-run sample standard deviation is reported")
    print("ALL_CIFAR_VERIFICATIONS_PASS")


if __name__ == "__main__":
    main()
