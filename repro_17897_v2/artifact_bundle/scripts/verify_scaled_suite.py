# /// script
# requires-python = ">=3.11"
# ///
"""Fail-closed verification for the scaled ML-Agent reproduction."""

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
    tasks = data["task_manifest"]
    diversity = data["diversity"]
    generalization = data["generalization"]
    reward = data["reward"]

    require(protocol["training_tasks"] == 9, "exactly 9 training tasks")
    require(protocol["heldout_tasks"] == 10, "exactly 10 held-out tasks")
    require(protocol["candidate_actions"] >= 100, "at least 100 candidate actions")
    require(protocol["selected_actions"] == 10, "exactly 10 selected actions")
    require(len(diversity["selected_ids"]) == 10, "selected-id manifest is complete")
    require(len(set(diversity["selected_ids"])) == 10, "selected actions are unique")
    require(sum(task["split"] == "train" for task in tasks) == 9, "task manifest has 9 training rows")
    require(sum(task["split"] == "heldout" for task in tasks) == 10, "task manifest has 10 held-out rows")
    heldout_modalities = {task["modality"] for task in tasks if task["split"] == "heldout"}
    require(heldout_modalities == {"image", "tabular", "text"}, "held-out tasks span image, tabular, and text")
    require(diversity["random_trials"] == 5000, "5,000 random diversity controls")
    require(math.isfinite(diversity["selected_mean_pairwise_distance"]), "selected diversity is finite")
    require(diversity["selected_mean_pairwise_distance"] > diversity["random_mean"], "selected set is more diverse than random mean")
    require(diversity["selected_mean_pairwise_distance"] > diversity["random_p95"], "selected set exceeds random 95th percentile")
    require(generalization["training_tasks"] == 9 and generalization["heldout_tasks"] == 10, "ranker train/test counts are exact")
    require(generalization["modalities"] == ["image", "tabular", "text"], "ranker evaluation covers all modalities")
    require(0.0 <= generalization["agent_mean"] <= 1.0, "agent accuracy is bounded")
    require(generalization["agent_mean"] > generalization["random_mean"], "trained ranker beats random selected actions on held-out tasks")
    require(reward["error_rewards"] == [0.0], "invalid/error trajectories receive reward 0")
    require(reward["neutral_rewards"] == [0.5], "non-edit/corner outcomes receive reward 0.5")
    require(reward["successful_edits"] == 10, "sigmoid reward evaluated on all held-out edits")
    require(0.0 <= reward["successful_edit_min_reward"] <= reward["successful_edit_max_reward"] <= 1.0, "success rewards are sigmoid bounded")
    require(reward["actual_error_feedback"].startswith("ValueError:"), "error branch comes from an executed invalid action")
    require(not protocol["offline_text_smoke"], "evidence run uses real 20 Newsgroups text, not smoke corpus")
    print("ALL_VERIFICATIONS_PASS")


if __name__ == "__main__":
    main()
