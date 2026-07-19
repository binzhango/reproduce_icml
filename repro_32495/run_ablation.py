# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2",
#   "huggingface-hub>=0.34.0",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.5",
#   "torch>=2.4",
#   "trackio>=0.10.0",
#   "transformers>=4.48",
# ]
# ///
"""Scaled proxy ablation for ICML 2026 paper FOfvTwBGUX.

This is intentionally labelled a proxy: a small open model chooses among
standard ML pipelines on three public sklearn tasks. It tests the mechanism
and direction of iteration and fixed-role coordination, not the paper's
Kaggle-scale result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


ALGORITHMS = {
    "logreg",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "svm_rbf",
    "knn",
}

DEFAULT_PROPOSAL = {"algorithm": "logreg", "params": {"C": 1.0}, "rationale": "robust default"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output-dir", default="outputs/gpu")
    p.add_argument("--mock", action="store_true", help="Smoke-test without loading an LLM")
    p.add_argument("--trackio-space", default=os.getenv("TRACKIO_SPACE", ""))
    p.add_argument("--upload-repo", default=os.getenv("UPLOAD_REPO", ""))
    return p.parse_args()


def load_tasks() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "digits": load_digits(return_X_y=True),
    }


def dataset_summary(x: np.ndarray, y: np.ndarray) -> str:
    rows = pd.DataFrame(x[:4]).round(3).to_dict(orient="records")
    classes, counts = np.unique(y, return_counts=True)
    return json.dumps(
        {
            "n_samples": int(len(y)),
            "n_features": int(x.shape[1]),
            "class_counts": dict(zip(map(int, classes), map(int, counts))),
            "feature_missing_rate": float(np.isnan(x).mean()),
            "feature_mean_abs": float(np.mean(np.abs(x))),
            "sample_rows": rows,
            "metric": "balanced_accuracy",
        }
    )


class Proposer:
    def __init__(self, model_id: str, mock: bool):
        self.mock = mock
        self.call_index = 0
        if mock:
            self.tokenizer = self.model = None
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def ask(self, system: str, user: str, seed: int, json_only: bool = False) -> str:
        self.call_index += 1
        if self.mock:
            algorithms = ["logreg", "random_forest", "extra_trees", "hist_gradient_boosting"]
            algo = algorithms[(seed + self.call_index) % len(algorithms)]
            return json.dumps({"algorithm": algo, "params": {}, "rationale": "mock smoke test"})
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_seed = seed + 7919 * self.call_index
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=180 if json_only else 240,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)


def parse_proposal(text: str) -> tuple[dict[str, Any], bool]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return DEFAULT_PROPOSAL.copy(), False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return DEFAULT_PROPOSAL.copy(), False
    if obj.get("algorithm") not in ALGORITHMS:
        return DEFAULT_PROPOSAL.copy(), False
    if not isinstance(obj.get("params", {}), dict):
        obj["params"] = {}
    return obj, True


def clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return min(hi, max(lo, float(value)))
    except (TypeError, ValueError):
        return default


def build_model(proposal: dict[str, Any], seed: int):
    name, params = proposal["algorithm"], proposal.get("params", {})
    if name == "logreg":
        c = clamp(params.get("C"), 0.01, 100.0, 1.0)
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=c, max_iter=1500))
    if name == "random_forest":
        n = int(clamp(params.get("n_estimators"), 50, 400, 160))
        depth = int(clamp(params.get("max_depth"), 2, 40, 14))
        return make_pipeline(SimpleImputer(), RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=seed, n_jobs=-1))
    if name == "extra_trees":
        n = int(clamp(params.get("n_estimators"), 50, 400, 180))
        depth = int(clamp(params.get("max_depth"), 2, 40, 18))
        return make_pipeline(SimpleImputer(), ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=seed, n_jobs=-1))
    if name == "hist_gradient_boosting":
        lr = clamp(params.get("learning_rate"), 0.01, 0.4, 0.1)
        leaves = int(clamp(params.get("max_leaf_nodes"), 5, 63, 31))
        return make_pipeline(SimpleImputer(), HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, random_state=seed))
    if name == "svm_rbf":
        c = clamp(params.get("C"), 0.05, 100.0, 2.0)
        return make_pipeline(SimpleImputer(), StandardScaler(), SVC(C=c, gamma="scale"))
    k = int(clamp(params.get("n_neighbors"), 1, 31, 5))
    return make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def evaluate(proposal: dict[str, Any], split: tuple[np.ndarray, ...], seed: int) -> tuple[float, str]:
    x_train, x_val, y_train, y_val = split
    try:
        model = build_model(proposal, seed)
        model.fit(x_train, y_train)
        return float(balanced_accuracy_score(y_val, model.predict(x_val))), "ok"
    except Exception as exc:  # execution feedback is part of the mechanism being tested
        return 0.0, f"{type(exc).__name__}: {exc}"[:300]


BUILDER_SYSTEM = """You are an ML engineering agent choosing a sklearn pipeline.
Return exactly one JSON object with keys algorithm, params, rationale. algorithm must be one of:
logreg, random_forest, extra_trees, hist_gradient_boosting, svm_rbf, knn.
Valid params include C, n_estimators, max_depth, learning_rate, max_leaf_nodes, n_neighbors.
Optimize balanced accuracy. Do not use markdown."""


def single_agent(proposer: Proposer, summary: str, split: tuple[np.ndarray, ...], seed: int, rounds: int) -> tuple[dict, list[dict]]:
    history: list[dict] = []
    prompt = f"Dataset summary: {summary}\nPropose the first pipeline."
    best = {"score": -1.0, "proposal": DEFAULT_PROPOSAL.copy()}
    for round_id in range(1, rounds + 1):
        text = proposer.ask(BUILDER_SYSTEM, prompt, seed + round_id, json_only=True)
        proposal, valid = parse_proposal(text)
        score, feedback = evaluate(proposal, split, seed)
        row = {"round": round_id, "proposal": proposal, "score": score, "feedback": feedback, "valid_json": valid}
        history.append(row)
        if score > best["score"]:
            best = {"score": score, "proposal": proposal}
        prompt = (
            f"Dataset summary: {summary}\nPrevious trial: {json.dumps(row)}\n"
            "Use this measured execution feedback to propose a different or improved pipeline."
        )
    return best, history


def multi_agent(proposer: Proposer, summary: str, split: tuple[np.ndarray, ...], seed: int, rounds: int) -> tuple[dict, list[dict]]:
    explorer = proposer.ask(
        "You are the Explorer in a fixed-role ML team. Analyze the dataset and recommend algorithms and risks concisely.",
        f"Dataset summary: {summary}",
        seed + 101,
    )
    history: list[dict] = []
    best = {"score": -1.0, "proposal": DEFAULT_PROPOSAL.copy()}
    critique = "No prior trial."
    for round_id in range(1, rounds + 1):
        builder_prompt = (
            f"Dataset summary: {summary}\nExplorer message: {explorer}\nEvaluator message: {critique}\n"
            "Propose the next pipeline."
        )
        text = proposer.ask(BUILDER_SYSTEM, builder_prompt, seed + 200 + round_id, json_only=True)
        proposal, valid = parse_proposal(text)
        score, feedback = evaluate(proposal, split, seed)
        row = {"round": round_id, "proposal": proposal, "score": score, "feedback": feedback, "valid_json": valid}
        history.append(row)
        if score > best["score"]:
            best = {"score": score, "proposal": proposal}
        critique = proposer.ask(
            "You are the Evaluator in a fixed-role ML team. Critique the measured trial and give one concrete change.",
            f"Dataset summary: {summary}\nExplorer: {explorer}\nTrial: {json.dumps(row)}",
            seed + 300 + round_id,
        )
    return best, history


def summarize(trials: list[dict]) -> dict:
    frame = pd.DataFrame(trials)
    pivot = frame.pivot_table(index=["task", "seed"], columns="condition", values="score")
    pairs = pivot.dropna()
    iteration_delta = pairs["single_iterative"] - pairs["single_no_feedback"]
    coordination_delta = pairs["multi_agent"] - pairs["single_iterative"]
    eps = 1e-9
    return {
        "n_task_seed_pairs": int(len(pairs)),
        "mean_scores": frame.groupby("condition")["score"].mean().to_dict(),
        "iteration_gain_mean": float(iteration_delta.mean()),
        "iteration_gain_std": float(iteration_delta.std(ddof=1)),
        "coordination_delta_mean": float(coordination_delta.mean()),
        "coordination_delta_std": float(coordination_delta.std(ddof=1)),
        "multi_agent_helps": int((coordination_delta > eps).sum()),
        "multi_agent_hurts": int((coordination_delta < -eps).sum()),
        "multi_agent_ties": int((coordination_delta.abs() <= eps).sum()),
        "claim1_proxy_supported": bool(iteration_delta.mean() > abs(coordination_delta.mean())),
        "claim2_proxy_supported": bool((coordination_delta < -eps).sum() >= (coordination_delta > eps).sum()),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    proposer = Proposer(args.model, args.mock)

    import trackio

    init_kwargs: dict[str, Any] = {
        "project": "icml-32495-component-ablation",
        "name": "mock-smoke" if args.mock else "qwen-1.5b-gpu-proxy",
        "config": vars(args),
    }
    if args.trackio_space:
        init_kwargs["space_id"] = args.trackio_space
    trackio.init(**init_kwargs)

    trials: list[dict] = []
    details: list[dict] = []
    for task, (x, y) in load_tasks().items():
        summary = dataset_summary(x, y)
        for seed in range(args.seeds):
            split = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
            no_feedback, hist_one = single_agent(proposer, summary, split, seed, rounds=1)
            iterative, hist_iter = single_agent(proposer, summary, split, seed, rounds=args.rounds)
            coordinated, hist_multi = multi_agent(proposer, summary, split, seed, rounds=args.rounds)
            for condition, result in [
                ("single_no_feedback", no_feedback),
                ("single_iterative", iterative),
                ("multi_agent", coordinated),
            ]:
                row = {"task": task, "seed": seed, "condition": condition, "score": result["score"], "algorithm": result["proposal"]["algorithm"]}
                trials.append(row)
                trackio.log({"balanced_accuracy": result["score"], "task_index": list(load_tasks()).index(task), "seed": seed}, step=len(trials))
            details.append({"task": task, "seed": seed, "single_no_feedback": hist_one, "single_iterative": hist_iter, "multi_agent": hist_multi})
            print(json.dumps(trials[-3:]), flush=True)

    summary = summarize(trials)
    summary.update(
        {
            "model": "mock" if args.mock else args.model,
            "rounds": args.rounds,
            "tasks": list(load_tasks()),
            "wall_time_seconds": time.time() - started,
            "hardware": "local-mock" if args.mock else __import__("torch").cuda.get_device_name(0),
            "scope_label": "toy-smoke" if args.mock else "scaled-proxy",
        }
    )
    pd.DataFrame(trials).to_csv(output_dir / "trials.csv", index=False)
    (output_dir / "details.json").write_text(json.dumps(details, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    trackio.log({
        "iteration_gain_mean": summary["iteration_gain_mean"],
        "coordination_delta_mean": summary["coordination_delta_mean"],
        "multi_agent_helps": summary["multi_agent_helps"],
        "multi_agent_hurts": summary["multi_agent_hurts"],
    })
    trackio.finish()

    if args.upload_repo and os.getenv("HF_TOKEN"):
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.upload_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.upload_repo,
            repo_type="dataset",
            path_in_repo="gpu_proxy",
            commit_message="Add ICML 32495 scaled proxy results",
        )
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="run_ablation.py",
            repo_id=args.upload_repo,
            repo_type="dataset",
            commit_message="Add reproduction script",
        )
        print(f"UPLOADED=https://huggingface.co/datasets/{args.upload_repo}", flush=True)


if __name__ == "__main__":
    main()
