# Claim 1 - Iterative feedback vs architectural complexity


---
<!-- trackio-cell
{"type": "code", "id": "cell_6b05ebaac635", "created_at": "2026-07-18T02:10:08+00:00", "title": "Run: uv run_ablation.py (exit 2)", "command": ["uv", "run", "run_ablation.py", "--mock", "--seeds", "1", "--rounds", "2", "--output-dir", "outputs/smoke"], "exit_code": 2, "duration_s": 0.037}
-->
````bash
$ uv run run_ablation.py --mock --seeds 1 --rounds 2 --output-dir outputs/smoke
````

exit 2 · 0.0s


````python title=run_ablation.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
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
        generator = torch.Generator(device=self.model.device).manual_seed(seed + 7919 * self.call_index)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=180 if json_only else 240,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                generator=generator,
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

````


````output
error: Failed to initialize cache at `/Users/binzhang/.cache/uv`
  Caused by: failed to open file `/Users/binzhang/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_37ffb4b47581", "created_at": "2026-07-18T02:10:31+00:00", "title": "Run: uv run_ablation.py (exit 2)", "command": ["uv", "run", "run_ablation.py", "--mock", "--seeds", "1", "--rounds", "2", "--output-dir", "outputs/smoke"], "exit_code": 2, "duration_s": 4.261}
-->
````bash
$ uv run run_ablation.py --mock --seeds 1 --rounds 2 --output-dir outputs/smoke
````

exit 2 · 4.3s


````python title=run_ablation.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
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
        generator = torch.Generator(device=self.model.device).manual_seed(seed + 7919 * self.call_index)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=180 if json_only else 240,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                generator=generator,
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

````


````output
WARN Retry attempt #0. Sleeping 1.458020273s before the next attempt
WARN Retry attempt #0. Sleeping 1.473489128s before the next attempt
WARN Retry attempt #0. Sleeping 1.276818695s before the next attempt
WARN Retry attempt #0. Sleeping 1.222221186s before the next attempt
WARN Retry attempt #0. Sleeping 1.025876035s before the next attempt
WARN Retry attempt #0. Sleeping 1.017180827s before the next attempt
WARN Retry attempt #0. Sleeping 1.569510147s before the next attempt
WARN Retry attempt #1. Sleeping 3.524044086s before the next attempt
WARN Retry attempt #1. Sleeping 3.104914598s before the next attempt
WARN Retry attempt #1. Sleeping 1.763576015s before the next attempt
WARN Retry attempt #1. Sleeping 3.662724803s before the next attempt
WARN Retry attempt #1. Sleeping 1.598361289s before the next attempt
WARN Retry attempt #1. Sleeping 3.402488094s before the next attempt
WARN Retry attempt #1. Sleeping 2.120895167s before the next attempt
WARN Retry attempt #2. Sleeping 4.322595307s before the next attempt
WARN Retry attempt #2. Sleeping 1.127582178s before the next attempt
WARN Retry attempt #2. Sleeping 3.921687376s before the next attempt
WARN Retry attempt #2. Sleeping 7.95935386s before the next attempt
error: Request failed after 3 retries in 4.2s
  Caused by: Failed to fetch: `https://pypi.org/simple/torch/`
  Caused by: error sending request for url (https://pypi.org/simple/torch/)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: nodename nor servname provided, or not known

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_b15eaa1553c6", "created_at": "2026-07-18T02:19:40+00:00", "title": "Run: uv run_ablation.py (exit 0)", "command": ["uv", "run", "run_ablation.py", "--mock", "--seeds", "1", "--rounds", "2", "--output-dir", "outputs/smoke"], "exit_code": 0, "duration_s": 532.915}
-->
````bash
$ uv run run_ablation.py --mock --seeds 1 --rounds 2 --output-dir outputs/smoke
````

exit 0 · 532.9s


````python title=run_ablation.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
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
        generator = torch.Generator(device=self.model.device).manual_seed(seed + 7919 * self.call_index)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=180 if json_only else 240,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                generator=generator,
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

````


````output
WARN Fixing invalid version specifier by removing stray quotes (before: `>= '2.7'`; after: `>= 2.7`)
Downloading pygments (1.2MiB)
Downloading hf-xet (3.7MiB)
Downloading tokenizers (2.8MiB)
Downloading networkx (2.0MiB)
Downloading scipy (19.5MiB)
Downloading transformers (11.1MiB)
Downloading numpy (5.1MiB)
Downloading uvloop (1.3MiB)
Downloading pandas (9.4MiB)
Downloading torch (106.1MiB)
Downloading scikit-learn (7.8MiB)
Downloading sympy (6.0MiB)
Downloading pillow (4.6MiB)
Downloading trackio (1.9MiB)
 Downloaded pygments
 Downloaded uvloop
 Downloaded trackio
 Downloaded networkx
 Downloaded tokenizers
 Downloaded hf-xet
 Downloaded pillow
 Downloaded numpy
 Downloaded sympy
 Downloaded scikit-learn
 Downloaded pandas
 Downloaded transformers
 Downloaded scipy
 Downloaded torch
Installed 55 packages in 292ms
* Trackio project initialized: icml-32495-component-ablation
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "icml-32495-component-ablation"[0m
* or by running in Python: trackio.show(project="icml-32495-component-ablation")
* Created new run: mock-smoke
[{"task": "breast_cancer", "seed": 0, "condition": "single_no_feedback", "score": 0.9297605140186915, "algorithm": "extra_trees"}, {"task": "breast_cancer", "seed": 0, "condition": "single_iterative", "score": 0.946918808411215, "algorithm": "random_forest"}, {"task": "breast_cancer", "seed": 0, "condition": "multi_agent", "score": 0.946918808411215, "algorithm": "random_forest"}]
[{"task": "wine", "seed": 0, "condition": "single_no_feedback", "score": 1.0, "algorithm": "extra_trees"}, {"task": "wine", "seed": 0, "condition": "single_iterative", "score": 1.0, "algorithm": "hist_gradient_boosting"}, {"task": "wine", "seed": 0, "condition": "multi_agent", "score": 1.0, "algorithm": "extra_trees"}]
[{"task": "digits", "seed": 0, "condition": "single_no_feedback", "score": 0.9831157731157731, "algorithm": "extra_trees"}, {"task": "digits", "seed": 0, "condition": "single_iterative", "score": 0.9815449365449366, "algorithm": "hist_gradient_boosting"}, {"task": "digits", "seed": 0, "condition": "multi_agent", "score": 0.9831157731157731, "algorithm": "extra_trees"}]
FINAL_SUMMARY={"claim1_proxy_supported": true, "claim2_proxy_supported": false, "coordination_delta_mean": 0.0005236121902788549, "coordination_delta_std": 0.0009069229170253992, "hardware": "local-mock", "iteration_gain_mean": 0.005195819273895637, "iteration_gain_std": 0.010389537530896744, "mean_scores": {"multi_agent": 0.9766781938423295, "single_iterative": 0.9761545816520506, "single_no_feedback": 0.9709587623781548}, "model": "mock", "multi_agent_helps": 1, "multi_agent_hurts": 0, "multi_agent_ties": 2, "n_task_seed_pairs": 3, "rounds": 2, "scope_label": "toy-smoke", "tasks": ["breast_cancer", "wine", "digits"], "wall_time_seconds": 14.8217191696167}
* Run finished. Uploading logs to Trackio (please wait...)

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_81b427389bf4", "created_at": "2026-07-18T02:19:40+00:00", "title": "Artifact: trials.csv", "path": "outputs/smoke/trials.csv", "size": 534, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/smoke/trials.csv` · dataset · 534 B

https://huggingface.co/buckets/binzhango/repro-investigating-component-contributions-artifacts#logbook-files/outputs/smoke/trials.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5d484b823417", "created_at": "2026-07-18T02:24:27+00:00", "title": "Claim and protocol"}
-->
Exact claim: **Iterative feedback contributes more to performance than architectural complexity in autonomous ML engineering agents.**

Planned test: compare a no-feedback single agent, a three-round iterative single agent, and a fixed-role Explorer–Builder–Evaluator pipeline. The substantive run uses Qwen/Qwen2.5-Coder-1.5B-Instruct on three sklearn classification tasks with three split/generation seeds. This is a **toy scaled proxy**, not the paper’s 100-competition, frontier-model protocol. Paper: https://openreview.net/forum?id=FOfvTwBGUX. Released benchmark URL checked: https://github.com/mireskandari/klive.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8d9e3ee0f485", "created_at": "2026-07-18T02:30:43+00:00", "title": "Paper-reported reference values"}
-->
Paper-reported context (source confirmation only, not reproduction credit): Table 6 reports C1 baseline 81.3 K-LIVE percentile, no-iteration C2 47.2 (−34.1), fixed-role multi-agent C3 72.8 (−8.5), memory C4 82.1 (+0.8), and planning C5 81.9 (+0.6). The paper’s protocol used DeepSeek-V3.2, 25 K-LIVE competitions, 10 iterations, and three replicates; the released PDF states 3,973 successful primary-ablation experiments.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a518f332047f", "created_at": "2026-07-18T02:32:47+00:00", "title": "Setup failures resolved"}
-->
Resolution of the two setup failures above: the first was a sandbox-only uv cache permission error and the second was blocked dependency networking. Both were resolved by moving `UV_CACHE_DIR` to a writable temporary path and rerunning with approved network access; successful smoke run: `cell_b15eaa1553c6` (exit 0). Neither failed run entered the measurement loop.


---
<!-- trackio-cell
{"type": "code", "id": "cell_644b03d68160", "created_at": "2026-07-18T02:40:30+00:00", "title": "Run: python3 recover_job_results.py (exit 0)", "command": ["python3", "recover_job_results.py"], "exit_code": 0, "duration_s": 0.024}
-->
````bash
$ python3 recover_job_results.py
````

exit 0 · 0.0s


````python title=recover_job_results.py
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

````


````output
{
  "claim1_proxy_supported": false,
  "claim2_proxy_supported": true,
  "coordination_delta_mean": -0.0010526894489158506,
  "coordination_delta_std": 0.0031580683467475517,
  "hardware": "Tesla T4",
  "iteration_gain_mean": 0.0007761951833213562,
  "iteration_gain_std": 0.0015503102314546717,
  "mean_scores": {
    "multi_agent": 0.970130468671021,
    "single_iterative": 0.9711831581199368,
    "single_no_feedback": 0.9704069629366155
  },
  "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "multi_agent_helps": 0,
  "multi_agent_hurts": 1,
  "multi_agent_ties": 8,
  "n_task_seed_pairs": 9,
  "rounds": 3,
  "scope_label": "scaled-proxy",
  "tasks": [
    "breast_cancer",
    "wine",
    "digits"
  ],
  "wall_time_seconds": 708.963506937027
}
Recovered 27 trial rows into outputs/gpu

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_5f7df2592d6c", "created_at": "2026-07-18T02:40:30+00:00", "title": "Artifact: trials.csv", "path": "outputs/gpu/trials.csv", "size": 1485, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/gpu/trials.csv` · dataset · 1.5 kB

https://huggingface.co/buckets/binzhango/repro-investigating-component-contributions-artifacts#logbook-files/outputs/gpu/trials.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_9673c2bf6424", "created_at": "2026-07-18T02:41:02+00:00", "title": "Run: uv ingest_gpu_results.py (exit 2)", "command": ["uv", "run", "ingest_gpu_results.py", "--results", "outputs/gpu", "--job-url", "https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a"], "exit_code": 2, "duration_s": 4.028}
-->
````bash
$ uv run ingest_gpu_results.py --results outputs/gpu --job-url https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a
````

exit 2 · 4.0s


````python title=ingest_gpu_results.py
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

````


````output
WARN Retry attempt #0. Sleeping 1.975546497s before the next attempt
WARN Retry attempt #0. Sleeping 1.020501525s before the next attempt
WARN Retry attempt #1. Sleeping 1.113165463s before the next attempt
WARN Retry attempt #1. Sleeping 1.271761132s before the next attempt
WARN Retry attempt #2. Sleeping 1.86135401s before the next attempt
WARN Retry attempt #2. Sleeping 1.071366938s before the next attempt
error: Request failed after 3 retries in 4.0s
  Caused by: Failed to fetch: `https://pypi.org/simple/pandas/`
  Caused by: error sending request for url (https://pypi.org/simple/pandas/)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: nodename nor servname provided, or not known

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_1797bec7bf10", "created_at": "2026-07-18T02:41:22+00:00", "title": "Run: uv ingest_gpu_results.py (exit 0)", "command": ["uv", "run", "ingest_gpu_results.py", "--results", "outputs/gpu", "--job-url", "https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a"], "exit_code": 0, "duration_s": 8.283}
-->
````bash
$ uv run ingest_gpu_results.py --results outputs/gpu --job-url https://huggingface.co/jobs/binzhango/6a5ae418bee6ee1cf4ece59a
````

exit 0 · 8.3s


````python title=ingest_gpu_results.py
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

````


````output
Installed 32 packages in 136ms
* Trackio project initialized: icml-32495-component-ablation
* Trackio metrics logged to: /Users/binzhang/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "icml-32495-component-ablation"[0m
* or by running in Python: trackio.show(project="icml-32495-component-ablation")
* Created new run: qwen-1.5b-gpu-proxy-imported
* Run finished. Uploading logs to Trackio (please wait...)
{
  "claim1_proxy_supported": false,
  "claim2_proxy_supported": true,
  "coordination_delta_mean": -0.0010526894489158506,
  "coordination_delta_std": 0.0031580683467475517,
  "hardware": "Tesla T4",
  "iteration_gain_mean": 0.0007761951833213562,
  "iteration_gain_std": 0.0015503102314546717,
  "mean_scores": {
    "multi_agent": 0.970130468671021,
    "single_iterative": 0.9711831581199368,
    "single_no_feedback": 0.9704069629366155
  },
  "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "multi_agent_helps": 0,
  "multi_agent_hurts": 1,
  "multi_agent_ties": 8,
  "n_task_seed_pairs": 9,
  "rounds": 3,
  "scope_label": "scaled-proxy",
  "tasks": [
    "breast_cancer",
    "wine",
    "digits"
  ],
  "wall_time_seconds": 708.963506937027
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_0c0db8c10bbf", "created_at": "2026-07-18T02:43:58+00:00", "title": "Run: uv make_results_figure.py (exit 0)", "command": ["uv", "run", "make_results_figure.py", "--results", "outputs/gpu", "--html", "outputs/results_figure.html", "--svg", "outputs/results_chart.svg"], "exit_code": 0, "duration_s": 5.78}
-->
````bash
$ uv run make_results_figure.py --results outputs/gpu --html outputs/results_figure.html --svg outputs/results_chart.svg
````

exit 0 · 5.8s


````python title=make_results_figure.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""Create a self-contained HTML/SVG summary from raw GPU trials."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd


COLORS = {"single_no_feedback": "#94a3b8", "single_iterative": "#2563eb", "multi_agent": "#d97706"}
LABELS = {"single_no_feedback": "No feedback", "single_iterative": "Iterative single", "multi_agent": "Fixed-role multi"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/gpu")
    p.add_argument("--html", default="outputs/results_figure.html")
    p.add_argument("--svg", default="outputs/results_chart.svg")
    args = p.parse_args()
    root = Path(args.results)
    trials = pd.read_csv(root / "trials.csv")
    summary = json.loads((root / "summary.json").read_text())
    means = trials.groupby("condition")["score"].mean()
    pivot = trials.pivot(index=["task", "seed"], columns="condition", values="score")
    deltas = (pivot["multi_agent"] - pivot["single_iterative"]).tolist()

    w, h = 1100, 620
    bars = []
    for i, condition in enumerate(["single_no_feedback", "single_iterative", "multi_agent"]):
        value = float(means[condition])
        x = 90 + i * 255
        bar_h = value * 400
        y = 500 - bar_h
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="160" height="{bar_h:.1f}" rx="10" fill="{COLORS[condition]}"/>'
            f'<text x="{x + 80}" y="{y - 14:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            f'<text x="{x + 80}" y="535" text-anchor="middle" class="label">{html.escape(LABELS[condition])}</text>'
        )
    dots = []
    for i, delta in enumerate(deltas):
        cx = 850 + (i % 3) * 65
        cy = 300 - delta * 1700
        color = "#dc2626" if delta < 0 else "#16a34a" if delta > 0 else "#64748b"
        dots.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="10" fill="{color}" opacity="0.85"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<style>.title{{font:700 30px system-ui;fill:#0f172a}}.value{{font:700 22px system-ui;fill:#0f172a}}.label{{font:600 18px system-ui;fill:#334155}}.note{{font:500 18px system-ui;fill:#475569}}</style>
<rect width="100%" height="100%" fill="#ffffff"/><text x="50" y="48" class="title">Scaled proxy: balanced accuracy and coordination deltas</text>
<line x1="55" y1="500" x2="790" y2="500" stroke="#94a3b8" stroke-width="2"/>{''.join(bars)}
<text x="835" y="120" class="title">Multi − single</text><line x1="825" y1="300" x2="1050" y2="300" stroke="#64748b" stroke-width="2"/>{''.join(dots)}
<text x="835" y="480" class="note">helps {summary['multi_agent_helps']} · hurts {summary['multi_agent_hurts']} · ties {summary['multi_agent_ties']}</text>
<text x="55" y="585" class="note">Iteration gain {summary['iteration_gain_mean']:+.4f}; coordination mean {summary['coordination_delta_mean']:+.4f}. Qwen2.5-Coder-1.5B, 3 tasks × 3 seeds.</text>
</svg>'''
    svg_path = Path(args.svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg)
    Path(args.html).write_text(f'<!doctype html><meta charset="utf-8"><title>ICML 32495 proxy results</title><style>body{{margin:0;background:#f8fafc;display:grid;place-items:center;min-height:100vh}}svg{{max-width:100%;height:auto}}</style>{svg}')
    print(f"Wrote {args.html} and {args.svg}")


if __name__ == "__main__":
    main()

````


````output
Installed 4 packages in 18ms
Wrote outputs/results_figure.html and outputs/results_chart.svg

````


---
<!-- trackio-cell
{"type": "figure", "id": "cell_f12d88d388b6", "created_at": "2026-07-18T02:43:59+00:00", "title": "GPU proxy condition means and paired coordination deltas"}
-->
````html
<!doctype html><meta charset="utf-8"><title>ICML 32495 proxy results</title><style>body{margin:0;background:#f8fafc;display:grid;place-items:center;min-height:100vh}svg{max-width:100%;height:auto}</style><svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">
<style>.title{font:700 30px system-ui;fill:#0f172a}.value{font:700 22px system-ui;fill:#0f172a}.label{font:600 18px system-ui;fill:#334155}.note{font:500 18px system-ui;fill:#475569}</style>
<rect width="100%" height="100%" fill="#ffffff"/><text x="50" y="48" class="title">Scaled proxy: balanced accuracy and coordination deltas</text>
<line x1="55" y1="500" x2="790" y2="500" stroke="#94a3b8" stroke-width="2"/><rect x="90" y="111.8" width="160" height="388.2" rx="10" fill="#94a3b8"/><text x="170" y="97.8" text-anchor="middle" class="value">0.970</text><text x="170" y="535" text-anchor="middle" class="label">No feedback</text><rect x="345" y="111.5" width="160" height="388.5" rx="10" fill="#2563eb"/><text x="425" y="97.5" text-anchor="middle" class="value">0.971</text><text x="425" y="535" text-anchor="middle" class="label">Iterative single</text><rect x="600" y="111.9" width="160" height="388.1" rx="10" fill="#d97706"/><text x="680" y="97.9" text-anchor="middle" class="value">0.970</text><text x="680" y="535" text-anchor="middle" class="label">Fixed-role multi</text>
<text x="835" y="120" class="title">Multi − single</text><line x1="825" y1="300" x2="1050" y2="300" stroke="#64748b" stroke-width="2"/><circle cx="850" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="915" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="980" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="850" cy="316.1" r="10" fill="#dc2626" opacity="0.85"/><circle cx="915" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="980" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="850" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="915" cy="300.0" r="10" fill="#64748b" opacity="0.85"/><circle cx="980" cy="300.0" r="10" fill="#64748b" opacity="0.85"/>
<text x="835" y="480" class="note">helps 0 · hurts 1 · ties 8</text>
<text x="55" y="585" class="note">Iteration gain +0.0008; coordination mean -0.0011. Qwen2.5-Coder-1.5B, 3 tasks × 3 seeds.</text>
</svg>
````

````raw
task,seed,condition,score,algorithm
breast_cancer,0,single_no_feedback,0.9578709112149533,logreg
breast_cancer,0,single_iterative,0.9578709112149533,logreg
breast_cancer,0,multi_agent,0.9578709112149533,logreg
breast_cancer,1,single_no_feedback,0.9515917056074766,logreg
breast_cancer,1,single_iterative,0.954731308411215,hist_gradient_boosting
breast_cancer,1,multi_agent,0.954731308411215,random_forest
breast_cancer,2,single_no_feedback,0.9562646028037383,logreg
breast_cancer,2,single_iterative,0.9562646028037383,logreg
breast_cancer,2,multi_agent,0.9562646028037383,logreg
wine,0,single_no_feedback,1.0,logreg
wine,0,single_iterative,1.0,logreg
wine,0,multi_agent,1.0,logreg
wine,1,single_no_feedback,0.9841269841269842,logreg
wine,1,single_iterative,0.9841269841269842,logreg
wine,1,multi_agent,0.9841269841269842,logreg
wine,2,single_no_feedback,0.9841269841269842,logreg
wine,2,single_iterative,0.9841269841269842,logreg
wine,2,multi_agent,0.9841269841269842,logreg
digits,0,single_no_feedback,0.9776987826987827,random_forest
digits,0,single_iterative,0.9815449365449366,hist_gradient_boosting
digits,0,multi_agent,0.9720707315046939,logreg
digits,1,single_no_feedback,0.9664776209115832,logreg
digits,1,single_iterative,0.9664776209115832,logreg
digits,1,multi_agent,0.9664776209115832,logreg
digits,2,single_no_feedback,0.9555050749390371,logreg
digits,2,single_iterative,0.9555050749390371,logreg
digits,2,multi_agent,0.9555050749390371,logreg

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6fd337796f90", "created_at": "2026-07-18T02:45:07+00:00", "title": "Claim 1 verdict and measured evidence"}
-->
Verdict: **INCONCLUSIVE / PROXY DOES NOT SUPPORT THE CLAIM.** On nine paired task/seed cases, no-feedback mean balanced accuracy was 0.970407 and the three-round iterative single agent was 0.971183: paired gain **+0.000776 ± 0.001550 SD**. The fixed-role architecture changed performance by **−0.001053 ± 0.003158 SD** versus the iterative single agent, a larger absolute mean effect than iteration in this toy, saturated small-task proxy. This does not falsify the paper’s Kaggle/frontier-model claim; it shows that the claimed dominance did not transfer to this reduced proposal-selection setup.

Clean completed rerun: https://huggingface.co/jobs/binzhango/6a5b47cfd216bd6f3a1fdea9 on 1× NVIDIA T4 (16 GB), Qwen/Qwen2.5-Coder-1.5B-Instruct, 3 tasks × 3 seeds × 3 conditions, 711.62 s measurement wall time. Raw rows and revision-pinned script: https://huggingface.co/datasets/binzhango/icml-32495-reproduction-artifacts. Dashboard: https://huggingface.co/spaces/binzhango/icml-32495-trackio.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3cdb180d0a51", "created_at": "2026-07-18T02:45:07+00:00", "title": "GPU setup and upload failures resolved"}
-->
Job provenance note: two setup Jobs failed before measurement (`6a5ae2f6bee6ee1cf4ece592`: missing accelerate; `6a5ae3a6bee6ee1cf4ece596`: Transformers generator API). The original measurement Job (`6a5ae418bee6ee1cf4ece59a`) completed all 27 measurements and printed `FINAL_SUMMARY`, then received a 403 only during its terminal direct upload. Its stdout was recovered and validated with an exact 27-row assertion. To remove any ambiguity from that terminal ERROR status, Job `6a5b47cfd216bd6f3a1fdea9` reran the exact revision-pinned script with no Hub write or upload step and finished **COMPLETED** with `failureCount: 0`. Its 27 rows and `FINAL_SUMMARY` reproduce the published results exactly.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_clean_gpu_rerun_32495", "created_at": "2026-07-18T09:46:00+00:00", "title": "Clean completed GPU rerun"}
-->
Canonical Job: https://huggingface.co/jobs/binzhango/6a5b47cfd216bd6f3a1fdea9 — **COMPLETED**, `failureCount: 0`, NVIDIA T4, 745 s Job runtime and 711.62 s measurement wall time. It executed the immutable script revision `67259e2f0261b7e2d315c9f4a12de79fd183f941` without a Hub token or terminal upload. The emitted `FINAL_SUMMARY` exactly matches the published means and deltas: iteration `+0.0007761952`, coordination `−0.0010526894`, and help/hurt/tie counts `0/1/8` across nine paired cases.
