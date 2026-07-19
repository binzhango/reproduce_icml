# Claim 4: Behaviorally Distant Exploration


---
<!-- trackio-cell
{"type": "code", "id": "cell_5ab05527c4a0", "created_at": "2026-07-17T19:35:34+00:00", "title": "Corrected maximin selection over the preserved 2,280-fit matrix", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "scripts/run_scaled_suite.py", "--output-dir", "outputs/scaled_suite_corrected", "--data-home", "/tmp/ml-agent-20news", "--reuse-matrix", "outputs/scaled_suite/execution_matrix.csv"], "exit_code": 134, "duration_s": 16.181}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_scaled_suite.py --output-dir outputs/scaled_suite_corrected --data-home /tmp/ml-agent-20news --reuse-matrix outputs/scaled_suite/execution_matrix.csv
````

exit 134 · 16.2s


````python title=run_scaled_suite.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.5",
# ]
# ///
"""Scaled independent reproduction of ML-Agent claims 2, 4, and 5.

The experiment uses 120 executable pipeline actions, exactly nine training
tasks, and ten held-out tasks spanning tabular, image, and text modalities.
Candidate diversity is computed from measured validation-score behavior, not
from idea text. A task-conditioned ranker is trained only on the nine training
tasks and evaluated on all held-out tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC


SEED = 17897
TRAIN_TASKS = 9
HELDOUT_TASKS = 10
CANDIDATES = 120
SELECTED = 10


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    strength: float
    balanced: bool
    variant: str


@dataclass
class Task:
    name: str
    split: str
    modality: str
    X: Any
    y: np.ndarray
    sparse: bool = False


def candidate_pool() -> list[Candidate]:
    pool: list[Candidate] = []
    strengths = np.logspace(-3, 3, 20)
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"logreg-c{value:.5g}-b{int(balanced)}", "logreg", float(value), balanced, "l2"))
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"linearsvc-c{value:.5g}-b{int(balanced)}", "linearsvc", float(value), balanced, "hinge"))
    alphas = np.logspace(-7, -2, 10)
    for loss in ("hinge", "log_loss"):
        for alpha in alphas:
            pool.append(Candidate(f"sgd-{loss}-a{alpha:.3g}", "sgd", float(alpha), True, loss))
    for weights in ("uniform", "distance"):
        for neighbors in (1, 3, 5, 7, 9, 11, 15, 21, 31, 45):
            pool.append(Candidate(f"knn-k{neighbors}-{weights}", "knn", float(neighbors), False, weights))
    assert len(pool) == CANDIDATES
    return pool


def estimator(candidate: Candidate, sparse: bool, seed: int):
    class_weight = "balanced" if candidate.balanced else None
    if candidate.family == "logreg":
        model = LogisticRegression(
            C=candidate.strength,
            class_weight=class_weight,
            max_iter=500,
            solver="liblinear" if not sparse else "saga",
            random_state=seed,
        )
    elif candidate.family == "linearsvc":
        model = LinearSVC(C=candidate.strength, class_weight=class_weight, random_state=seed, max_iter=5000)
    elif candidate.family == "sgd":
        model = SGDClassifier(
            loss=candidate.variant,
            alpha=candidate.strength,
            class_weight=class_weight,
            random_state=seed,
            max_iter=2000,
            tol=1e-4,
        )
    elif candidate.family == "knn":
        model = KNeighborsClassifier(n_neighbors=int(candidate.strength), weights=candidate.variant, n_jobs=1)
    else:
        raise ValueError(candidate.family)
    scaler = MaxAbsScaler() if sparse else StandardScaler()
    return make_pipeline(scaler, model)


def stratified_cap(X, y: np.ndarray, cap: int, seed: int):
    if len(y) <= cap:
        return X, y
    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=seed)
    return X_keep, y_keep


def numeric_task(name: str, split: str, modality: str, X, y, transform=None) -> Task:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if transform is not None:
        X, y = transform(X, y)
    X, y = stratified_cap(X, y, 1400, SEED)
    return Task(name, split, modality, X, y, sparse=False)


def load_tasks(offline_text: bool = False, data_home: Path | None = None) -> list[Task]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    diabetes = load_diabetes()
    digits = load_digits()

    rng = np.random.default_rng(SEED)

    tasks = [
        numeric_task("train-iris-multiclass", "train", "tabular", iris.data, iris.target),
        numeric_task("train-wine-multiclass", "train", "tabular", wine.data, wine.target),
        numeric_task("train-breast-cancer", "train", "tabular", cancer.data, cancer.target),
        numeric_task("train-digits-multiclass", "train", "image", digits.data, digits.target),
        numeric_task("train-digits-parity", "train", "image", digits.data, digits.target % 2),
        numeric_task("train-digits-low-high", "train", "image", digits.data, (digits.target >= 5).astype(int)),
    ]

    def half_features(X, y):
        return X[:, ::2], y

    def wine_zero(X, y):
        return X, (y == 0).astype(int)

    def iris_setosa(X, y):
        return X, (y == 0).astype(int)

    def noisy_digits(X, y):
        noise = rng.normal(0, 2.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y

    def rolled_digits(X, y):
        images = X.reshape(-1, 8, 8)
        rolled = np.roll(images, shift=1, axis=2)
        return rolled.reshape(-1, 64), y

    def noisy_parity(X, y):
        noise = rng.normal(0, 3.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y % 2

    tasks.extend(
        [
            numeric_task("heldout-iris-setosa", "heldout", "tabular", iris.data, iris.target, iris_setosa),
            numeric_task("heldout-wine-class-zero", "heldout", "tabular", wine.data, wine.target, wine_zero),
            numeric_task("heldout-cancer-half-features", "heldout", "tabular", cancer.data, cancer.target, half_features),
            numeric_task(
                "heldout-diabetes-above-median",
                "heldout",
                "tabular",
                diabetes.data,
                (diabetes.target >= np.median(diabetes.target)).astype(int),
            ),
            numeric_task("heldout-digits-noisy", "heldout", "image", digits.data, digits.target, noisy_digits),
            numeric_task("heldout-digits-shifted", "heldout", "image", digits.data, digits.target, rolled_digits),
            numeric_task("heldout-digits-noisy-parity", "heldout", "image", digits.data, digits.target, noisy_parity),
        ]
    )

    text_specs = [
        ("train-text-comp-vs-space", "train", ["comp.graphics", "sci.space"]),
        ("train-text-autos-vs-motorcycles", "train", ["rec.autos", "rec.motorcycles"]),
        ("train-text-politics", "train", ["talk.politics.guns", "talk.politics.misc"]),
        ("heldout-text-religion", "heldout", ["alt.atheism", "soc.religion.christian"]),
        ("heldout-text-computer-systems", "heldout", ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]),
        ("heldout-text-science", "heldout", ["sci.electronics", "sci.med"]),
    ]
    if offline_text:
        # Deterministic smoke corpus; full evidence runs must omit --offline-text.
        seeds = {
            "comp": ["graphics card pixel render", "space orbit rocket nasa"],
            "autos": ["engine wheel road car", "motorcycle helmet bike ride"],
            "politics": ["firearm law rights", "election policy debate"],
            "religion": ["atheist secular reason", "church christian faith"],
            "systems": ["ibm pc motherboard", "mac apple hardware"],
            "science": ["circuit voltage electronics", "medical patient treatment"],
        }
        for index, (name, split, _) in enumerate(text_specs):
            pair = list(seeds.values())[index]
            texts, labels = [], []
            for label, base in enumerate(pair):
                for j in range(80):
                    texts.append(f"{base} document topic {j % 11}")
                    labels.append(label)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=1)
            X = vectorizer.fit_transform(texts)
            tasks.append(Task(name, split, "text", X, np.asarray(labels), sparse=True))
    else:
        for name, split, categories in text_specs:
            data = fetch_20newsgroups(
                subset="all",
                categories=categories,
                remove=("headers", "footers", "quotes"),
                data_home=str(data_home) if data_home is not None else None,
            )
            X_text, y = stratified_cap(np.asarray(data.data, dtype=object), np.asarray(data.target), 1400, SEED)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, min_df=2, sublinear_tf=True)
            X = vectorizer.fit_transform(X_text.tolist())
            tasks.append(Task(name, split, "text", X, y, sparse=True))

    tasks.sort(key=lambda task: (task.split != "train", task.name))
    assert sum(task.split == "train" for task in tasks) == TRAIN_TASKS
    assert sum(task.split == "heldout" for task in tasks) == HELDOUT_TASKS
    assert {task.modality for task in tasks if task.split == "heldout"} == {"tabular", "image", "text"}
    return tasks


def task_features(task: Task) -> dict[str, float]:
    y = np.asarray(task.y)
    _, counts = np.unique(y, return_counts=True)
    n_samples, n_features = task.X.shape
    return {
        "log_samples": math.log1p(n_samples),
        "log_features": math.log1p(n_features),
        "classes": float(len(counts)),
        "imbalance": float(counts.max() / counts.sum()),
        "sparse": float(task.sparse),
        "mod_tabular": float(task.modality == "tabular"),
        "mod_image": float(task.modality == "image"),
        "mod_text": float(task.modality == "text"),
    }


def candidate_features(candidate: Candidate) -> dict[str, float]:
    return {
        "fam_logreg": float(candidate.family == "logreg"),
        "fam_linearsvc": float(candidate.family == "linearsvc"),
        "fam_sgd": float(candidate.family == "sgd"),
        "fam_knn": float(candidate.family == "knn"),
        "strength": math.log1p(candidate.strength) if candidate.family == "knn" else math.log10(candidate.strength),
        "balanced": float(candidate.balanced),
        "variant_distance": float(candidate.variant == "distance"),
        "variant_log": float(candidate.variant == "log_loss"),
    }


def evaluate(tasks: list[Task], candidates: list[Candidate], output_dir: Path) -> pd.DataFrame:
    rows = []
    total = len(tasks) * len(candidates)
    done = 0
    for task in tasks:
        X_train, X_valid, y_train, y_valid = train_test_split(
            task.X,
            task.y,
            test_size=0.30,
            random_state=SEED,
            stratify=task.y,
        )
        for candidate in candidates:
            started = time.perf_counter()
            status = "success"
            error = ""
            score = float("nan")
            try:
                model = estimator(candidate, task.sparse, SEED)
                model.fit(X_train, y_train)
                score = float(accuracy_score(y_valid, model.predict(X_valid)))
            except Exception as exc:  # preserved as real execution feedback
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            row = {
                "task": task.name,
                "split": task.split,
                "modality": task.modality,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "score": score,
                "status": status,
                "error": error,
                "fit_seconds": elapsed,
            }
            row.update({f"task_{key}": value for key, value in task_features(task).items()})
            row.update({f"cand_{key}": value for key, value in candidate_features(candidate).items()})
            rows.append(row)
            done += 1
            if done % 120 == 0:
                print(f"PROGRESS={done}/{total} last_task={task.name}", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "execution_matrix.partial.csv", index=False)
    return pd.DataFrame(rows)


def select_diverse(matrix: pd.DataFrame, candidates: list[Candidate]) -> tuple[list[str], dict[str, Any]]:
    train = matrix[(matrix["split"] == "train") & (matrix["status"] == "success")]
    pivot = train.pivot(index="candidate_id", columns="task", values="score").reindex([c.candidate_id for c in candidates])
    if pivot.isna().any().any():
        pivot = pivot.fillna(pivot.mean(axis=0)).fillna(0.0)
    values = pivot.to_numpy(float)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    standardized = (values - means) / np.where(stds < 1e-9, 1.0, stds)
    distances = pairwise_distances(standardized, metric="euclidean")
    mean_distance = distances.mean(axis=1)

    # Maximin farthest-point sampling optimizes distance *within* the selected
    # set. Ranking by distance to the whole pool can select a tight minority
    # cluster whose members are far from the majority but close to each other.
    first_pair = np.unravel_index(np.argmax(distances), distances.shape)
    selected_idx = [int(first_pair[0]), int(first_pair[1])]
    while len(selected_idx) < SELECTED:
        remaining = np.asarray([index for index in range(len(pivot)) if index not in selected_idx])
        to_selected = distances[np.ix_(remaining, np.asarray(selected_idx))]
        min_distance = to_selected.min(axis=1)
        mean_to_selected = to_selected.mean(axis=1)
        order = np.lexsort((-mean_to_selected, -min_distance))
        selected_idx.append(int(remaining[order[0]]))
    selected = pivot.index[selected_idx].tolist()
    selected_pairwise = distances[np.ix_(selected_idx, selected_idx)]
    selected_mean = float(selected_pairwise[np.triu_indices(SELECTED, 1)].mean())

    rng = np.random.default_rng(SEED)
    random_means = []
    for _ in range(5000):
        idx = rng.choice(len(candidates), size=SELECTED, replace=False)
        sample = distances[np.ix_(idx, idx)]
        random_means.append(float(sample[np.triu_indices(SELECTED, 1)].mean()))
    random_means_arr = np.asarray(random_means)
    stats = {
        "selected_ids": selected,
        "selected_mean_pairwise_distance": selected_mean,
        "random_trials": 5000,
        "random_mean": float(random_means_arr.mean()),
        "random_sd": float(random_means_arr.std(ddof=1)),
        "random_p95": float(np.quantile(random_means_arr, 0.95)),
        "random_max": float(random_means_arr.max()),
        "empirical_p_ge_selected": float((np.sum(random_means_arr >= selected_mean) + 1) / (len(random_means_arr) + 1)),
        "selection_method": "standardize per training task, then deterministic maximin farthest-point sampling",
        "candidate_mean_distances": {pivot.index[i]: float(mean_distance[i]) for i in range(len(pivot))},
    }
    return selected, stats


def fit_ranker(matrix: pd.DataFrame, selected_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = matrix[(matrix["candidate_id"].isin(selected_ids)) & (matrix["status"] == "success")].copy()
    feature_cols = [column for column in subset.columns if column.startswith("task_") or column.startswith("cand_")]
    train = subset[subset["split"] == "train"]
    heldout = subset[subset["split"] == "heldout"]
    ranker = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    ranker.fit(train[feature_cols], train["score"])
    heldout = heldout.copy()
    heldout["predicted_score"] = ranker.predict(heldout[feature_cols])

    global_best_id = train.groupby("candidate_id")["score"].mean().idxmax()
    rows = []
    rng = np.random.default_rng(SEED)
    for task_name, group in heldout.groupby("task"):
        chosen = group.sort_values("predicted_score", ascending=False).iloc[0]
        global_row = group[group["candidate_id"] == global_best_id].iloc[0]
        random_scores = group["score"].to_numpy()
        bootstrap = rng.choice(random_scores, size=2000, replace=True)
        oracle = group.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "task": task_name,
                "modality": chosen["modality"],
                "agent_candidate": chosen["candidate_id"],
                "agent_score": float(chosen["score"]),
                "global_candidate": global_best_id,
                "global_score": float(global_row["score"]),
                "random_mean": float(bootstrap.mean()),
                "random_sd": float(bootstrap.std(ddof=1)),
                "oracle_candidate": oracle["candidate_id"],
                "oracle_score": float(oracle["score"]),
            }
        )
    results = pd.DataFrame(rows)
    differences = results["agent_score"] - results["global_score"]
    summary = {
        "training_tasks": TRAIN_TASKS,
        "heldout_tasks": HELDOUT_TASKS,
        "selected_candidates": SELECTED,
        "agent_mean": float(results["agent_score"].mean()),
        "global_mean": float(results["global_score"].mean()),
        "random_mean": float(results["random_mean"].mean()),
        "oracle_mean": float(results["oracle_score"].mean()),
        "agent_minus_global_mean": float(differences.mean()),
        "agent_wins_vs_global": int((differences > 0).sum()),
        "agent_ties_vs_global": int((differences == 0).sum()),
        "modalities": sorted(results["modality"].unique().tolist()),
        "feature_columns": feature_cols,
        "global_best_candidate": global_best_id,
    }
    return results, summary


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def reward_table(generalization: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        estimator(Candidate("diagnostic-invalid", "not-a-model", 1.0, False, "invalid"), False, SEED)
        actual_error = "unexpected_success"
    except Exception as exc:
        actual_error = f"{type(exc).__name__}: {exc}"
    latency_budget = float(matrix.loc[matrix["status"] == "success", "fit_seconds"].quantile(0.95))
    corner_row = matrix[matrix["status"] == "success"].sort_values("fit_seconds", ascending=False).iloc[0]
    rows = [
        {"task": "branch-test", "outcome": "invalid_action", "delta": None, "alpha": None, "reward": 0.0, "feedback": "action is outside the declared candidate action space"},
        {"task": "branch-test", "outcome": "execution_error", "delta": None, "alpha": None, "reward": 0.0, "feedback": actual_error},
        {"task": "branch-test", "outcome": "valid_non_edit", "delta": 0.0, "alpha": None, "reward": 0.5, "feedback": "list task metadata"},
        {
            "task": str(corner_row["task"]),
            "outcome": "corner_case",
            "delta": 0.0,
            "alpha": None,
            "reward": 0.5,
            "feedback": f"fit_seconds={corner_row['fit_seconds']:.6f} exceeded predeclared p95 budget={latency_budget:.6f}",
        },
    ]
    for result in generalization.to_dict(orient="records"):
        task_rows = matrix[(matrix["task"] == result["task"]) & (matrix["status"] == "success")]
        initial = float(result["global_score"])
        best = float(task_rows["score"].max())
        final = float(result["agent_score"])
        denominator = best - initial
        alpha = 100.0 / denominator if denominator > 1e-12 else 0.0
        delta = final - initial
        reward = sigmoid(alpha * delta) if alpha else 0.5
        rows.append(
            {
                "task": result["task"],
                "outcome": "successful_edit",
                "delta": delta,
                "alpha": alpha,
                "reward": reward,
                "feedback": f"metric moved from {initial:.6f} to {final:.6f}; oracle={best:.6f}",
            }
        )
    rewards = pd.DataFrame(rows)
    successful = rewards[rewards["outcome"] == "successful_edit"]
    summary = {
        "error_rewards": sorted(rewards[rewards["outcome"].isin(["invalid_action", "execution_error"])]["reward"].unique().tolist()),
        "neutral_rewards": sorted(rewards[rewards["outcome"].isin(["valid_non_edit", "corner_case"])]["reward"].unique().tolist()),
        "successful_edit_mean_reward": float(successful["reward"].mean()),
        "successful_edit_min_reward": float(successful["reward"].min()),
        "successful_edit_max_reward": float(successful["reward"].max()),
        "successful_edits": int(len(successful)),
        "actual_error_feedback": actual_error,
        "corner_latency_budget_seconds": latency_budget,
    }
    return rewards, summary


def make_figures(output_dir: Path, diversity: dict[str, Any], generalization: pd.DataFrame, rewards: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(
        ["selected 10", "random mean", "random 95%"],
        [diversity["selected_mean_pairwise_distance"], diversity["random_mean"], diversity["random_p95"]],
        color=["#2f6bff", "#a8b4c8", "#f0a43a"],
    )
    axes[0].set_title("Behavioral diversity")
    axes[0].set_ylabel("mean pairwise distance")

    means = generalization[["agent_score", "global_score", "random_mean", "oracle_score"]].mean()
    axes[1].bar(["agent", "global", "random", "oracle"], means, color=["#2f6bff", "#6b7280", "#a8b4c8", "#27ae60"])
    axes[1].set_ylim(max(0, means.min() - 0.1), min(1.0, means.max() + 0.05))
    axes[1].set_title("10 held-out tasks")
    axes[1].set_ylabel("validation accuracy")

    successful = rewards[rewards["outcome"] == "successful_edit"]
    axes[2].hist(successful["reward"], bins=np.linspace(0, 1, 11), color="#2f6bff", edgecolor="white")
    axes[2].axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5)
    axes[2].set_title("Successful-edit rewards")
    axes[2].set_xlabel("sigmoid-scaled reward")
    axes[2].set_ylabel("tasks")
    fig.tight_layout()
    fig.savefig(output_dir / "scaled_suite_summary.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scaled_suite"))
    parser.add_argument("--data-home", type=Path, default=Path("/tmp/ml-agent-20news"))
    parser.add_argument("--reuse-matrix", type=Path, help="recompute selection/evaluation from a previously executed matrix")
    parser.add_argument("--offline-text", action="store_true", help="use only for a deterministic local smoke test")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    candidates = candidate_pool()
    if args.reuse_matrix is not None:
        matrix = pd.read_csv(args.reuse_matrix)
        expected_rows = (TRAIN_TASKS + HELDOUT_TASKS) * CANDIDATES
        if len(matrix) != expected_rows or matrix["candidate_id"].nunique() != CANDIDATES or matrix["task"].nunique() != TRAIN_TASKS + HELDOUT_TASKS:
            raise ValueError("reuse matrix does not match the declared 19-task x 120-action protocol")
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        print(f"REUSED_EXECUTION_MATRIX={args.reuse_matrix} rows={len(matrix)}", flush=True)
    else:
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        matrix = evaluate(tasks, candidates, output_dir)
    matrix.to_csv(output_dir / "execution_matrix.csv", index=False)

    selected_ids, diversity = select_diverse(matrix, candidates)
    generalization, generalization_summary = fit_ranker(matrix, selected_ids)
    generalization.to_csv(output_dir / "heldout_results.csv", index=False)
    rewards, reward_summary = reward_table(generalization, matrix)
    rewards.to_csv(output_dir / "reward_events.csv", index=False)
    make_figures(output_dir, diversity, generalization, rewards)

    task_manifest = [
        {
            "name": task.name,
            "split": task.split,
            "modality": task.modality,
            "samples": int(task.X.shape[0]),
            "features": int(task.X.shape[1]),
            "classes": int(np.unique(task.y).size),
        }
        for task in tasks
    ]
    result = {
        "protocol": {
            "label": "scaled_independent_reproduction",
            "paper": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
            "openreview_id": "kcPPWaoegr",
            "seed": SEED,
            "training_tasks": TRAIN_TASKS,
            "heldout_tasks": HELDOUT_TASKS,
            "candidate_actions": CANDIDATES,
            "selected_actions": SELECTED,
            "random_subsets": 5000,
            "offline_text_smoke": bool(args.offline_text),
            "scope_boundary": "offline pipeline-action environment; no unreleased ML-Agent checkpoint, trajectories, or PPO weights",
        },
        "task_manifest": task_manifest,
        "diversity": diversity,
        "generalization": generalization_summary,
        "reward": reward_summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    checksums = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"CHECKSUMS.json", "execution_matrix.partial.csv"}
    }
    (output_dir / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")

    print(f"RESULTS_JSON={result_path}")
    print(f"TRAIN_TASKS={TRAIN_TASKS}")
    print(f"HELDOUT_TASKS={HELDOUT_TASKS}")
    print(f"CANDIDATES={CANDIDATES}")
    print(f"SELECTED={SELECTED}")
    print(f"DIVERSITY_P={diversity['empirical_p_ge_selected']:.8f}")
    print(f"AGENT_HELDOUT_MEAN={generalization_summary['agent_mean']:.6f}")
    print(f"GLOBAL_HELDOUT_MEAN={generalization_summary['global_mean']:.6f}")
    print(f"AGENT_MINUS_GLOBAL={generalization_summary['agent_minus_global_mean']:.6f}")
    print(f"WALL_SECONDS={result['environment']['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()

````


````output
/Users/binzhang/.matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /var/folders/zw/08w66lb16553mj5xxd__26tc0000gn/T/matplotlib-f3u5uy64 because there was an issue with the default path ({configdir}); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.

Fontconfig error: No writable cache directories
	/opt/homebrew/var/cache/fontconfig
	/Users/binzhang/.cache/fontconfig
	/Users/binzhang/.fontconfig
Matplotlib is building the font cache; this may take a moment.
REUSED_EXECUTION_MATRIX=outputs/scaled_suite/execution_matrix.csv rows=2280

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_1cd498bd5842", "created_at": "2026-07-17T19:35:34+00:00", "title": "Artifact: execution_matrix.csv", "path": "outputs/scaled_suite_corrected/execution_matrix.csv", "size": 556443, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/execution_matrix.csv` · dataset · 0.6 MB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/execution_matrix.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_687a3d16d6fd", "created_at": "2026-07-17T19:35:34+00:00", "title": "Artifact: heldout_results.csv", "path": "outputs/scaled_suite_corrected/heldout_results.csv", "size": 1860, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/heldout_results.csv` · dataset · 1.9 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/heldout_results.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_cc3e3bc3f59d", "created_at": "2026-07-17T19:35:34+00:00", "title": "Artifact: reward_events.csv", "path": "outputs/scaled_suite_corrected/reward_events.csv", "size": 1697, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/reward_events.csv` · dataset · 1.7 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/reward_events.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_951ef2078621", "created_at": "2026-07-17T19:35:58+00:00", "title": "Resolved rerun with writable plotting caches", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "MPLCONFIGDIR=/tmp/mpl-cache", "XDG_CACHE_HOME=/tmp/xdg-cache", "uv", "run", "scripts/run_scaled_suite.py", "--output-dir", "outputs/scaled_suite_corrected", "--data-home", "/tmp/ml-agent-20news", "--reuse-matrix", "outputs/scaled_suite/execution_matrix.csv"], "exit_code": 134, "duration_s": 15.721}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache uv run scripts/run_scaled_suite.py --output-dir outputs/scaled_suite_corrected --data-home /tmp/ml-agent-20news --reuse-matrix outputs/scaled_suite/execution_matrix.csv
````

exit 134 · 15.7s


````python title=run_scaled_suite.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.5",
# ]
# ///
"""Scaled independent reproduction of ML-Agent claims 2, 4, and 5.

The experiment uses 120 executable pipeline actions, exactly nine training
tasks, and ten held-out tasks spanning tabular, image, and text modalities.
Candidate diversity is computed from measured validation-score behavior, not
from idea text. A task-conditioned ranker is trained only on the nine training
tasks and evaluated on all held-out tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC


SEED = 17897
TRAIN_TASKS = 9
HELDOUT_TASKS = 10
CANDIDATES = 120
SELECTED = 10


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    strength: float
    balanced: bool
    variant: str


@dataclass
class Task:
    name: str
    split: str
    modality: str
    X: Any
    y: np.ndarray
    sparse: bool = False


def candidate_pool() -> list[Candidate]:
    pool: list[Candidate] = []
    strengths = np.logspace(-3, 3, 20)
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"logreg-c{value:.5g}-b{int(balanced)}", "logreg", float(value), balanced, "l2"))
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"linearsvc-c{value:.5g}-b{int(balanced)}", "linearsvc", float(value), balanced, "hinge"))
    alphas = np.logspace(-7, -2, 10)
    for loss in ("hinge", "log_loss"):
        for alpha in alphas:
            pool.append(Candidate(f"sgd-{loss}-a{alpha:.3g}", "sgd", float(alpha), True, loss))
    for weights in ("uniform", "distance"):
        for neighbors in (1, 3, 5, 7, 9, 11, 15, 21, 31, 45):
            pool.append(Candidate(f"knn-k{neighbors}-{weights}", "knn", float(neighbors), False, weights))
    assert len(pool) == CANDIDATES
    return pool


def estimator(candidate: Candidate, sparse: bool, seed: int):
    class_weight = "balanced" if candidate.balanced else None
    if candidate.family == "logreg":
        model = LogisticRegression(
            C=candidate.strength,
            class_weight=class_weight,
            max_iter=500,
            solver="liblinear" if not sparse else "saga",
            random_state=seed,
        )
    elif candidate.family == "linearsvc":
        model = LinearSVC(C=candidate.strength, class_weight=class_weight, random_state=seed, max_iter=5000)
    elif candidate.family == "sgd":
        model = SGDClassifier(
            loss=candidate.variant,
            alpha=candidate.strength,
            class_weight=class_weight,
            random_state=seed,
            max_iter=2000,
            tol=1e-4,
        )
    elif candidate.family == "knn":
        model = KNeighborsClassifier(n_neighbors=int(candidate.strength), weights=candidate.variant, n_jobs=1)
    else:
        raise ValueError(candidate.family)
    scaler = MaxAbsScaler() if sparse else StandardScaler()
    return make_pipeline(scaler, model)


def stratified_cap(X, y: np.ndarray, cap: int, seed: int):
    if len(y) <= cap:
        return X, y
    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=seed)
    return X_keep, y_keep


def numeric_task(name: str, split: str, modality: str, X, y, transform=None) -> Task:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if transform is not None:
        X, y = transform(X, y)
    X, y = stratified_cap(X, y, 1400, SEED)
    return Task(name, split, modality, X, y, sparse=False)


def load_tasks(offline_text: bool = False, data_home: Path | None = None) -> list[Task]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    diabetes = load_diabetes()
    digits = load_digits()

    rng = np.random.default_rng(SEED)

    tasks = [
        numeric_task("train-iris-multiclass", "train", "tabular", iris.data, iris.target),
        numeric_task("train-wine-multiclass", "train", "tabular", wine.data, wine.target),
        numeric_task("train-breast-cancer", "train", "tabular", cancer.data, cancer.target),
        numeric_task("train-digits-multiclass", "train", "image", digits.data, digits.target),
        numeric_task("train-digits-parity", "train", "image", digits.data, digits.target % 2),
        numeric_task("train-digits-low-high", "train", "image", digits.data, (digits.target >= 5).astype(int)),
    ]

    def half_features(X, y):
        return X[:, ::2], y

    def wine_zero(X, y):
        return X, (y == 0).astype(int)

    def iris_setosa(X, y):
        return X, (y == 0).astype(int)

    def noisy_digits(X, y):
        noise = rng.normal(0, 2.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y

    def rolled_digits(X, y):
        images = X.reshape(-1, 8, 8)
        rolled = np.roll(images, shift=1, axis=2)
        return rolled.reshape(-1, 64), y

    def noisy_parity(X, y):
        noise = rng.normal(0, 3.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y % 2

    tasks.extend(
        [
            numeric_task("heldout-iris-setosa", "heldout", "tabular", iris.data, iris.target, iris_setosa),
            numeric_task("heldout-wine-class-zero", "heldout", "tabular", wine.data, wine.target, wine_zero),
            numeric_task("heldout-cancer-half-features", "heldout", "tabular", cancer.data, cancer.target, half_features),
            numeric_task(
                "heldout-diabetes-above-median",
                "heldout",
                "tabular",
                diabetes.data,
                (diabetes.target >= np.median(diabetes.target)).astype(int),
            ),
            numeric_task("heldout-digits-noisy", "heldout", "image", digits.data, digits.target, noisy_digits),
            numeric_task("heldout-digits-shifted", "heldout", "image", digits.data, digits.target, rolled_digits),
            numeric_task("heldout-digits-noisy-parity", "heldout", "image", digits.data, digits.target, noisy_parity),
        ]
    )

    text_specs = [
        ("train-text-comp-vs-space", "train", ["comp.graphics", "sci.space"]),
        ("train-text-autos-vs-motorcycles", "train", ["rec.autos", "rec.motorcycles"]),
        ("train-text-politics", "train", ["talk.politics.guns", "talk.politics.misc"]),
        ("heldout-text-religion", "heldout", ["alt.atheism", "soc.religion.christian"]),
        ("heldout-text-computer-systems", "heldout", ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]),
        ("heldout-text-science", "heldout", ["sci.electronics", "sci.med"]),
    ]
    if offline_text:
        # Deterministic smoke corpus; full evidence runs must omit --offline-text.
        seeds = {
            "comp": ["graphics card pixel render", "space orbit rocket nasa"],
            "autos": ["engine wheel road car", "motorcycle helmet bike ride"],
            "politics": ["firearm law rights", "election policy debate"],
            "religion": ["atheist secular reason", "church christian faith"],
            "systems": ["ibm pc motherboard", "mac apple hardware"],
            "science": ["circuit voltage electronics", "medical patient treatment"],
        }
        for index, (name, split, _) in enumerate(text_specs):
            pair = list(seeds.values())[index]
            texts, labels = [], []
            for label, base in enumerate(pair):
                for j in range(80):
                    texts.append(f"{base} document topic {j % 11}")
                    labels.append(label)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=1)
            X = vectorizer.fit_transform(texts)
            tasks.append(Task(name, split, "text", X, np.asarray(labels), sparse=True))
    else:
        for name, split, categories in text_specs:
            data = fetch_20newsgroups(
                subset="all",
                categories=categories,
                remove=("headers", "footers", "quotes"),
                data_home=str(data_home) if data_home is not None else None,
            )
            X_text, y = stratified_cap(np.asarray(data.data, dtype=object), np.asarray(data.target), 1400, SEED)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, min_df=2, sublinear_tf=True)
            X = vectorizer.fit_transform(X_text.tolist())
            tasks.append(Task(name, split, "text", X, y, sparse=True))

    tasks.sort(key=lambda task: (task.split != "train", task.name))
    assert sum(task.split == "train" for task in tasks) == TRAIN_TASKS
    assert sum(task.split == "heldout" for task in tasks) == HELDOUT_TASKS
    assert {task.modality for task in tasks if task.split == "heldout"} == {"tabular", "image", "text"}
    return tasks


def task_features(task: Task) -> dict[str, float]:
    y = np.asarray(task.y)
    _, counts = np.unique(y, return_counts=True)
    n_samples, n_features = task.X.shape
    return {
        "log_samples": math.log1p(n_samples),
        "log_features": math.log1p(n_features),
        "classes": float(len(counts)),
        "imbalance": float(counts.max() / counts.sum()),
        "sparse": float(task.sparse),
        "mod_tabular": float(task.modality == "tabular"),
        "mod_image": float(task.modality == "image"),
        "mod_text": float(task.modality == "text"),
    }


def candidate_features(candidate: Candidate) -> dict[str, float]:
    return {
        "fam_logreg": float(candidate.family == "logreg"),
        "fam_linearsvc": float(candidate.family == "linearsvc"),
        "fam_sgd": float(candidate.family == "sgd"),
        "fam_knn": float(candidate.family == "knn"),
        "strength": math.log1p(candidate.strength) if candidate.family == "knn" else math.log10(candidate.strength),
        "balanced": float(candidate.balanced),
        "variant_distance": float(candidate.variant == "distance"),
        "variant_log": float(candidate.variant == "log_loss"),
    }


def evaluate(tasks: list[Task], candidates: list[Candidate], output_dir: Path) -> pd.DataFrame:
    rows = []
    total = len(tasks) * len(candidates)
    done = 0
    for task in tasks:
        X_train, X_valid, y_train, y_valid = train_test_split(
            task.X,
            task.y,
            test_size=0.30,
            random_state=SEED,
            stratify=task.y,
        )
        for candidate in candidates:
            started = time.perf_counter()
            status = "success"
            error = ""
            score = float("nan")
            try:
                model = estimator(candidate, task.sparse, SEED)
                model.fit(X_train, y_train)
                score = float(accuracy_score(y_valid, model.predict(X_valid)))
            except Exception as exc:  # preserved as real execution feedback
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            row = {
                "task": task.name,
                "split": task.split,
                "modality": task.modality,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "score": score,
                "status": status,
                "error": error,
                "fit_seconds": elapsed,
            }
            row.update({f"task_{key}": value for key, value in task_features(task).items()})
            row.update({f"cand_{key}": value for key, value in candidate_features(candidate).items()})
            rows.append(row)
            done += 1
            if done % 120 == 0:
                print(f"PROGRESS={done}/{total} last_task={task.name}", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "execution_matrix.partial.csv", index=False)
    return pd.DataFrame(rows)


def select_diverse(matrix: pd.DataFrame, candidates: list[Candidate]) -> tuple[list[str], dict[str, Any]]:
    train = matrix[(matrix["split"] == "train") & (matrix["status"] == "success")]
    pivot = train.pivot(index="candidate_id", columns="task", values="score").reindex([c.candidate_id for c in candidates])
    if pivot.isna().any().any():
        pivot = pivot.fillna(pivot.mean(axis=0)).fillna(0.0)
    values = pivot.to_numpy(float)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    standardized = (values - means) / np.where(stds < 1e-9, 1.0, stds)
    distances = pairwise_distances(standardized, metric="euclidean")
    mean_distance = distances.mean(axis=1)

    # Maximin farthest-point sampling optimizes distance *within* the selected
    # set. Ranking by distance to the whole pool can select a tight minority
    # cluster whose members are far from the majority but close to each other.
    first_pair = np.unravel_index(np.argmax(distances), distances.shape)
    selected_idx = [int(first_pair[0]), int(first_pair[1])]
    while len(selected_idx) < SELECTED:
        remaining = np.asarray([index for index in range(len(pivot)) if index not in selected_idx])
        to_selected = distances[np.ix_(remaining, np.asarray(selected_idx))]
        min_distance = to_selected.min(axis=1)
        mean_to_selected = to_selected.mean(axis=1)
        order = np.lexsort((-mean_to_selected, -min_distance))
        selected_idx.append(int(remaining[order[0]]))
    selected = pivot.index[selected_idx].tolist()
    selected_pairwise = distances[np.ix_(selected_idx, selected_idx)]
    selected_mean = float(selected_pairwise[np.triu_indices(SELECTED, 1)].mean())

    rng = np.random.default_rng(SEED)
    random_means = []
    for _ in range(5000):
        idx = rng.choice(len(candidates), size=SELECTED, replace=False)
        sample = distances[np.ix_(idx, idx)]
        random_means.append(float(sample[np.triu_indices(SELECTED, 1)].mean()))
    random_means_arr = np.asarray(random_means)
    stats = {
        "selected_ids": selected,
        "selected_mean_pairwise_distance": selected_mean,
        "random_trials": 5000,
        "random_mean": float(random_means_arr.mean()),
        "random_sd": float(random_means_arr.std(ddof=1)),
        "random_p95": float(np.quantile(random_means_arr, 0.95)),
        "random_max": float(random_means_arr.max()),
        "empirical_p_ge_selected": float((np.sum(random_means_arr >= selected_mean) + 1) / (len(random_means_arr) + 1)),
        "selection_method": "standardize per training task, then deterministic maximin farthest-point sampling",
        "candidate_mean_distances": {pivot.index[i]: float(mean_distance[i]) for i in range(len(pivot))},
    }
    return selected, stats


def fit_ranker(matrix: pd.DataFrame, selected_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = matrix[(matrix["candidate_id"].isin(selected_ids)) & (matrix["status"] == "success")].copy()
    feature_cols = [column for column in subset.columns if column.startswith("task_") or column.startswith("cand_")]
    train = subset[subset["split"] == "train"]
    heldout = subset[subset["split"] == "heldout"]
    ranker = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    ranker.fit(train[feature_cols], train["score"])
    heldout = heldout.copy()
    heldout["predicted_score"] = ranker.predict(heldout[feature_cols])

    global_best_id = train.groupby("candidate_id")["score"].mean().idxmax()
    rows = []
    rng = np.random.default_rng(SEED)
    for task_name, group in heldout.groupby("task"):
        chosen = group.sort_values("predicted_score", ascending=False).iloc[0]
        global_row = group[group["candidate_id"] == global_best_id].iloc[0]
        random_scores = group["score"].to_numpy()
        bootstrap = rng.choice(random_scores, size=2000, replace=True)
        oracle = group.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "task": task_name,
                "modality": chosen["modality"],
                "agent_candidate": chosen["candidate_id"],
                "agent_score": float(chosen["score"]),
                "global_candidate": global_best_id,
                "global_score": float(global_row["score"]),
                "random_mean": float(bootstrap.mean()),
                "random_sd": float(bootstrap.std(ddof=1)),
                "oracle_candidate": oracle["candidate_id"],
                "oracle_score": float(oracle["score"]),
            }
        )
    results = pd.DataFrame(rows)
    differences = results["agent_score"] - results["global_score"]
    summary = {
        "training_tasks": TRAIN_TASKS,
        "heldout_tasks": HELDOUT_TASKS,
        "selected_candidates": SELECTED,
        "agent_mean": float(results["agent_score"].mean()),
        "global_mean": float(results["global_score"].mean()),
        "random_mean": float(results["random_mean"].mean()),
        "oracle_mean": float(results["oracle_score"].mean()),
        "agent_minus_global_mean": float(differences.mean()),
        "agent_wins_vs_global": int((differences > 0).sum()),
        "agent_ties_vs_global": int((differences == 0).sum()),
        "modalities": sorted(results["modality"].unique().tolist()),
        "feature_columns": feature_cols,
        "global_best_candidate": global_best_id,
    }
    return results, summary


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def reward_table(generalization: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        estimator(Candidate("diagnostic-invalid", "not-a-model", 1.0, False, "invalid"), False, SEED)
        actual_error = "unexpected_success"
    except Exception as exc:
        actual_error = f"{type(exc).__name__}: {exc}"
    latency_budget = float(matrix.loc[matrix["status"] == "success", "fit_seconds"].quantile(0.95))
    corner_row = matrix[matrix["status"] == "success"].sort_values("fit_seconds", ascending=False).iloc[0]
    rows = [
        {"task": "branch-test", "outcome": "invalid_action", "delta": None, "alpha": None, "reward": 0.0, "feedback": "action is outside the declared candidate action space"},
        {"task": "branch-test", "outcome": "execution_error", "delta": None, "alpha": None, "reward": 0.0, "feedback": actual_error},
        {"task": "branch-test", "outcome": "valid_non_edit", "delta": 0.0, "alpha": None, "reward": 0.5, "feedback": "list task metadata"},
        {
            "task": str(corner_row["task"]),
            "outcome": "corner_case",
            "delta": 0.0,
            "alpha": None,
            "reward": 0.5,
            "feedback": f"fit_seconds={corner_row['fit_seconds']:.6f} exceeded predeclared p95 budget={latency_budget:.6f}",
        },
    ]
    for result in generalization.to_dict(orient="records"):
        task_rows = matrix[(matrix["task"] == result["task"]) & (matrix["status"] == "success")]
        initial = float(result["global_score"])
        best = float(task_rows["score"].max())
        final = float(result["agent_score"])
        denominator = best - initial
        alpha = 100.0 / denominator if denominator > 1e-12 else 0.0
        delta = final - initial
        reward = sigmoid(alpha * delta) if alpha else 0.5
        rows.append(
            {
                "task": result["task"],
                "outcome": "successful_edit",
                "delta": delta,
                "alpha": alpha,
                "reward": reward,
                "feedback": f"metric moved from {initial:.6f} to {final:.6f}; oracle={best:.6f}",
            }
        )
    rewards = pd.DataFrame(rows)
    successful = rewards[rewards["outcome"] == "successful_edit"]
    summary = {
        "error_rewards": sorted(rewards[rewards["outcome"].isin(["invalid_action", "execution_error"])]["reward"].unique().tolist()),
        "neutral_rewards": sorted(rewards[rewards["outcome"].isin(["valid_non_edit", "corner_case"])]["reward"].unique().tolist()),
        "successful_edit_mean_reward": float(successful["reward"].mean()),
        "successful_edit_min_reward": float(successful["reward"].min()),
        "successful_edit_max_reward": float(successful["reward"].max()),
        "successful_edits": int(len(successful)),
        "actual_error_feedback": actual_error,
        "corner_latency_budget_seconds": latency_budget,
    }
    return rewards, summary


def make_figures(output_dir: Path, diversity: dict[str, Any], generalization: pd.DataFrame, rewards: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(
        ["selected 10", "random mean", "random 95%"],
        [diversity["selected_mean_pairwise_distance"], diversity["random_mean"], diversity["random_p95"]],
        color=["#2f6bff", "#a8b4c8", "#f0a43a"],
    )
    axes[0].set_title("Behavioral diversity")
    axes[0].set_ylabel("mean pairwise distance")

    means = generalization[["agent_score", "global_score", "random_mean", "oracle_score"]].mean()
    axes[1].bar(["agent", "global", "random", "oracle"], means, color=["#2f6bff", "#6b7280", "#a8b4c8", "#27ae60"])
    axes[1].set_ylim(max(0, means.min() - 0.1), min(1.0, means.max() + 0.05))
    axes[1].set_title("10 held-out tasks")
    axes[1].set_ylabel("validation accuracy")

    successful = rewards[rewards["outcome"] == "successful_edit"]
    axes[2].hist(successful["reward"], bins=np.linspace(0, 1, 11), color="#2f6bff", edgecolor="white")
    axes[2].axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5)
    axes[2].set_title("Successful-edit rewards")
    axes[2].set_xlabel("sigmoid-scaled reward")
    axes[2].set_ylabel("tasks")
    fig.tight_layout()
    fig.savefig(output_dir / "scaled_suite_summary.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scaled_suite"))
    parser.add_argument("--data-home", type=Path, default=Path("/tmp/ml-agent-20news"))
    parser.add_argument("--reuse-matrix", type=Path, help="recompute selection/evaluation from a previously executed matrix")
    parser.add_argument("--offline-text", action="store_true", help="use only for a deterministic local smoke test")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    candidates = candidate_pool()
    if args.reuse_matrix is not None:
        matrix = pd.read_csv(args.reuse_matrix)
        expected_rows = (TRAIN_TASKS + HELDOUT_TASKS) * CANDIDATES
        if len(matrix) != expected_rows or matrix["candidate_id"].nunique() != CANDIDATES or matrix["task"].nunique() != TRAIN_TASKS + HELDOUT_TASKS:
            raise ValueError("reuse matrix does not match the declared 19-task x 120-action protocol")
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        print(f"REUSED_EXECUTION_MATRIX={args.reuse_matrix} rows={len(matrix)}", flush=True)
    else:
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        matrix = evaluate(tasks, candidates, output_dir)
    matrix.to_csv(output_dir / "execution_matrix.csv", index=False)

    selected_ids, diversity = select_diverse(matrix, candidates)
    generalization, generalization_summary = fit_ranker(matrix, selected_ids)
    generalization.to_csv(output_dir / "heldout_results.csv", index=False)
    rewards, reward_summary = reward_table(generalization, matrix)
    rewards.to_csv(output_dir / "reward_events.csv", index=False)
    make_figures(output_dir, diversity, generalization, rewards)

    task_manifest = [
        {
            "name": task.name,
            "split": task.split,
            "modality": task.modality,
            "samples": int(task.X.shape[0]),
            "features": int(task.X.shape[1]),
            "classes": int(np.unique(task.y).size),
        }
        for task in tasks
    ]
    result = {
        "protocol": {
            "label": "scaled_independent_reproduction",
            "paper": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
            "openreview_id": "kcPPWaoegr",
            "seed": SEED,
            "training_tasks": TRAIN_TASKS,
            "heldout_tasks": HELDOUT_TASKS,
            "candidate_actions": CANDIDATES,
            "selected_actions": SELECTED,
            "random_subsets": 5000,
            "offline_text_smoke": bool(args.offline_text),
            "scope_boundary": "offline pipeline-action environment; no unreleased ML-Agent checkpoint, trajectories, or PPO weights",
        },
        "task_manifest": task_manifest,
        "diversity": diversity,
        "generalization": generalization_summary,
        "reward": reward_summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    checksums = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"CHECKSUMS.json", "execution_matrix.partial.csv"}
    }
    (output_dir / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")

    print(f"RESULTS_JSON={result_path}")
    print(f"TRAIN_TASKS={TRAIN_TASKS}")
    print(f"HELDOUT_TASKS={HELDOUT_TASKS}")
    print(f"CANDIDATES={CANDIDATES}")
    print(f"SELECTED={SELECTED}")
    print(f"DIVERSITY_P={diversity['empirical_p_ge_selected']:.8f}")
    print(f"AGENT_HELDOUT_MEAN={generalization_summary['agent_mean']:.6f}")
    print(f"GLOBAL_HELDOUT_MEAN={generalization_summary['global_mean']:.6f}")
    print(f"AGENT_MINUS_GLOBAL={generalization_summary['agent_minus_global_mean']:.6f}")
    print(f"WALL_SECONDS={result['environment']['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()

````


````output
Matplotlib is building the font cache; this may take a moment.
REUSED_EXECUTION_MATRIX=outputs/scaled_suite/execution_matrix.csv rows=2280

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_73e7be1cc46c", "created_at": "2026-07-17T19:35:58+00:00", "title": "Artifact: execution_matrix.csv", "path": "outputs/scaled_suite_corrected/execution_matrix.csv", "size": 556443, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/execution_matrix.csv` · dataset · 0.6 MB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/execution_matrix.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_fda759028ed2", "created_at": "2026-07-17T19:35:58+00:00", "title": "Artifact: heldout_results.csv", "path": "outputs/scaled_suite_corrected/heldout_results.csv", "size": 1860, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/heldout_results.csv` · dataset · 1.9 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/heldout_results.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_e7d02b95ced5", "created_at": "2026-07-17T19:35:58+00:00", "title": "Artifact: reward_events.csv", "path": "outputs/scaled_suite_corrected/reward_events.csv", "size": 1697, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/reward_events.csv` · dataset · 1.7 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/reward_events.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_ac2704987e35", "created_at": "2026-07-17T19:36:51+00:00", "title": "Successful headless rerun: deterministic maximin selection", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "MPLCONFIGDIR=/tmp/mpl-cache", "XDG_CACHE_HOME=/tmp/xdg-cache", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "uv", "run", "scripts/run_scaled_suite.py", "--output-dir", "outputs/scaled_suite_corrected", "--data-home", "/tmp/ml-agent-20news", "--reuse-matrix", "outputs/scaled_suite/execution_matrix.csv"], "exit_code": 0, "duration_s": 6.472}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run scripts/run_scaled_suite.py --output-dir outputs/scaled_suite_corrected --data-home /tmp/ml-agent-20news --reuse-matrix outputs/scaled_suite/execution_matrix.csv
````

exit 0 · 6.5s


````python title=run_scaled_suite.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.5",
# ]
# ///
"""Scaled independent reproduction of ML-Agent claims 2, 4, and 5.

The experiment uses 120 executable pipeline actions, exactly nine training
tasks, and ten held-out tasks spanning tabular, image, and text modalities.
Candidate diversity is computed from measured validation-score behavior, not
from idea text. A task-conditioned ranker is trained only on the nine training
tasks and evaluated on all held-out tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC


SEED = 17897
TRAIN_TASKS = 9
HELDOUT_TASKS = 10
CANDIDATES = 120
SELECTED = 10


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    strength: float
    balanced: bool
    variant: str


@dataclass
class Task:
    name: str
    split: str
    modality: str
    X: Any
    y: np.ndarray
    sparse: bool = False


def candidate_pool() -> list[Candidate]:
    pool: list[Candidate] = []
    strengths = np.logspace(-3, 3, 20)
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"logreg-c{value:.5g}-b{int(balanced)}", "logreg", float(value), balanced, "l2"))
    for balanced in (False, True):
        for value in strengths:
            pool.append(Candidate(f"linearsvc-c{value:.5g}-b{int(balanced)}", "linearsvc", float(value), balanced, "hinge"))
    alphas = np.logspace(-7, -2, 10)
    for loss in ("hinge", "log_loss"):
        for alpha in alphas:
            pool.append(Candidate(f"sgd-{loss}-a{alpha:.3g}", "sgd", float(alpha), True, loss))
    for weights in ("uniform", "distance"):
        for neighbors in (1, 3, 5, 7, 9, 11, 15, 21, 31, 45):
            pool.append(Candidate(f"knn-k{neighbors}-{weights}", "knn", float(neighbors), False, weights))
    assert len(pool) == CANDIDATES
    return pool


def estimator(candidate: Candidate, sparse: bool, seed: int):
    class_weight = "balanced" if candidate.balanced else None
    if candidate.family == "logreg":
        model = LogisticRegression(
            C=candidate.strength,
            class_weight=class_weight,
            max_iter=500,
            solver="liblinear" if not sparse else "saga",
            random_state=seed,
        )
    elif candidate.family == "linearsvc":
        model = LinearSVC(C=candidate.strength, class_weight=class_weight, random_state=seed, max_iter=5000)
    elif candidate.family == "sgd":
        model = SGDClassifier(
            loss=candidate.variant,
            alpha=candidate.strength,
            class_weight=class_weight,
            random_state=seed,
            max_iter=2000,
            tol=1e-4,
        )
    elif candidate.family == "knn":
        model = KNeighborsClassifier(n_neighbors=int(candidate.strength), weights=candidate.variant, n_jobs=1)
    else:
        raise ValueError(candidate.family)
    scaler = MaxAbsScaler() if sparse else StandardScaler()
    return make_pipeline(scaler, model)


def stratified_cap(X, y: np.ndarray, cap: int, seed: int):
    if len(y) <= cap:
        return X, y
    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=seed)
    return X_keep, y_keep


def numeric_task(name: str, split: str, modality: str, X, y, transform=None) -> Task:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if transform is not None:
        X, y = transform(X, y)
    X, y = stratified_cap(X, y, 1400, SEED)
    return Task(name, split, modality, X, y, sparse=False)


def load_tasks(offline_text: bool = False, data_home: Path | None = None) -> list[Task]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    diabetes = load_diabetes()
    digits = load_digits()

    rng = np.random.default_rng(SEED)

    tasks = [
        numeric_task("train-iris-multiclass", "train", "tabular", iris.data, iris.target),
        numeric_task("train-wine-multiclass", "train", "tabular", wine.data, wine.target),
        numeric_task("train-breast-cancer", "train", "tabular", cancer.data, cancer.target),
        numeric_task("train-digits-multiclass", "train", "image", digits.data, digits.target),
        numeric_task("train-digits-parity", "train", "image", digits.data, digits.target % 2),
        numeric_task("train-digits-low-high", "train", "image", digits.data, (digits.target >= 5).astype(int)),
    ]

    def half_features(X, y):
        return X[:, ::2], y

    def wine_zero(X, y):
        return X, (y == 0).astype(int)

    def iris_setosa(X, y):
        return X, (y == 0).astype(int)

    def noisy_digits(X, y):
        noise = rng.normal(0, 2.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y

    def rolled_digits(X, y):
        images = X.reshape(-1, 8, 8)
        rolled = np.roll(images, shift=1, axis=2)
        return rolled.reshape(-1, 64), y

    def noisy_parity(X, y):
        noise = rng.normal(0, 3.0, size=X.shape).astype(np.float32)
        return np.clip(X + noise, 0, 16), y % 2

    tasks.extend(
        [
            numeric_task("heldout-iris-setosa", "heldout", "tabular", iris.data, iris.target, iris_setosa),
            numeric_task("heldout-wine-class-zero", "heldout", "tabular", wine.data, wine.target, wine_zero),
            numeric_task("heldout-cancer-half-features", "heldout", "tabular", cancer.data, cancer.target, half_features),
            numeric_task(
                "heldout-diabetes-above-median",
                "heldout",
                "tabular",
                diabetes.data,
                (diabetes.target >= np.median(diabetes.target)).astype(int),
            ),
            numeric_task("heldout-digits-noisy", "heldout", "image", digits.data, digits.target, noisy_digits),
            numeric_task("heldout-digits-shifted", "heldout", "image", digits.data, digits.target, rolled_digits),
            numeric_task("heldout-digits-noisy-parity", "heldout", "image", digits.data, digits.target, noisy_parity),
        ]
    )

    text_specs = [
        ("train-text-comp-vs-space", "train", ["comp.graphics", "sci.space"]),
        ("train-text-autos-vs-motorcycles", "train", ["rec.autos", "rec.motorcycles"]),
        ("train-text-politics", "train", ["talk.politics.guns", "talk.politics.misc"]),
        ("heldout-text-religion", "heldout", ["alt.atheism", "soc.religion.christian"]),
        ("heldout-text-computer-systems", "heldout", ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]),
        ("heldout-text-science", "heldout", ["sci.electronics", "sci.med"]),
    ]
    if offline_text:
        # Deterministic smoke corpus; full evidence runs must omit --offline-text.
        seeds = {
            "comp": ["graphics card pixel render", "space orbit rocket nasa"],
            "autos": ["engine wheel road car", "motorcycle helmet bike ride"],
            "politics": ["firearm law rights", "election policy debate"],
            "religion": ["atheist secular reason", "church christian faith"],
            "systems": ["ibm pc motherboard", "mac apple hardware"],
            "science": ["circuit voltage electronics", "medical patient treatment"],
        }
        for index, (name, split, _) in enumerate(text_specs):
            pair = list(seeds.values())[index]
            texts, labels = [], []
            for label, base in enumerate(pair):
                for j in range(80):
                    texts.append(f"{base} document topic {j % 11}")
                    labels.append(label)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=1)
            X = vectorizer.fit_transform(texts)
            tasks.append(Task(name, split, "text", X, np.asarray(labels), sparse=True))
    else:
        for name, split, categories in text_specs:
            data = fetch_20newsgroups(
                subset="all",
                categories=categories,
                remove=("headers", "footers", "quotes"),
                data_home=str(data_home) if data_home is not None else None,
            )
            X_text, y = stratified_cap(np.asarray(data.data, dtype=object), np.asarray(data.target), 1400, SEED)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, min_df=2, sublinear_tf=True)
            X = vectorizer.fit_transform(X_text.tolist())
            tasks.append(Task(name, split, "text", X, y, sparse=True))

    tasks.sort(key=lambda task: (task.split != "train", task.name))
    assert sum(task.split == "train" for task in tasks) == TRAIN_TASKS
    assert sum(task.split == "heldout" for task in tasks) == HELDOUT_TASKS
    assert {task.modality for task in tasks if task.split == "heldout"} == {"tabular", "image", "text"}
    return tasks


def task_features(task: Task) -> dict[str, float]:
    y = np.asarray(task.y)
    _, counts = np.unique(y, return_counts=True)
    n_samples, n_features = task.X.shape
    return {
        "log_samples": math.log1p(n_samples),
        "log_features": math.log1p(n_features),
        "classes": float(len(counts)),
        "imbalance": float(counts.max() / counts.sum()),
        "sparse": float(task.sparse),
        "mod_tabular": float(task.modality == "tabular"),
        "mod_image": float(task.modality == "image"),
        "mod_text": float(task.modality == "text"),
    }


def candidate_features(candidate: Candidate) -> dict[str, float]:
    return {
        "fam_logreg": float(candidate.family == "logreg"),
        "fam_linearsvc": float(candidate.family == "linearsvc"),
        "fam_sgd": float(candidate.family == "sgd"),
        "fam_knn": float(candidate.family == "knn"),
        "strength": math.log1p(candidate.strength) if candidate.family == "knn" else math.log10(candidate.strength),
        "balanced": float(candidate.balanced),
        "variant_distance": float(candidate.variant == "distance"),
        "variant_log": float(candidate.variant == "log_loss"),
    }


def evaluate(tasks: list[Task], candidates: list[Candidate], output_dir: Path) -> pd.DataFrame:
    rows = []
    total = len(tasks) * len(candidates)
    done = 0
    for task in tasks:
        X_train, X_valid, y_train, y_valid = train_test_split(
            task.X,
            task.y,
            test_size=0.30,
            random_state=SEED,
            stratify=task.y,
        )
        for candidate in candidates:
            started = time.perf_counter()
            status = "success"
            error = ""
            score = float("nan")
            try:
                model = estimator(candidate, task.sparse, SEED)
                model.fit(X_train, y_train)
                score = float(accuracy_score(y_valid, model.predict(X_valid)))
            except Exception as exc:  # preserved as real execution feedback
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            row = {
                "task": task.name,
                "split": task.split,
                "modality": task.modality,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "score": score,
                "status": status,
                "error": error,
                "fit_seconds": elapsed,
            }
            row.update({f"task_{key}": value for key, value in task_features(task).items()})
            row.update({f"cand_{key}": value for key, value in candidate_features(candidate).items()})
            rows.append(row)
            done += 1
            if done % 120 == 0:
                print(f"PROGRESS={done}/{total} last_task={task.name}", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "execution_matrix.partial.csv", index=False)
    return pd.DataFrame(rows)


def select_diverse(matrix: pd.DataFrame, candidates: list[Candidate]) -> tuple[list[str], dict[str, Any]]:
    train = matrix[(matrix["split"] == "train") & (matrix["status"] == "success")]
    pivot = train.pivot(index="candidate_id", columns="task", values="score").reindex([c.candidate_id for c in candidates])
    if pivot.isna().any().any():
        pivot = pivot.fillna(pivot.mean(axis=0)).fillna(0.0)
    values = pivot.to_numpy(float)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    standardized = (values - means) / np.where(stds < 1e-9, 1.0, stds)
    distances = pairwise_distances(standardized, metric="euclidean")
    mean_distance = distances.mean(axis=1)

    # Maximin farthest-point sampling optimizes distance *within* the selected
    # set. Ranking by distance to the whole pool can select a tight minority
    # cluster whose members are far from the majority but close to each other.
    first_pair = np.unravel_index(np.argmax(distances), distances.shape)
    selected_idx = [int(first_pair[0]), int(first_pair[1])]
    while len(selected_idx) < SELECTED:
        remaining = np.asarray([index for index in range(len(pivot)) if index not in selected_idx])
        to_selected = distances[np.ix_(remaining, np.asarray(selected_idx))]
        min_distance = to_selected.min(axis=1)
        mean_to_selected = to_selected.mean(axis=1)
        order = np.lexsort((-mean_to_selected, -min_distance))
        selected_idx.append(int(remaining[order[0]]))
    selected = pivot.index[selected_idx].tolist()
    selected_pairwise = distances[np.ix_(selected_idx, selected_idx)]
    selected_mean = float(selected_pairwise[np.triu_indices(SELECTED, 1)].mean())

    rng = np.random.default_rng(SEED)
    random_means = []
    for _ in range(5000):
        idx = rng.choice(len(candidates), size=SELECTED, replace=False)
        sample = distances[np.ix_(idx, idx)]
        random_means.append(float(sample[np.triu_indices(SELECTED, 1)].mean()))
    random_means_arr = np.asarray(random_means)
    stats = {
        "selected_ids": selected,
        "selected_mean_pairwise_distance": selected_mean,
        "random_trials": 5000,
        "random_mean": float(random_means_arr.mean()),
        "random_sd": float(random_means_arr.std(ddof=1)),
        "random_p95": float(np.quantile(random_means_arr, 0.95)),
        "random_max": float(random_means_arr.max()),
        "empirical_p_ge_selected": float((np.sum(random_means_arr >= selected_mean) + 1) / (len(random_means_arr) + 1)),
        "selection_method": "standardize per training task, then deterministic maximin farthest-point sampling",
        "candidate_mean_distances": {pivot.index[i]: float(mean_distance[i]) for i in range(len(pivot))},
    }
    return selected, stats


def fit_ranker(matrix: pd.DataFrame, selected_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = matrix[(matrix["candidate_id"].isin(selected_ids)) & (matrix["status"] == "success")].copy()
    feature_cols = [column for column in subset.columns if column.startswith("task_") or column.startswith("cand_")]
    train = subset[subset["split"] == "train"]
    heldout = subset[subset["split"] == "heldout"]
    ranker = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    ranker.fit(train[feature_cols], train["score"])
    heldout = heldout.copy()
    heldout["predicted_score"] = ranker.predict(heldout[feature_cols])

    global_best_id = train.groupby("candidate_id")["score"].mean().idxmax()
    rows = []
    rng = np.random.default_rng(SEED)
    for task_name, group in heldout.groupby("task"):
        chosen = group.sort_values("predicted_score", ascending=False).iloc[0]
        global_row = group[group["candidate_id"] == global_best_id].iloc[0]
        random_scores = group["score"].to_numpy()
        bootstrap = rng.choice(random_scores, size=2000, replace=True)
        oracle = group.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "task": task_name,
                "modality": chosen["modality"],
                "agent_candidate": chosen["candidate_id"],
                "agent_score": float(chosen["score"]),
                "global_candidate": global_best_id,
                "global_score": float(global_row["score"]),
                "random_mean": float(bootstrap.mean()),
                "random_sd": float(bootstrap.std(ddof=1)),
                "oracle_candidate": oracle["candidate_id"],
                "oracle_score": float(oracle["score"]),
            }
        )
    results = pd.DataFrame(rows)
    differences = results["agent_score"] - results["global_score"]
    summary = {
        "training_tasks": TRAIN_TASKS,
        "heldout_tasks": HELDOUT_TASKS,
        "selected_candidates": SELECTED,
        "agent_mean": float(results["agent_score"].mean()),
        "global_mean": float(results["global_score"].mean()),
        "random_mean": float(results["random_mean"].mean()),
        "oracle_mean": float(results["oracle_score"].mean()),
        "agent_minus_global_mean": float(differences.mean()),
        "agent_wins_vs_global": int((differences > 0).sum()),
        "agent_ties_vs_global": int((differences == 0).sum()),
        "modalities": sorted(results["modality"].unique().tolist()),
        "feature_columns": feature_cols,
        "global_best_candidate": global_best_id,
    }
    return results, summary


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def reward_table(generalization: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        estimator(Candidate("diagnostic-invalid", "not-a-model", 1.0, False, "invalid"), False, SEED)
        actual_error = "unexpected_success"
    except Exception as exc:
        actual_error = f"{type(exc).__name__}: {exc}"
    latency_budget = float(matrix.loc[matrix["status"] == "success", "fit_seconds"].quantile(0.95))
    corner_row = matrix[matrix["status"] == "success"].sort_values("fit_seconds", ascending=False).iloc[0]
    rows = [
        {"task": "branch-test", "outcome": "invalid_action", "delta": None, "alpha": None, "reward": 0.0, "feedback": "action is outside the declared candidate action space"},
        {"task": "branch-test", "outcome": "execution_error", "delta": None, "alpha": None, "reward": 0.0, "feedback": actual_error},
        {"task": "branch-test", "outcome": "valid_non_edit", "delta": 0.0, "alpha": None, "reward": 0.5, "feedback": "list task metadata"},
        {
            "task": str(corner_row["task"]),
            "outcome": "corner_case",
            "delta": 0.0,
            "alpha": None,
            "reward": 0.5,
            "feedback": f"fit_seconds={corner_row['fit_seconds']:.6f} exceeded predeclared p95 budget={latency_budget:.6f}",
        },
    ]
    for result in generalization.to_dict(orient="records"):
        task_rows = matrix[(matrix["task"] == result["task"]) & (matrix["status"] == "success")]
        initial = float(result["global_score"])
        best = float(task_rows["score"].max())
        final = float(result["agent_score"])
        denominator = best - initial
        alpha = 100.0 / denominator if denominator > 1e-12 else 0.0
        delta = final - initial
        reward = sigmoid(alpha * delta) if alpha else 0.5
        rows.append(
            {
                "task": result["task"],
                "outcome": "successful_edit",
                "delta": delta,
                "alpha": alpha,
                "reward": reward,
                "feedback": f"metric moved from {initial:.6f} to {final:.6f}; oracle={best:.6f}",
            }
        )
    rewards = pd.DataFrame(rows)
    successful = rewards[rewards["outcome"] == "successful_edit"]
    summary = {
        "error_rewards": sorted(rewards[rewards["outcome"].isin(["invalid_action", "execution_error"])]["reward"].unique().tolist()),
        "neutral_rewards": sorted(rewards[rewards["outcome"].isin(["valid_non_edit", "corner_case"])]["reward"].unique().tolist()),
        "successful_edit_mean_reward": float(successful["reward"].mean()),
        "successful_edit_min_reward": float(successful["reward"].min()),
        "successful_edit_max_reward": float(successful["reward"].max()),
        "successful_edits": int(len(successful)),
        "actual_error_feedback": actual_error,
        "corner_latency_budget_seconds": latency_budget,
    }
    return rewards, summary


def make_figures(output_dir: Path, diversity: dict[str, Any], generalization: pd.DataFrame, rewards: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(
        ["selected 10", "random mean", "random 95%"],
        [diversity["selected_mean_pairwise_distance"], diversity["random_mean"], diversity["random_p95"]],
        color=["#2f6bff", "#a8b4c8", "#f0a43a"],
    )
    axes[0].set_title("Behavioral diversity")
    axes[0].set_ylabel("mean pairwise distance")

    means = generalization[["agent_score", "global_score", "random_mean", "oracle_score"]].mean()
    axes[1].bar(["agent", "global", "random", "oracle"], means, color=["#2f6bff", "#6b7280", "#a8b4c8", "#27ae60"])
    axes[1].set_ylim(max(0, means.min() - 0.1), min(1.0, means.max() + 0.05))
    axes[1].set_title("10 held-out tasks")
    axes[1].set_ylabel("validation accuracy")

    successful = rewards[rewards["outcome"] == "successful_edit"]
    axes[2].hist(successful["reward"], bins=np.linspace(0, 1, 11), color="#2f6bff", edgecolor="white")
    axes[2].axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5)
    axes[2].set_title("Successful-edit rewards")
    axes[2].set_xlabel("sigmoid-scaled reward")
    axes[2].set_ylabel("tasks")
    fig.tight_layout()
    fig.savefig(output_dir / "scaled_suite_summary.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scaled_suite"))
    parser.add_argument("--data-home", type=Path, default=Path("/tmp/ml-agent-20news"))
    parser.add_argument("--reuse-matrix", type=Path, help="recompute selection/evaluation from a previously executed matrix")
    parser.add_argument("--offline-text", action="store_true", help="use only for a deterministic local smoke test")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    candidates = candidate_pool()
    if args.reuse_matrix is not None:
        matrix = pd.read_csv(args.reuse_matrix)
        expected_rows = (TRAIN_TASKS + HELDOUT_TASKS) * CANDIDATES
        if len(matrix) != expected_rows or matrix["candidate_id"].nunique() != CANDIDATES or matrix["task"].nunique() != TRAIN_TASKS + HELDOUT_TASKS:
            raise ValueError("reuse matrix does not match the declared 19-task x 120-action protocol")
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        print(f"REUSED_EXECUTION_MATRIX={args.reuse_matrix} rows={len(matrix)}", flush=True)
    else:
        tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
        matrix = evaluate(tasks, candidates, output_dir)
    matrix.to_csv(output_dir / "execution_matrix.csv", index=False)

    selected_ids, diversity = select_diverse(matrix, candidates)
    generalization, generalization_summary = fit_ranker(matrix, selected_ids)
    generalization.to_csv(output_dir / "heldout_results.csv", index=False)
    rewards, reward_summary = reward_table(generalization, matrix)
    rewards.to_csv(output_dir / "reward_events.csv", index=False)
    make_figures(output_dir, diversity, generalization, rewards)

    task_manifest = [
        {
            "name": task.name,
            "split": task.split,
            "modality": task.modality,
            "samples": int(task.X.shape[0]),
            "features": int(task.X.shape[1]),
            "classes": int(np.unique(task.y).size),
        }
        for task in tasks
    ]
    result = {
        "protocol": {
            "label": "scaled_independent_reproduction",
            "paper": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
            "openreview_id": "kcPPWaoegr",
            "seed": SEED,
            "training_tasks": TRAIN_TASKS,
            "heldout_tasks": HELDOUT_TASKS,
            "candidate_actions": CANDIDATES,
            "selected_actions": SELECTED,
            "random_subsets": 5000,
            "offline_text_smoke": bool(args.offline_text),
            "scope_boundary": "offline pipeline-action environment; no unreleased ML-Agent checkpoint, trajectories, or PPO weights",
        },
        "task_manifest": task_manifest,
        "diversity": diversity,
        "generalization": generalization_summary,
        "reward": reward_summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    checksums = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"CHECKSUMS.json", "execution_matrix.partial.csv"}
    }
    (output_dir / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")

    print(f"RESULTS_JSON={result_path}")
    print(f"TRAIN_TASKS={TRAIN_TASKS}")
    print(f"HELDOUT_TASKS={HELDOUT_TASKS}")
    print(f"CANDIDATES={CANDIDATES}")
    print(f"SELECTED={SELECTED}")
    print(f"DIVERSITY_P={diversity['empirical_p_ge_selected']:.8f}")
    print(f"AGENT_HELDOUT_MEAN={generalization_summary['agent_mean']:.6f}")
    print(f"GLOBAL_HELDOUT_MEAN={generalization_summary['global_mean']:.6f}")
    print(f"AGENT_MINUS_GLOBAL={generalization_summary['agent_minus_global_mean']:.6f}")
    print(f"WALL_SECONDS={result['environment']['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()

````


````output
REUSED_EXECUTION_MATRIX=outputs/scaled_suite/execution_matrix.csv rows=2280
RESULTS_JSON=outputs/scaled_suite_corrected/results.json
TRAIN_TASKS=9
HELDOUT_TASKS=10
CANDIDATES=120
SELECTED=10
DIVERSITY_P=0.00019996
AGENT_HELDOUT_MEAN=0.895810
GLOBAL_HELDOUT_MEAN=0.902477
AGENT_MINUS_GLOBAL=-0.006667
WALL_SECONDS=5.356

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_8675e7aacb38", "created_at": "2026-07-17T19:36:51+00:00", "title": "Artifact: execution_matrix.csv", "path": "outputs/scaled_suite_corrected/execution_matrix.csv", "size": 556443, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/execution_matrix.csv` · dataset · 0.6 MB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/execution_matrix.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_3c1eff6098c3", "created_at": "2026-07-17T19:36:51+00:00", "title": "Artifact: heldout_results.csv", "path": "outputs/scaled_suite_corrected/heldout_results.csv", "size": 1860, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/heldout_results.csv` · dataset · 1.9 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/heldout_results.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_016819e0a2c9", "created_at": "2026-07-17T19:36:51+00:00", "title": "Artifact: reward_events.csv", "path": "outputs/scaled_suite_corrected/reward_events.csv", "size": 1697, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite_corrected/reward_events.csv` · dataset · 1.7 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite_corrected/reward_events.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e4331a536cb7", "created_at": "2026-07-17T19:38:08+00:00", "title": "Claim 4 verdict: mechanism reproduced at scaled action level"}
-->
Exact claim: Exploration-enriched fine-tuning generates over 100 diverse initial ideas and selects the 10 most behaviorally distant ones before subsequent RL training (Section 4).

Verdict: PARTIAL/PROXY. The run generated 120 executable pipeline actions and measured each on nine training tasks. Their nine-dimensional validation-score vectors define behavior; no text embeddings or paper values enter the distance calculation. Deterministic maximin farthest-point sampling selected ten. Their mean pairwise standardized distance was 5.847667 versus 3.694899 across 5,000 random ten-action subsets (random p95 4.828127; maximum 5.657392; empirical p=0.00019996). The original first attempt—ranking distance to the whole pool—selected a tight KNN cluster and failed (2.476498; p=0.9332); that negative run is retained, followed by the reasoned maximin correction. This reproduces the selection mechanism at scaled action level, not the unreleased EFT idea corpus or subsequent RL.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_cfa64024f5c0", "created_at": "2026-07-17T19:38:23+00:00", "title": "Behavioral diversity and controls"}
-->
````html
<img src="data:image/png;base64,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gwcPUmseAAAAAFLoqaeechm+atas6csi8OKLL6YowK5v374WHx/vti+55BIbP368xcXF+Z7v16+ftW/f3qZOnerb/9dff+WzAwAAALKRl156ybddp04dV7VIWbmC2bt3r6tolCtXrnRqIaLFW2ClsWagbGsAAMQKitADyHI0IVOuXDlfZrv169dndJMAAAAAINNSIJwy16W2hObvv/9uCxYscNuFChWyL7/8MkFwnRQpUsRlsdO9/Pbbb+4GAAAAIPtQ5muPxgehguu8MqKtW7e2HDlypEPrEE2HDh3yjREp7wsAiGVksAOQ5WgAVbFiRduyZYuvjFE4VqxY4QZtKjV74MABK168uJ133nku7bgGZ5FQKvIpU6a4yaMdO3ZYyZIlXVprlbcNZ4Cg18ycOdO1ReehLHxqT+3atd0xdLxAlNnh/fffd9tXXXWVtW3bNtn3+uGHH+zrr79227fffrs1bdrU95yySXz33Xe+wEWVbwrm8OHDLm37woULbfv27ZY3b16rUKGCm4yrW7duyDYMHz7cfvnlF7c9cOBAN8l28uRJdw0XLVrkroGyXLzxxhtRuU5p6d9//7UJEybY33//7f6uUqWKtWvXzl2LcAS75tH+bKP1/U/pZ/fXX3/Z9OnTbc2aNe799LqyZcu60mD67BJPtCZ3fZQ6ftKkSW6VokeZLIsVK5bg9VrteN9997m2vfvuu+6xVq1a2bXXXpvs9dQ5jRkzxm137NiRVPUAAAARUp/No/6p+n+BlC5d2u68804bOnSo73Xnn39+VK/3smXLXP9RZWkLFCjgxhBXX311RGM/jXt++uknW7lype3cudNNCGniT33o5CYAw6Vxltqp/uv+/fvttNNOc2OsCy+80NLavn37bNq0aS4wUudXsGBBX3+9Ro0aIV/bp08f18/XdVAWwlBUMljH15hJ2RI9H374oRtT6/09999/f5KsJJdeeqnddNNNYZ9Xao+rDBuzZ8929xrvHD161I09Ne5t3ry5Lzg0OSplpvHLP//84yYVNQbTvwldMx1H38vUUGCqxpAaj+XLl89dZ5WKSs82AAAApIT6JV7QlfrnZ599dqou5MSJE91NHn/8cdenDebHH3+0r776ym337t3bzjrrrJDHVj9Wv3NrfKFt9Z/Ur1X2b/XbVdo2OdE4hkrlemOTXbt2+cYmLVq0sKpVqyb7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rUSJEtaoUSPr0aOH1a1bN1XHPnnypH3++ec2atQoW758uR05csROO+00u+KKK9zxvfJFgSxdutR++OEHmzVrlm3YsMH+/fdfy5s3r1WtWtVatGhh3bp1s9Kl9WNgYB07drTvvvsuZPs08bRly5ZUnSMAAAAApLe3337b/v77bzfm0jjul19+sTVr1ljJkiXto48+snPPPTdVx+/evXuSx5QV77///a916dIlVccGAGQMBded2va/38FyFTezOD4JAMhKCLADAAAAgDSiwLdOnTrZ4cOHfY8pkE3BbR9//LG99tprdt9996Xo2Dt27LB27drZnDlzEjy+adMmW7hwob311lv25ZdfWsuWLZO89o033rAHHngg4HHXrl1rP/30k8vO8Omnn1qbNm0C7rdv3z7buXNnyDbmypUronMCAAAAgFigsZQWSfmrUaOGC65r3Lhxqo5du3Ztu+CCC9ziphw5crjx4cSJE93ipK5du7ox2XPPPRfRMXUcAAAAAGknZxoeGwAAAACyrV9//dXuuOMOF1x3/vnn2+TJk12mOJUCuuyyy+zEiRPWs2dPmzJlSorKE11//fUuuE5Z4vr16+cmZTZu3OiyytWvX9/27t3r9vnzzz+TvF5ZGKRUqVL2yCOPuImj9evX22+//WYDBw60YsWK2e7du+26665zx00u84IyOgS6bd26NeJzAwAAAICMdu+997pFR8ood/fdd1vNmjXd2KpJkyb24IMP+sZUkYiLi3NjLo2xhg8fbv3797enn37aRo8e7bLlKdO5PP/88zZv3rw0OCsAAAAAKUUGOwAAAABIAw8//LAdO3bMqlSpYj///LMVKVLEPX766afbpEmTXNYDZZp76KGH3ARLJBkHxo8f7wL15OWXX7ZevXr5nlNZ2EsuucRq1arlgubUDu3vL3/+/G4ip2/fvlawYEHf4xUrVrR69erZVVdd5TIqHD161F588UUbMWJE0LboWArUAwAAAICs4qabbkrw96lTp+zdd9+1Hj16uEzkZ5xxht1///0RHVNjL423AilfvryNGzfOqlevbvv377d33nnHLrzwwogWYYVChjsAAAAgdQiwM7N169a59NvJadGihVulBAAAAAChKLBt2rRpbvvRRx/1Bdd58uTJ47IVXH311bZ8+XKbP39+RJMnY8eOdfdly5Z1EzyBJm70vio/++23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AwAAABC7ateubblz57a//vrL91i9evWsaNGiLkvdBx98YHfddZcv4O7TTz912+ecc06GtRkAAAAAAAAAAACxJ0MC7AoUKBDR44HcdNNNUWwRAAAAgKykWLFidskll9jUqVPtl19+sYsuusiVi7333nvdop27777b3n33XffYwoUL7dixY25RzQ033JDRTQcAAAAAAAAAAEAMyXQlYvPnz2+dO3d2meUAAAAAIJhHH33U6tatawcOHPA91r9/f1uxYoWNGzfOfv31V9/jJUuWtDFjxliRIkW4oAAAAAAAAAAAAMjYALuZM2cm+LtZs2YBH/eXK1cui4uLc6VhlWUiPTNfVKhQwcqUKZNu7wkAAAAg9Vq0aOFu/jSW+Oabb9zYQ7eDBw/aGWecYddff70rHwsAAAAAAAAAAABkeIBd06ZNE/xduXLlgI/HgieffNLdAAAAAGQe27Zts7///tstlFEAXWJa5OMt9AEAAAAAAACA1Nj34Zt2at8et130VDEz68EFBYAsJKfFgHXr1rkbAAAAAETD2LFjrXHjxjZo0CAuKAAAAAAAAIA0dWTOz3bkp0nuVuD3n7naAJDFZEgGu0idOHHCli5daqdOnbJzzz3X8uTJk9FNAgAAABDDChUq5O7j4+MzuikAAAAAAAAAAADIxGIig93+/fvt0UcftRdeeCHJcwsXLrRq1apZvXr17IILLnDlZKdNm5Yh7QQAAACQOVSqVMndb9myJaObAgAAAAAAAAAAgEwsJgLsRo8ebS+99JKtXbs2weNHjx616667zjZs2OB7bPPmzdauXTvbtGlTBrQUAAAAQGbQqFEjK168uFucs2vXroxuDgAAAAAAAAAAADKpmAmwkxtvvDHB41999ZULritbtqzNnj3bVq5caeedd57LePfqq69mUGsBAAAAxLq8efO6RTwaO3To0MF27NiR0U0CAAAAAAAAAABAJpTbYsDq1avd/Zlnnpng8YkTJ7r7Xr16WePGjd32gAED7KqrrrIffvghA1oKAAAAIDP47bffbOvWrVazZk376aefrEqVKtaqVSurUaOGFShQIOjrLrjgAmvdunW6thUAAAAAAAAAAACxKyYC7Lxyr+XLl0/w+Jw5c9y9/wRXw4YN3f3ff/+drm0EAAAAkHnMnz/fnnrqKd/fBw8etK+//jrZ13Xv3p0AOwAAAAAAAAAA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7Z9//rEOHTq4cdx///vfDGotAAAAgOwkJgLs5s6d6+7vvffeZPctX768u9++fXuatwvA/2PvPqCkKtKGAb9EyZJFRBRzQsUsyJpQzFl3jWsOa1yza9xvzbprWtdVMWfFLGZFxYxZFMwJEAEVSSJp/lP1fTM/AzMjAzNMM/M859xz79xYfbunu6vrrbcAABZeZ511Vs5kN3369LzcoUOHWHfddWOrrbaKjTbaKNq3bx8HHnhg/PLLL7HEEktEv379arrIAAAAddbnn3+e58svv3yp9QMGDMjz1DkqBd+lRAxXXnllNGjQIAYNGhRTpkypkfICAAB1R0EE2P3888+lgueKpSFjZ1e/fkEUGQAAKHBNmjSJ5557Lvbff//c8PLbb7/FW2+9lYcbeuWVV2L8+PF5v759++ahhmavjwAAALDgjBgxIs9nr5u9+uqreZ6yjhfr1KlTLLnkkjFjxoyc0Q4AAKA6NYwC0Lp16xg7dmyMGjUqunTpUuG+X375ZZ537NhxAZUOAABYWLVo0SIPA5sy2KWsBx999FGMGzcur09ZEVJw3RprrFHTxQQAAKjz0pCvSXFnqNkD7DbYYINS61O9Lpk6dWqdv3cAAEAdCLBba6214umnn477778/jj322FIZ7IorVMXSPmVVpAAAAMrTrVu3OOqoo9wgAACAApWGfk0GDx4cW221VV5OWchTcoY2bdrEKqusUmr/tL44mx0A1LSON9xXsvzWZxFxRY0WB4AqVhDjre611155nrJKDBo0qNz9Pvzww7jkkkvy8r777rvAygcAAAAAAED16dOnT55ffvnl8eSTT8Ynn3wSf/3rX/O6bbbZJurX//9NWlOmTIkff/wxmjVrFh06dPC0AAAAtT/Abp999skZ6VLa70022SQHz91yyy0l21988cU4/fTTo2fPnjFx4sTYcsstY7vttqvRMgMAAAAAAFA19txzz1hhhRXip59+iq233jpWWmmlePnll6NRo0Zx0kknldr3hRdeyCMgpTYlAACAOhFg16BBg3j44YdzkN3MmTPj9ttvj/333z8vJ6mCdP755+fgul69esVdd91V00UGAAAWApMnT87ZD3r37h3t2rXLdY969eqVOx1++OE1XWQAAIA6qUmTJvH000/n4LqUrS7V0VZcccV45JFHYo011ii1b2pTSrbffvsaKi0AAFCXNIwC0bFjx3jppZfixhtvjH79+sW7774bM2bMyNtSJap79+5xyCGHxKGHHhqNGzeu6eICAAAFbuzYsbHpppvGkCFDarooAAAAzIWllloqHn/88TwEbGojat68eZn7nXDCCXHYYYfF8ssv774CAAB1J8AuSWm+U4UoTSnTxA8//JArUCn4rlWrVjVdPAAAYCFy5JFH5uC6Dh06xNlnnx2fffZZXHHFFTnDQdqW/r755pvj7bffjvbt28f1118fq6yySk0XGwAAoE768ssvY5lllinJZleR5ZZbLs8HDx6c249SpjsAAIBaPURsWZo1axbdunXLlSTBdQAAQGWMHj06+vfvn5fvv//+HFC30kor5b87d+4cffv2jaOOOirefPPN+Otf/5qz3V133XWxwgoruNEAAAA14OCDD4533nlnrvdPnaW23HLLmDBhQrWWCwDmxuiDdo9R2/fOU6d/7O6mAdQyBRtgN6vp06fHe++9lytW06ZNq+niAAAABS5lMZg5c2asscYa0bt373L3q1+/flx66aXRvXv3eOKJJyrVmAMAAEDVGT9+fGy99dbxxRdf/O6+7777bmyxxRYxbtw4TwEAAFA3AuxS76JTTz01zj///Dm2vfXWWzkleI8ePWLttdeOpZZaKgYOHFgj5QQAABaeDHbJrMMENWjQIM9/++23OYLsdt5557z8/PPPL9ByAgAA8L/WX3/9XJdLGcd/+OGHcm/LBx98kIPrfv7559h8881jtdVWcwsBAIDaH2B33333xUUXXRRfffVVqfWp4WuXXXaJ7777rmTd999/HzvuuGOMGDGiBkoKAAAsDJo1a5bniyyySMm6Fi1alAq+m1WbNm3yfOTIkQusjAAAAPx/V1xxRUkGu2222abMoV+HDBmSg+p+/PHH2GSTTeKRRx6JJk2auI0AAEDdCLBL9thjj1Lr+/fvn4PrFltssXjllVdi2LBheYinVKm6/PLLa6i0AABAoevSpUueDx8+vGRdt27d8vy9996LoqKiUvunukZSr169BVpOAAAA/lfDhg1ze9F6660X77zzTs40PnXq1JLb8/HHH+fgurFjx0bv3r3jscceK+lcBQAAUOsD7D7//PM8X3755UutT5Wj5Nhjj42ePXvm4Z0uuOCCvO6ZZ56pgZICAAALg1VXXTU3znz66acl63r06BGLLrpozlLXr1+/kvUp4O62227Ly4YWAgAAqDnNmzePAQMGxAorrBDPPfdc/PnPf84dpFKnqM022yxnJE/tRY8//njeFwAAoM4E2BUP99q5c+dS61999dU8TynBi6WeS0lKEQ4AAFCW1q1bx8Ybb5zrGq+99lrJcLFHHHFEXj700ENjnXXWiV69esX6668fkydPjg4dOsRuu+3mhgIAANSg9u3bx1NPPRWdOnWKu+++O/bff/8cXPfDDz/EBhtsEE8++WS0aNHCcwQAACwwDaMAFA/PNH78+FxxSlJD2Lfffpt7IHXv3r1k3+JK06xpwQEAAGZ36qmnxpprrhkTJ04sWXfOOefE0KFD4+GHH4633367ZH27du3igQceiJYtW7qRAAAANWzppZeOJ554InecuvXWW/O6ddddNwfXqbcBAAB1MsCua9eueeimwYMHl2Sre/TRR/M8ZZNo0KBByb6jRo3K89RzCQAAoDx9+vTJ06xSFruHHnooBg0alKdJkybFsssuG7vuumsePhYAAIDCkDpMpfrbVlttFauttlo8/fTT6m0AAEDdDbBLjV4pwO7kk0+Ojh07xvTp0+P888/P27bffvtS+w4fPjzPl1pqqRopKwAAsPDr3bt3ngAAAFjw0lCvb7311lztO2PGjHjvvfdKRkCa3RtvvBFrr712FZcQAACgwALsTjzxxJzie8iQIbHOOuuUrE9Z6g488MBS+w4cODDPN9100wVeTgAAAAAAAOZPSrSQAufm1syZM8vdVlRU5OkAAABqf4Bdt27dcmrvo446Kt555508JGzqvXTttddGq1atSu378MMPl5nZDgAAAAAAgMJ31VVXxS+//FIl51phhRWq5DwAAAAFHWCXbLjhhvH222/HxIkTo3Hjxnkqyw033JB7Kq2xxhoLvIwAAAAAAADMf5sQAADAwqJgAuyKtWjRosLtq6+++gIrCwAAAAAAAABARRq061CyPKN5h4jyRzcHYCFUcAF2AAAAAAAAMLthw4bloWVXWWWVaNmypRsEQMFod/F/Spbf+iwirqjR4gCwsAfYjRs3rmS5devWc6yrjOLjAQAAAAAAWLhdcMEFOYDu9NNPLxVA9/3338dOO+0Ub775Zv67WbNmcdlll8Whhx5ag6UFAADqigUeYNemTZuS5aKiojnWVUbx8QAAAAAAACy8Pvzww/jb3/4Wa665Zlx44YWlth100EElwXXJ5MmT4/DDD49lllkm+vTpUwOlBQAA6hJDxP6f0aNHx1NPPZUraCNGjIixY8dGixYtYoUVVoi+ffvmqX79+jX7bAEAAAAAANRC9957b57vsccecwwL+8QTT0TDhg3jjjvuiA022CBOPPHEuO++++K8884TYAcAANS+ALt33313rtYtSKknVOoVVVZGvFRpu+KKK6JHjx7Rv3//3BsKAAAAAACAqvP555/n+fLLL19q/YABA/J8hx12KAm+u/LKK+OBBx6IQYMGxZQpU6JJkyaeCgAAoPYE2KXU3nOzbkFK2eqWXnrp2GKLLXJZllxyyWjevHmMHDkyHn744bj//vtzEOCWW26ZU5Q3bdq0RssLAAAAAABQm6TRhZLOnTuXWv/qq6/m+dZbb12yrlOnTrkt5+uvv45vvvkmVlxxxQVcWgAobcbY0VE0Y0ZebjCuQUR0dIsAahFDxEbk7HWXXnppmTdo7733zhnsjjvuuPjiiy9yT6nddtttQT9PAADAPJg5c2Y89dRT8eKLL+bGmpTZoKzM1cVSp5pDDz3UvQYAAFjAiutq48ePLzPALg0NO6sWLVrk+dSpUxdYGQGgPD+ecmTMHD0qL3do0ymi1X1uFkAtUhABdqnX0WabbRabbrppHoq1QYMU0b3gtG3btsLt+++/fw6wS7799tsFVCoAAGB+pIC6HXfcMd5+++25PqZ9+/ZuOgAAQA3o2rVrng8ePDi22mqrvPzWW2/FqFGjok2bNrHKKquU2j+tL85mBwAAUOsD7J588sk8JYsuumj84Q9/yMF2aVpjjTWiXr16NVq+4cOHlyyvsMIKNVoWAABg7vzxj38sCa5beeWVY/XVV//dALrevXu7vQAAADWgT58+ceedd8bll18e6667bnTr1i3++te/5m3bbLNN1K9fv2TflJ38xx9/jGbNmkWHDh08XwAAQO0PsEtDtD7//PO5J9Ivv/wSjz76aJ6Ks8ttsskmJQF3q6666gIt2/vvvx8HHHBAXk4VupRtDwAAKGzpe/wrr7ySl6+88so46qijarzjDgAAAOXbc88948ILL4xPP/20VFtMo0aN4qSTTiq17wsvvJCHlE3tRwAAAHUiwO68887L84kTJ8ZLL70UAwcOzAF37733Xvz000/xwAMP5Cnp2LFjScDd4YcfXuVlSUPVTp48OaZOnZrTi3///ffRtGnTOPDAA+Oyyy5b4MPXAgAAlffBBx/k+dprrx1HH320WwgAAFDgmjRpEk8//XQcccQR8dRTT+UAujSqUMpol0Y7mtXDDz+c59tvv32Vl+Ozzz6LDz/8MEaMGBE///xzHp42Xb9Xr15V1kaU2qCee+65+Pjjj+O3337Lw+OmoMJ27dpVyfkBAIBaGGBXrEWLFjnNd5qScePG5V5IKdguBd199NFHMXr06Lj33nvzVB0Bdm+++WZMmjSp1LoePXrElltuGa1atar0+WTJAACABS81xCSrrbaa2w8AALCQWGqppeLxxx/PQ8DOmDEjmjdvXuZ+J5xwQhx22GGx/PLLV9m1TzvttLjrrrvim2++KXP7EkssEZdccknOtDc/BgwYEIccckhO8DCrlKkvjfh09tlna1sCAIACU1ABdrNr3bp17LjjjrH00kvn6Z577skBcNUpBfKlSlsKsvv666+jf//+8eSTT8arr74aTzzxRNx0000qNgAAUOC6deuW57/88ksUgnfffTdefPHFGDNmTLRt2zY22GCD6NmzZ5XULVLHpJT54JNPPokJEyZEs2bNYrnllovNN988ZwAvz8033xxDhgyp8NwpO8NFF11Uo48PAACom9nsKpLqPFXtwQcfLAmuW3PNNWOjjTbKiRc+//zzePTRR3NGu7322ivXe4455ph5ukYKHkztXqkdavHFF49dd901J59IdbrBgwfH3//+9xg/fnz861//quJHBwAA1LoAu2HDhuWsdWlKGex+/PHHkm2NGzeO9dZbLw/lWh3WXXfdUn8fdNBBucdSqjTdcsstscUWW8Tee+9d6cwZ5dHgBAAAVW/DDTeMxRZbLHegSUF2iy66aI3c5pEjR8af//znePbZZ8use9xxxx3znHEhNcicddZZueElZXcoKzguZUW47LLLymyceuihh0qGVZrXALvqfHwAAAALUmqvSQFvqZ61+uqrl9r27bffxg477BDvv/9+nHzyybHbbrtF586dK3X+yZMnx6GHHprrcuuss04888wzOdHErFn5Uv0u1eH22GOP3HEJAAAoDAURYJcyxRUH1KUGsNRIM2uDTnFA3aabbpp7DKWMDAtSSvd93XXX5WC/FGRXmQA7AABgwWvYsGFcfvnl+bv8PvvsE3fffXe5QwtVlxTYlzrofPzxx3mon9QAs8oqq8RXX32V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23TTTXO9KA0JO3DgwFhyySXn6VzFnY+SH3/8sdz9fvrpp1KJKAAAWDik4LqD61hgbb9jI9ZZPuqMggmwS8OxFgfUpUrKqFGjSm1PAW/du3fPlZnNNtssZ1moasVBdWkCAAAWfi+//HLO5Pbdd9/Nse29996Lhx56KM4888w8pNA+++xTZdctzk6QhhCqSHHgWWUzIKSGpK233jo3+KShYVOg3KyNLykA7uijj45+/frlILwUWJiGkZ3VuuuuW+a5U2em0047LR9zyy235KFoUya8li1bLpDHJ/MCAADUTcVZ6FK2uGJrrrlmPPzww/HYY4/FIYccUqo+V7xfcRBeVUptVKktaujQoTm4LiWCSPN5lTp5Lb744jnBQ0UjNH3++eclyyuuuOI8Xw8AAKhaVZMGbj6lSklK5Z0qR3fddVdJcN1KK60URxxxRNx3330xevToeP/99+Pyyy/Pw8bquQMAAFTk7bffjr59+5YE162zzjrxl7/8JWduO/bYY3NjSYMGDXKwWspid/fdd1fZDS3OTlBRZoJZsxPMms1gbtx44405uC4Fw918881z1I9S5rh///vfufNQCrZL9ajKSvckmTZtWm68WpCPDwAAqHuKM2+nJAzFdtttt5wc4ZFHHonjjz8+d6JKnYB22mmn3DknJWZIgWtVaeTIkTnb+azBdVUxBG2qkybpMZRn0KBBJZ2UVl555fm+JgAAUIsy2H377bd5vswyy+RGruIsdYZnBQAA5tWRRx4ZkydPzvWM1JFnvfXWm2Ofr7/+Ovbdd9/cwHHMMcfkYVZTZoH5VZxp4Jtvvonp06dHw4YNK8xOUNnMBB988EGep8dWXmPSIossEuuvv348+uijubNSZc1aHxs3btwCfXwAAEDds/nmm0erVq1yHWbixIm5brbaaqvljlJXX311XHbZZXkqluohs/5dFVJWvNRGleoyKaguBftVRXBdkoIC02N7991389SjR49S21PnpltvvTUvb7PNNrlOBwAAFIaCCLBLGRdShWX2IYsAAADmxWeffRZvvPFGNG7cOJ566qlYbrnlytwvNZQ88cQTscoqq+RMd08++WTOkDC/iodfTdnj3nrrrdhggw3m2GfEiBHx5ZdflspkMLeKh15NwW0VKd7+e0O5lmXWYXXbtm27QB8fALXHPQPeqOkiUID+uO36NV0EoACl+lsa4WjGjBk5g1uxK6+8MtfdrrnmmlzHSIF1qQ5y3nnnxR/+8Icqu36qA6W2qjSEa7du3XJwXWWGhU0dky699NK8vPXWW0evXr1Kbd9zzz3j7LPPzkF8Bx54YDz77LPRrl27vC1l40sZ+lInsJSx76STTqqyxwXAgjF12EdRNG1qXm48qnFErOrWA9QiBRFgl7JJpJTeTz/9dE0XBQAAqAWKhzRNGRDKC64rlrIi7L333nHhhRfm46oiwC5l5E7Dtv7yyy/Rr1+/MgPQrr/++jxPjUM77rhjpc6//PLLl2SQSw1MKZPd7FLGh9deey0v/949KMu1115bMtzs7JkVqvvxAQAAdVPTpk3nWFe/fv048cQT85Q6EaW/01QdQ9Sm4Lpko402KqnTlCV1IkoZ6WYPsEtBf0nr1q3nCLBLjy3Vn1Lm9FT3XGmllWLnnXeO5s2bx/PPP1+SqTw9zrIysANQ2MZdck7MHD0qL7dt0ymi1X01XSQAaluAXeqlk3okAQAAVIUJEybMMcxpRYr3Gz9+fJVlXjj22GPjf/7nf+LGG2+MrbbaqlTg3qBBg+Kiiy7KywcddFC0b99+jnOkhpdhw4blbaeeemqpbelc//jHP3KWg3322Sfuv//+UkPF/vTTT/m8xUO7/vGPfyx1/Ouvv56PTYFxKTvCrFJWunS9Bx98sKR8s2aPqKrHBwAAUFmpA091SR2Yit12220V7pvqObMH2M2Nvn37xmOPPRaHHHJIzpg3axBf6tx0xhlnxN/+9rdKnxcAAKgDAXYpBfbo0aNzQ06qQAAAAMyP4iFN01Cxc+PTTz8tqZtUlVNOOSUPPzt48ODYY489YpNNNolVV101vvrqqzwUbepktMIKK5RkOJhd//798/C2yy677BwBdt27d48TTjghDz+UstSl86Tzd+3aNUaOHBkvvvhi/Pzzz3nfP/3pT3l4olm98MILcdppp8USSyyRy5SC81JGunRsGgbpxx9/LBkKNmX2q47HBwAAMKvTTz89Z5A7//zzy8zSPb/7/55zzjknZ8ibG7Nn+U7atGmTO0IVZ8CrKMgu1ZtSvWzo0KExderUWHLJJWPLLbfM9TIAAKDwFESA3dprr50bZlJK7LKGFgIAAKiM9ddfP2dme+WVV3KjRQr+Ks/XX38dt99+e17ecMMNq+xGp6xvKdDs6KOPjrvuuisHrqWp2DbbbJOzFcxrUN/FF1+cA+rOPffc3GEpZUGYVWqYOe6443IGhNmlx7nxxhvHyy+/HCNGjJhjexrO6Igjjoizzjqr3E5Q1f34AACAuiV1MHr77bfzEKlzEzBX2f1/z+wdmyor1cHKqn+VpUGDBrH55pvnCQAAKHwFEWB32GGH5QC7s88+Ox5//PFcsQAAAJhXKSPbjjvuGA899FDssMMOOdPbfvvtVypYbObMmTko7aijjspDwy6//PKx2WabVXkmvTvuuCMPl5qGTR0zZkzOapAC3JZbbrkKj01DBvXp0yfvX5YUQJiC2w4//PB4880383CyaWjc5s2b53OnIMPZh3YtloLrUuBhGko2dXQaPnx4znjXqlWrnHVunXXWiUUWWaRaHx8AAMD8KCoqKqkbAQAA1PoAu9Twdfzxx8e//vWv2HTTTfNQRT179pQKGwAAmGdXXnllHr40ZWhLnXpSnWO11VbLAWCTJk2Kjz76KAeYJU2bNo2bb745GjasnipSly5dYs8996zUMbvuuutc7deoUaPo1atXniorBchVRVDhvDw+AACA+fHDDz/keepkBAAAUOsD7GZtxEpZD9KU1K9fv8KeR9OnT18g5QMAABY+Sy65ZLz22mtx8MEHx9NPP52D6t5444059lt99dXjhhtuyFnbAAAAWDCmTp2aM4vPnpEurZ8yZUq5x6UM5Pfdd1/uTJXal5ZeeukFUl4AAKDuKogAuxkzZpS5ftaKFQAAwLwE2T311FPx4Ycf5iC7Tz75JDfGpAwHqRFmk002id69e7uxAAAAC1gayejtt9+eY31lsnPvsMMO0aRJkyouGQAAQAEG2D3xxBM1XQQAAKAW6969e54AAABY+KURkLbddtu45pprarooAABAHVAQAXZbbbVVTRcBAAAAAACABeSGG26ICRMmlPx98MEH56zj/fr1ixVXXLHcwLpmzZrFsssuGy1btvRcAQAAdSfADgAAAAAAgLpjjTXWKPX3CiusEFOmTIn11ltPBnIAAKCgCLADAAAWaq+++mrce++9eblXr16x++67l1pXGcXHAwAAsGA98sgjbjkAAFCQFniA3bhx40qWW7duPce6yig+HgAAqLs++OCDuOKKK/JyynaQAuRmXVcZxccDAAAAAABAjQTYtWnTpmS5qKhojnWVUXw8AABQd3Xr1i123HHHvLzmmmvOsa4yio8HAAAAAJhbTXpuEjPH/29ioTEzW0d85d4B1CaGiAUAABZqffv2zdPvrQMAAAAAqA6tDjqyZPnzzyKi8oNrAFDAFniA3bvvvjtX6wAAAAAAAAAAAKBOBdiVNeSSYZgAAICqNG7cuBg1alS0adMmFltssSrfHwAAAAAAgLqhfk0XAAAAoKrdfffdsfLKK8fZZ59dLfsDAAAAAABQNwiwAwAAAAAAAAAAgEIYIvb3DBs2LF5//fU8PNOUKVMq3Pecc85ZYOUCAABqr19//TXPGzduXNNFAQAAAAAWMlNeGRgz/+83xqbjmkbEpjVdJABqY4DdN998EwcccEAMHDhwro8RYAcAAFSFt99+O8/bt2/vhgIAAAAAlTL+xv/EzNGj8vKibTpFtBJgB1CbFESA3bhx42KTTTaJr7/+Oho1ahQ9evSIN998M2/r3bt3jB49Oj799NMoKirK69Zff/1o0qRJDZcaAAAoFG+88UY8+OCDJX+/9957JetPPfXUco+bNm1azqL9xBNP5L833HDDBVBaAAAAAAAAFhYFEWB35ZVX5uC6JZdcMl544YVYZpllol69ennbSy+9VJLh7q9//WtuNGvVqlVJAxgAAMC7774bF1100Rw3IgXaFQfb/Z5NN900Nt98czcTAAAAAACAwgqwe/TRR/P8jDPOyMF1ZVlqqaWif//+se2228aTTz4ZN9xwQxx66KELuKQAAEAh6tKlS6nguBEjRuTMdEsssUSstNJK5R7XuHHjWHzxxWPjjTeOPffcM+rXr7+ASgwAAAAAAMDCoCAC7NLwr0nfvn3n2DZjxoxo0KBBXk6NXWeddVYOsLvzzjsF2AEAANl2222Xp2L//e9/44gjjsjr0jIAAAAAAAAstAF2U6ZMyfNOnTqVyiQxderUmDBhQrRu3bpk/ZprrpnnQ4YMqYGSAgAAC4M03OtNN90UK664Yk0XBQAAAAAAgIVYQYx/VBxY9/PPP5es69ChQ55/9dVXpfZNAXfJ+PHjF2gZAQCAhUcKrNt///1jww03rOmiAAAAAAAAsBAriAC7pZdeOs+HDx8+R6a6xx57rNS+xX8XB+ABAAAAAAAAAABArQ2w69OnT54PHDiwZN3uu++e5xdccEFcdtll8fLLL8cVV1wRxx57bF7ft2/fGiotAACwsJg8eXJcfvnl0bt372jXrl00aNAg6tWrV+50+OGH13SRAQAAAAAAKCAFEWC388475/ldd91Vsm6fffaJXr16xa+//hrHH398bhA77rjjYuLEidG+ffs4++yza7DEAABAoRs7dmysv/768de//jV32Pnpp59i5syZNV0sAAAAAAAAFiIFEWC32mqrxYQJE2LQoEEl61JmiccffzxnrOvYsWNe17Rp09hpp53itddei6WWWqoGSwwAABS6I488MoYMGRIdOnSIf//73yXZsLfffvt48skn46qrroq11147r0udeB588MHcuQcAAAAAAAAKKsAuadGiRTRv3rzUulatWuXhnH744Yf47bff8vBOqdFrueWWq7FyAgAAhW/06NHRv3//vHz//ffnYLuVVlop/925c+fo27dvHHXUUfHmm2/mDHcp2911110XK6ywQg2XHAAAAAAAgEJSMAF2v6dx48Y1XQQAAGAhMXjw4Dwc7BprrBG9e/cud7/69evHpZdeGt27d48nnngi3nnnnQVaTgAAAAAAAApbw5ouAAAAQHVksEtWXHHFknUNGjTI85Qde/Ygu5133jk+/PDDeP7552OttdbyhAAAAAAAc635DrvHzEkT8/KIyS0i3nPzAGqTggqwKyoqylkj7rnnnnjrrbdKGsU6duwY66yzTvzpT3+KrbbaKurVq1fTRQUAAApYs2bN8nyRRRYpWdeiRYs8L65nzKpNmzZ5PnLkyAVWRgAAAACgdmi+4x4lyxM/CwF2ALVMwQTYDR8+PPbcc894+eWX59g2duzY+Pjjj+PWW2+NjTfeOO68887o3LlzjZQTAAAofF26dCmpZxTr1q1bnr/33nu5c8+sHXeGDRuW5zrzAAAAAAAAUHABdr/88ktssskm8cUXX+S/119//ejTp0+pRrFnn3023njjjXjxxRfzvinDXatWrWq45AAAQCFaddVVo2HDhvHpp5+WrOvRo0csuuiiOUtdv3794pBDDikJuLvtttvy8mqrrVZjZQYAAAAAAKDw1I8CcP755+fguubNm8fDDz8cr7/+epx77rlx+OGH5yktp3VpW9rns88+iwsuuKCmiw0AABSo1q1b5+zXI0aMiNdee61kuNgjjjgiLx966KGxzjrrRK9evXIHn8mTJ0eHDh1it912q+GSAwAAAAAAUEgKIsDu/vvvz/OLLroodthhh3L3S9suvPDCvNy/f/8FVj4AAGDhc+qpp8YJJ5wQEydOLFl3zjnnxI477piX33777Xj11Vdj6tSp0a5du3jggQeiZcuWNVhiAAAAAAAACk1BDBH73Xff5fkee+zxu/umfY4++uiSYwAAAMrSp0+fPM0qZbF76KGHYtCgQXmaNGlSLLvssrHrrrvm4WMBAAAAACpr0sP3xsxJ/9vRt8XkFimywU0EqEUKIsCuRYsW8dNPP81VtohWrVrlucwSAADAvOrdu3eeAAAAAADm16RH7ouZo0fl5RZtOkW0EmAHUJsUxBCxa665Zp4PHjz4d/d9880387xHjx7VXi4AAAAAAAAAAADqroIIsDvssMPy/IQTTogJEyaUu1/alvaZ9RgAAAAAAAAAAACotUPE7rHHHvHqq6/GFVdckTPTnXrqqbHFFltEly5d8vbhw4fHM888ExdeeGF88cUXcdxxx8Wuu+5a08UGAAAKwOuvvx79+/evknNtuOGG6hoAAAAAAAAUVoBdw4b/vxgpgO6QQw6pcP+rrroqT2WZPn16lZcPAAAoXO+9917885//rJJzpUzZOvMAAAAAAABQUAF2M2bMqNb9AQCA2qtbt26x4447lrntgw8+iK+++iovt2zZMrp37x5t2rSJSZMmxccffxyjR4/O29q2bRu9e/eONddcc4GWHQAAAAAAgMJWEAF2TzzxRE0XAQAAWEj17ds3T7N75plnYvvtt4/WrVvHZZddFnvttVc0bty4ZHtRUVE89dRTceSRR8aXX34Zm2++eRx++OELuPQAAAAAAAAUsoIIsNtqq61quggAAEAtMnPmzDjkkENi2rRpMXDgwNhwww3n2KdevXq5LjJo0KCc2e7kk0+O3XffPTp16lQjZQYAAAAAAKDw1K/pAgAAAFS1N998M7755pvYeOONywyum1Xnzp1jv/32iylTpsQjjzziyQAAAAAAAKCEADsAAKDW+fzzz/N86aWXnqv9u3XrVuo4AAAAAAAASATYAQAAtU5RUVGef/vtt3O1f8p2Vzy0LAAAAAAAABQTYAcAANQ6q6yySp6/8MIL8c4771S475gxY+K2227Ly6uuuuoCKR8AAAAAAAALBwF2AABArbP22mvHWmutFTNmzIitttoq+vfvn5dnN2jQoNh4441zkF27du1i5513rpHyAgAAAAAAUJga1nQBAAAAqsMtt9xSEjy3++67R/v27WPNNdeM1q1bx6RJk+Kjjz4qGUK2UaNGcfPNN+dtAAAAAACV0erAv8TMX3/Ny1+NaxrxgvsHUJsIsAMAAGql1VZbLV5//fU47LDDYuDAgTF27Nh49tln59gvDQt77bXXRq9evaqtLD/99FO8+uqrOdivbdu2sd5668Xiiy9eJecuKiqKDz74ID755JOYMGFCNGvWLJZbbrno0aNHNGzY8HePHTZsWHz33XcxcuTIaNy4cXTr1i1n/1tkkUUqPPapp56Kr776qsJ96tevH4ceeug8PS4AAIAFLWU+f//99+OVV17JdbhU10rrUmetu+++e77OneqkG2200e/ut88++8QZZ5wxX9cCYMFr0mvTkuVfPwsBdgC1jAA7AACg1lp++eXj+eefjw8//DCefvrpHIQ2fvz4HIS2zDLL5Ax3vXv3rrbrT5w4MU466aS44YYbYtq0aaUCz/bYY4/497//nYemnVc33XRTnH322TlAbnYdOnSIU045JY4//vioV69eqW1vvfVWXHrppfHcc8/lRp7ZpSDAY445Jk477bQcdFeWa665Jh5++OEKy9egQQMBdgAAwEJh8uTJsdhii+V63OyqItv59OnTc53094waNWq+rwUAAFQtAXYAAECt17179zwtSFOnTo3tttsuXnzxxfz3uuuuG6usskrO+jZo0KCc/WDo0KHx0ksvRatWrSp9/gsvvDAHwCVNmjTJwYIpK97o0aPz+VO2vBNPPDG+/vrruOqqq0od+/LLL8c999xTMjzuBhtskDPXpWC7N998M8/POeecnLFhwIABFWbCS/e1Z8+e5QbYAQAALAxmzpwZv/76a84GnjKcp3pO6iyVOiZVtccffzzXwcrSpk2bKr8eAAAwfwTYAQAAVIN//etfJcF1N998c/z5z38uNbzq9ttvn4ceOuuss+Lyyy+v1Ll/+eWXHACXrL766vl8nTp1Ktn+888/x6677pqHxk1Z8o477rhYdtll52i0SQF4RxxxRKkGnDTM7F//+tfckJSy/qVMdUcffXS5Zdlss80qXX4AAIBC07x58xg3bly0aNGiZN39999fLddKwXUrrbRStZwbAACoevWjAHsIDR8+PD7++OMYMmRIhRMAAEAhSkP/XHLJJXl5n332KRVcl/Tt2zdOOOGEvJwC2FJAXGV89NFH8dtvv+Xlv//976WC65IUMPfPf/6z5O933nmn1PY0LG4amuhvf/vbHNkRWrZsGdddd11JY89dd91VqbIBAAAsjOrVq1cquA4AAKDgMtiNGDEiTj/99HjggQdyxoS5UVRUVO3lAgAAClsaxvTee+/Ny2kYn913373UusooPn5+vfDCC/HTTz/l5cMOO6zMfQ4//PA8zGsaSvbRRx+N/fbbb67P365du1JZFsrSrFmzMvdP1l577QrPX79+/Tzk7LBhw+Kbb76Z63IBAAAAQF00/oarY+b4cXl50ZmtI+LImi4SALUtwC5lrFtvvfXi+++/r+miAAAAC5kPPvggrrjiirw8ZcqUHCA367rKKD5+fr3xxht5vsgii8T6669f5j5LLbVULL300vH111/n/SsTYLf88stHjx494t13340rr7wyNt988xwUN6viDHbpOhtuuGGlH8PYsWPzvG3bthXuN2nSpHj22Wfju+++iyZNmuShjtZaa61o3Lhxpa8JAABQF/zP//xP7syUEk6krOKpfrfHHntEz549a7poAMyjKa++EDNHj8rLTdt0imglwA6gNimIALtzzjknB9elxpg0TNKOO+4YSy65ZDRsWBDFAwAAClgK6Ep1iGTNNdecY11lFB8/v9Lwq8XBbY0aNaowUC4F2KVMcZWRgunuvvvu2GmnneKxxx6LlVdeOXbZZZc8VOyYMWPyuvfffz/Xq+6///5o2rRppc6fzvHkk0/m5S222KLCffv165enWaUGomOPPTYPQVvR4wcAAKiL7rrrrlJ/v/TSS7mT2M477xw33XRTLLroojVWNgAAYE4FEcFW3HBz8cUXx9FHH13TxQEAABYiffv2zdPvrVuQfv755zxv3759hfsVD906btz/Dh9RGSussELOYHfqqafG5ZdfnoebndVee+0V//3vf6Nly5aVPvdf/vKXnJmuRYsWceKJJ5a7X7169WKVVVaJZZZZJj/WkSNHxiuvvJIff+pINXDgwHjqqadyJr+5lc4JAABQG3Xt2jX23HPPWHvttXPHsFT/GTJkSFx//fW5LvXggw/m+tRzzz03R5byiqhHAQBAHQiwGz16dJ7vtttuNV0UAACA+TZ58uQ8/73AsuLMcimYrbJGjBiRM9i99dZbJUPRpiFn0/o05Oydd96Z5w8//HCsuuqqlRqqqH///nk5Beh17ty5zP0OPPDAuPrqq2OJJZYotX78+PFx/PHHxw033BAvvvhinHnmmbkzFQAAQF3WoUOH+PLLL6NBgwal1qdgu/322y93bvrXv/4VL7zwQtxxxx2x77771lhZAQCA0ua++0s1at26dckwQgAAAAu7Jk2a5Pm0adMq3O+3334rtf/cSufdbLPNcnBdGtb2o48+ysFst9xySzz77LPx2Wef5Qx+X3zxRWy88cbx008/zdV5//3vf8fZZ5+dl9N87733LnffHXbYYY7guqRVq1Z5yNhtttkm/52C8Iof59woKiqqcAIAAFgYpcC62YPrZs1Ad9FFF+VOU8k999xTqXOrRwEAQB0IsEuZFpIPPvigposCAAAw31KQ2dwM/Vq8vXj/uXXrrbfGp59+mpfvuuuuWHbZZUtt79SpU26QSef98ccf8xCyv+c///lPHH300Xn5jDPOyEO8zo80zGxxNr933nlnvs4FAABQ2zVs2DB3pEqGDh1a08UBAAAKbYjYY489Nh577LH4xz/+EY888kjuqQMAADA3Xn311bj33nur5Gb16tUrdt999/k+z3LLLZfnX3/9dc4kUF4dJw0PlCy//PKVOv/gwYPzvGvXrrHSSiuVuc+iiy4aPXv2jCeffLJk//KkLHNHHXVUXj799NNz3Wx+devWrWR57Nix830+AACA2q5p06Z5PnXq1JouCgAAUGgBdn369IkzzzwzN+LsuOOOeTkNc9SoUaOaLhoAAFDgUibsK664okrONWXKlCoJsOvRo0eeT5w4MT788MNYffXV59gnZZb75JNPSu0/t4oz3/3e0LKNGzcutX9Z0r077rjj8vLf/va3OPfcc6MqjB49umS5shn6AAAA6qL33nsvz7t06VLTRQEAAAotwC6lvS726KOP5ilp0KBBhcdNnz692ssGAAAUtpQpLXXUqQqpo09V2HLLLXPwWwrYu+222+KSSy6ZY5/bb7+9JLvdDjvsUKnzL7XUUnn+xRdfxJgxY6JDhw5z7DNt2rR48803S+0/u3/+859x4oknlgwLWxWZ64rdeeedJfW9NdZYo8rOCwAAUBsNGDAgXnnllbzct2/fmi4OAABQaAF2M2bMqNR6AACAYqnhodAaH1q0aBEHHXRQHnr1qquuil133TU22GCDku2fffZZ/P3vf8/Lu+yySx7qdXaPP/54fPvtt3mo1z333LPUthSQd/HFF+c606GHHhp33XVXqWx2qTPSCSecEKNGjcp/lxWAmI4/5ZRT8nLKIv4///M/c/34Uua9VK5OnTqVuf3f//539OvXLy+nx966deu5PjcAAMDCaMSIEbH55pvn5VTXOuCAA0ptP/jgg2PDDTeMbbbZJtelUmer4ozjqf501lln5b/bt28fxxxzTA08AgAAoKAD7J544omaLgIAAECVSgF0KUjuq6++ik033TT23XffWHXVVfPfN998c/zyyy/RsWPHuPTSS8s8/sorr4ynnnoqll122TkC7Hr16hV//OMf45577omHHnooVlxxxZJAvZEjR8Zjjz0Ww4YNy/v+4Q9/yPvO6sYbbywJrltttdWic+fO8d///rfcx3LYYYeVNP4kDz74YJx++um5HN27d4/FF188B9yla6fHnIbtTZZeeum47LLL5uMuAgAALDhHHHFEDBw4sOTvVMdJ3n///VhppZVK1qd6VP/+/efIIp46IyU//vjjHOd+9tln44YbbsjLjRs3zkF2Kat5CsybOXNmXr/YYovlUZ7atm1bTY8QAABYaAPsttpqq5ouAgAAQJVq165dbkDZa6+94o033ojrr7++1PZVVlklZ55LQWjz4pZbbsmNL9dcc03OdHf55ZeX2p4C4tK1//Of/0T9+vVLbSsOgEuGDBmSG5EqkjItpKFei6288so5KG/QoEF5ml26Xspcl4IEy8tyBwAAUGi+++67kiC5WU2ZMqXU+pS1vLJuuummuP/++3PSia+//jrX44rrbqnT1O677x7HHXdcrksCAACFpSAC7ArFl19+GZ9++mnuLTR58uScTWK99daLbt261XTRAACAhdAyyywTr732Wrz00kt5GjNmTLRp0yYPC9SnT59SQWuz23bbbXPwXaqXlGWRRRaJK664Ik477bSc6S5lrJswYUI0b948lltuudhiiy3KDd5LmedSA9Hcmj1ALw05m6Z33nkn3nrrrRg+fHj8/PPP0apVq1hhhRVis802iyWXXHKuzw8AAFAIUmbviRMn/u5+TZs2nWPdEkssEUOHDs3LZdXjUmbzNCXTp0+PUaNGxYwZM/KQsKkeBwAAFK46H2CXUnYfddRR8cwzz+Shmsqy8cYbx1VXXZWHPgIAABYeqePMddddl7MEfPzxxzFu3LiSoXfKGwq1oqFS50XKRpDqFGmqjKOPPnqu9ksZ4v785z9X6twpM0Ka5tdaa62VJwAAgNqgS5cu83xso0aNSg0jW5HU2Wp+rgVA4Wl90jlRNG1qXv5kVOOIh2u6RADU6gC7lHXh9ddfzz13fi+jwjnnnDPf1/vtt99yg1uxNddcM3r06JErN2mopJRt4sUXX8wZJtLwThtssMF8XxMAAKh+Y8eOzdkB0vd6AAAAAIDq0nilVUuWpzZxnwFqm4IJsPvmm2/igAMOiIEDB871MVURYFecUSJlfDj11FNjxRVXLLUtZbbbbbfdYvz48XHggQfmrBcAAEDhO/LII3NwXYcOHeLss8+Ozz77LA+puv322+dt6e+bb7453n777Twkz/XXXx+rrLJKTRcbAAAAAACAAlI/CkAapmmTTTbJwXUphfZ6661Xsq1379456C0FwRVbf/31Kz28UnkaN26cs9TddNNNcwTXJVtssUVJIN/QoUPjww8/rJLrAgAA1Wf06NHRv3//vJyGh00BdcVD9XTu3Dn69u0bRx11VLz55pvx17/+NWe7S5mtV1hhBU8LAAAAAAAAhRVgd+WVV8bXX38dSy65ZB4i9o033ijZ9tJLL+V1X331Vey88855XatWreK5556rsgC7FLBXkRTkV2z48OFVcl0AAKD6DB48OGbOnBlrrLFGqe/zs6tfv35ceuml0b1793jiiSfinXfe8bQAAAAAAABQWAF2jz76aJ6fccYZscwyy5S5z1JLLZUzUGy11VZ52NYbbrhhgZXvhx9+KFlOQ0cBAACFn8EumTVLdYMGDfL8t99+myPIrrgzz/PPP79AywkAAAAAAEBhaxgF4NNPP83zNEzT7GbMmFHSEJYavs4666x48skn484774xDDz10gZQvDR+bdOjQIXr06LFArgkAAMy7Zs2a5fkiiyxSsq5Fixalgu9m1aZNmzwfOXKk2w4AAAAAVMq4i8+OGT//lJfbNmwbEX93BwFqkYIIsJsyZUqed+rUqdTQrVOnTo0JEyZE69atS9avueaaeT5kyJAFUrYHH3ww7r///rx8zjnnRMOGlbtl9erVq6aSAQAA5enSpUueDx8+vGRdt27d8vy9996LoqKiUt/Vhw0blue+vwMAAAAAlTX1k49j5uhReblxm04RrdxDgNqkIIaILQ6s+/nnn0vWpWxxyVdffVVq3xRwl4wfP77ay5Ua3vbbb7+8vM0228QRRxxR7dcEAADm36qrrpo7xxRny05SNupFF100Z6nr169fqe/9t912W15ebbXV3H4AAAAAAAAKK8Bu6aWXniO7RHGmuscee6zUvsV/FwfgVZehQ4fmIWsnTpwY6623Xtx1113zlM0iZcaoaAIAAKpeyoK98cYbx4gRI+K1114rGS62uNPMoYceGuuss0706tUr1l9//Zg8eXKuY+y2226eDgAAAAAAAApriNg+ffrESy+9FAMHDsyNXMnuu+8eAwYMiAsuuCBatGgR6667brz99ttxxhln5O0p+K06g+s23XTTGD16dC7PU089Fa1ayeEKAAALk1NPPTV33EmdZoqdc845+fv+ww8/nOsXxdq1axcPPPBAtGzZsoZKCwAAAAAAQCEqiAC7nXfeOc4666ycJe6kk07K6/bZZ5+4/vrr45VXXonjjz++1P7t27ePs88+u1rK8vHHH8dmm20WP/zwQ6y99trxzDPP5OwXAADAwiV15EnTrFIWu4ceeigGDRqUp0mTJsWyyy4bu+66ax4+FgAAAAAAAAouwG611VaLCRMmlBqCtUGDBvH444+XBN6lbHJNmzbNmesuueSSWGqppaq8HB9++GFsvvnmMWbMmBxc9+yzzwquAwCAWqh37955AgAAAAAAgIIPsEvSMLCzS8OyXn755XmaOnVqNG7cuNqu//777+fsFmPHjs3DwspcBwAAAAAAAAAAULfVj4VEdQbXvfvuu3lY2BRct+666wquAwCAhdzNN98ce+yxRzz66KMxffr0mi4OAAAAAAAAC6mCyWBXLDV+vfXWW/Hee+/FTz/9lDPXnXPOOdV2vR9++CFnrkvXatSoUR6Ctl+/fuXuv8UWW8Qaa6xRbeUBAADm35QpU+K+++7LU4cOHWLPPfeMfffdN2erBgAAAAAAgIUywO6WW26JM888M7777rtS62cNsFt11VXjk08+iWHDhsVyyy0339ccM2ZMDq5Lpk2bFueee26F+19zzTUC7AAAoMD17Nkzttlmm3j66afzd/4rr7wyT6usskoOtNtnn32iS5cuNV1MAAAAAAAAClzBBNidfPLJcckll+TllEkuBc8NHTp0jv223Xbb+Pjjj+Oxxx6L4447br6v2759+zjhhBPmev8111xzvq8JAABUr9VXXz0GDBgQo0ePjrvuuituvfXWeOedd3Jd4rTTTovTTz89Ntlkk9hvv/1i1113jRYtWnhKAAAAAAAAKMwAu2eeeSYH19WrVy8H2p1xxhm5gSv9Pbs0hGva96mnnqqSALtOnTrFpZdeOt/nAQAACk/Hjh3j2GOPzVMKrrvtttvijjvuyFmzn3/++Tz95S9/iZ133jkH2/Xp0yfq169f08UGAAAAAACgQBREy9FVV12V56ecckpceOGFFWaP6Nq1a55/9NFHC6x8AADAwi8ND3vBBRfEN998E88991zsv//+0bJly5g8eXIOukudeaqiEw8AAAAAAAC1R0EE2L3++ut5fsQRR/zuvp07d87zMWPGVHu5AACA2idlyt5ss83ipptuih9++CGuvvrqaNasWd42derUmi4eAAAAALCQaXfR1dG+3715GnP01TVdHABq4xCxP//8c6nguVkbvoqKikqtM1wTAAAwv6ZNmxaPP/54HjL2sccei99++81NBQAAAADmSYP2HUuWZ4x3EwFqm4IIsGvdunWMHTs2Ro0aFV26dKlw3y+//DLPO3b8/x9QAAAAc5s9OwXV3XPPPfHjjz+WrG/fvn386U9/mqus2gAAAAAAANQdBRFgt9Zaa8XTTz8d999/fxx77LEVZrBL+yQbbLDBAi8nAACw8EmddG6//fYcWPf555+XrF9kkUViu+22i/322y+23nrraNSoUY2WEwAAAAAAgMJTEAF2e+21Vw6wO+uss3KwXe/evcvc78MPP4xLLrkkL++7774LuJQAAMDC4pdffom77747B9W98sorpbb17NkzB9Xtscce0aZNmxorIwAAAAAAAIWvIALs9tlnn/jvf/+bh2vaZJNNcsBdnz59Sra/+OKLOQDvyiuvjIkTJ8aWW26ZM00AAACU5a677io13OsyyyyT6x0psG7ZZZd10wAAAAAAAFh4AuwaNGgQDz/8cOy44445yC4N35SmYinorlivXr1yYxkAAEBFWrduHbvvvnsOqttoo43cLAAAAACgWvx48l9ixo9j8nKH5h0i4j/uNEAtUj8KRMeOHeOll17KmezWWWedHHRXrF69erH66qvHVVddFc8//3y0bdu2RssKAAAUtp133jlGjRoV1113neA6AAAAAKBapeC6maNH5anBuP8NtAOg9iiIDHbFGjVqFIcddlieJk+eHD/88EPMmDEjB9+1atWqposHAAAsJBZbbLGaLgIAAAAAAAC1QEEF2M2qWbNm0a1bt5ouBgAAAAAAAAAAAHVUwQwRCwAAAAAAAAAAAIWkoDLYffrpp/Hggw/G0KFDY/z48TFz5swK93/ooYcWWNkAAAAAAAAAAACoWwoiwK6oqChOPvnk+Oc//5mXAQAAAAAAAAAAoKYVRIDd1VdfHZdeemlebteuXfTq1Ss6d+4cDRo0qOmiAQAAAAAAAAAAUEcVRIDdf//73zzfdttt4+67744WLVrUdJEAAAAAAAAAAACo4+pHAfjss8/y/JJLLhFcBwAAAAAAAAAAQEEoiAC74ox1yyyzTE0XBQAAAAAAAAAAAAonwK579+55/u2339Z0UQAAAAAAAAAAAKBwAuwOPvjgPP/3v/9d00UBAAAAAAAAAACArGEUgH322ScGDBgQV155ZTRv3jxOPvnkaN26dU0XCwAAAAAAAAAAgDqsIALskptvvjmGDRsWF1xwQVxyySWx1FJLRZMmTSo8ZsiQIQusfAAAAAAAAAAAs+t4w30ly299FhFXuEcAtUlBBNhNmjQpNttss3jvvffy39OnT48vvviiposFAAAAAAAAAABAHVYQAXbnnntuvPnmm3l5zTXXjC233DI6d+4cDRo0qOmiAQAAAAAAAAAAUEcVRIDdfff9b7rUww8/PK655pqaLg4AAAAAAAAAAABE/UK4B8OHD8/zk08+uaaLAgAAAAAAAAAAAIUTYNeiRYs8X2KJJWq6KAAAAAAAAAAAAFA4AXZrr712ng8dOrSmiwIAAAAAAAAAMNdGH7R7jNq+d546/WN3dw6glimIALsjjzwyz88999yaLgoAAAAAAAAAAAAUToDdDjvsECeddFL0798/dtlll3jrrbdi2rRpNV0sAAAAAAAAAAAA6rCGUQAaNvz/xXjwwQfzlDRo0KDC46ZPn17tZQMAAJhfM2bMiKFDh8aYMWOibdu2sfLKK0fjxo2r7MaOGzcuPv3005gwYUI0a9YslltuuejQocMCK191Pz4AAAAAAIA6ncEuNcakqbz15U0AAACFLNVbLrrooujcuXN07949Nttss1hzzTVjscUWi9NOOy1+++23+Tr/c889F717985Bbeuvv3706dMnevbsGR07dowePXrE/fffX63lq+7HBwAAUFPSSEupI9GQIUPiyy+/rPLzp/rUiBEj8rmnTJlS5ecHAABqWQa7J554oqaLAAAAUKWKiopi7733jnvuuSf/vfjii8cKK6wQ33zzTXz99ddx4YUXxuDBg3N9qFGjRpU+/y233BIHHHBAvk6y4oor5muMHj06hg0bFu+9917stttuccEFF8Spp55a5eWr7scHAACwIM2cOTPXX1555ZU8pfrMr7/+mrelDk2vv/56lVzn559/jjPPPDPuvPPOvFw80tMWW2yR61Grr756lVwHAACoZQF2W221VU0XAQAAoEpdf/31JcFn55xzTpxxxhnRoEGDHJjWr1+/OOyww3IGuhQAd9ZZZ1Xq3KmR55hjjsnn6tq1azz88MM5c1yxNFzs7rvvHh988EG+7n777ZezzFVl+arz8QEAACxokydPju22267k79RRqHnz5jFp0qQqu8b3338fvXr1iq+++ir/3aFDh3yN1FEpBfelOtQjjzwSffv2rbJrAgAAtWSIWAAAgNqW+eAf//hHXt52223j7LPPzsFnSb169eKQQw6JQw89NP99ySWXVLrB5v3334/x48fn5TRE66zBdUnKJHfNNdeUDDv02muvVWn5qvvxAQAALGipTpPqN+edd1688MIL8csvv1R5goh99tknB9e1atUqd5RKGcjT36mT1FprrRVTp06NP/7xjzFmzJgqvS4AADB/BNgBAABUsRTQNnz48LycMs2V5dhjj83ziRMnxuOPP16p87do0aJkebHFFitzn06dOpW5f1WUr7ofHwAAwILWtGnTeOyxx+Jvf/tbbLzxxvnvqvT888/nKbn66qtjhx12KNm23HLL5YC7lM0uBfZdfPHFVXptAABg/giwAwAAqGKvvvpqnjds2DB69+5d5j4rr7xyybCtxfvPrZVWWimWXXbZvHzbbbeVuc9NN92U5+3bt48NNtigSstX3Y8PAACgtrnrrrvyfIkllog999xzju1dunQpWX/33XdHUVHRAi8jAABQNgF2AAAAVWzo0KF53rVr1wqzHqShXJOPP/64UudPgW233357zl6XAun69OkT1157bc540K9fv9hpp53i3HPPzcMOpQC8RRddtErLV92PDwAAoLYZNGhQnm+22WZ5ONqy9O3bN89TxvA0dCwAAFAYGtZ0AQAAAGqbsWPHVjh8a7Hi7T/++GOlr5Gy0n300Ufxj3/8I6644op47rnnSm1PmQ8uueSSnB2hqstXnY+vXr16c70vAADAwmDq1Knx+eefl2T7Ls+s21J9b5llllkg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g/vvv78kuO7AAw/My61bt47JkyfHeeedF+eff37cfvvtscEGG+RscJWxzTbb5KkiK620Ug6Q23TTTWOZZZap0uNntd9++8Xll19eqfIDAAAUshkzZsRLL70UH3/8cfz222/RtWvX2HLLLXNnpXk1adKkuOyyy353v3XXXTf69u07z9cBoGY033GPkuWJn4UAO4BaRoAdAABANTj99NPzvFevXnH99ddH/fr189/NmjXLAXaffvpp9O/fP/7+97/HIYcckjPcVZVBgwbFJ598kpcPOuigBX48AADAwuqZZ56Jgw8+OL799ttS65s0aRJnn312nHLKKZXOQp5MmDAhzjzzzN/dL3XAEmAHAACFRYAdAABAFXv33XdLAtROOOGEkuC6WaVGmRRgN2bMmNyAs+2221bZ9fv161cyfOsuu+yywI8HAABYGD377LO5bjZt2rRo37597LTTTtGiRYt4/vnn44MPPojTTjstfvnll7jgggvm6zqHH354tGvXrsxtKcs5AABQWATYAQAAVLEXX3wxz1Ng3eabb17mPmuvvXZusBk7dmzev6oC7NKwrilwL9lnn31yloXqPv7XX3+NUaNG5X07dOgQDRuqagIAAAuXKVOmxIEHHpiD69Zcc80cbFccBFdUVBRHH310XH311XHRRRfFrrvuGuuss848X+vYY4+NlVZaqQpLDwAAVKc50ygAAAAwXz7++OM879KlS7Rq1arMfdKQQsUNKsX7V4U777wzJk+enJfTsEbVffwtt9ySH+MyyywTnTt3jtatW8d2220XL7zwwjyUHgAAoGbcdddd8d133+XlG2+8sVSGuVR/u+yyy2LppZfOwXYXX3yxpwkAAOoQAXYAAABVbPTo0Xm++OKLV7hfCkibdf+qcMMNN+R5yqaw+uqrV/vx48aNi2bNmkW3bt2iefPmMWnSpBgwYEBsuummcfrpp1f6+qnhqqIJAACgOjz44IN53qNHjzzNrlGjRrHffvvl5VTn+e233zwRAJSY9PC9MeHOG/PU4sV73RmAWsa4PQAAAFUsBZklTZs2rXC/FJiWTJgwoUqu+8EHH8Rbb701z9nrKnP8euutF3vssUdsueWWeajbZMaMGXm42+OPPz7ef//9OP/882PZZZfNwywBAAAUsuK60EYbbVTuPr17987zlPV76NCheSjZefHuu+/GwIEDc12wTZs2OaBvrbXWivr15cUAWFhNeuS+mDl6VF5u0aZTRKs9arpIAFQhAXYAAABVrLhRZObMmRXuN3369Dxv2LBqqmb9+vUrCdzbc889q/X4v/3tb3Osa9CgQWy22WYxaNCgWH/99XOD05lnnhkHHHDAXGefS8MtVUQWOwAAoKpNnDgxvv/++7ycOgmVZ7nllitZ/uSTT+Y5wG6vvfaaY126bhp6dpdddpmncwIAANVHVxgAAIAq1rJly5JGmooUby/ef36k4YnuuOOOvLz77rtHq1atFujxs0qP57TTTsvLI0eOjA8//HCezwUAAFDdxo0bV7Lcrl27cvdr27ZtyfIvv/wyT9fq0qVL7LDDDnHsscfGcccdF3369Mmdlb744ovYdddd49JLL630OVNHpIomAABg/shgBwAAUMW6du2a599++22F+xVvL95/fjzwwAPx008/zfPwsPN7/OzSEEfFvvvuu1h99dXn+5wAAADV4ddffy1Zbty4cbn7NWnSpGQ5DRNb2Y5Izz//fGy66aZzbPvss89it912iw8++CBOOeWUHHQ3r9nxAACAqieDHQAAQBUrDiYbO3ZsuUF2U6ZMiY8++igvd+/efb6vWTy860orrRQbbbTRAj9+XhuoAAAAatqsgXNTp04td79UjyvWtGnTSl2jefPmZQbXJcsvv3wMGDAgmjVrFjNnzoyrr766UucuKiqqcAIAAOaPADsAAIAqtuWWW5YMw/Pggw+WuU9qPEnDsibbbLPNfF3vq6++ioEDB+blgw46aIEfX5annnqqZHnFFVesknMCAABUh9atW5csF2f2LsvPP/9csrzoootWaRnS0LF9+/bNy6+99lqVnhsAAJg/AuwAAACqWOfOnWPrrbfOyxdddFHOZDd7drezzz47L6+11lqlhlMtNnz48Bg2bFh8+eWXv3u9G264IWclaNSoUey3336VLm9lj581O11Z3nrrrbj44ovz8gYbbFAlQ+ACAABUlzR8a6dOnfJyRXWwL774olo7EqUgu98L8gMAABY8AXYAAADVIAWYpSGDvv/+++jZs2fceeed8f7778dDDz0Uf/jDH/LwsA0bNozLL7+8zOMPPvjgWHnllXM2vIrMmDEjbr755ry8/fbbR8eOHStVznk5/oorrohVV101zjrrrLjvvvvi5Zdfjg8//DBnrTv22GPzELMTJkzIj//KK6+sVHkAAABqwtprr53nr7zySrn7DBo0KM9TXSfV16rayJEj87xNmzZVfm4AAGDeNZyPYwEAAChHCkDr379/7LXXXvHZZ5/F3nvvXWp7apC57rrronfv3vN1D5988skYMWJESVDegji+efPm8fHHH+eposwLt912W6y77rqVLhMAAMCCtuOOO8aAAQNyRu7Ugah79+6ltk+fPj1uvfXWvLzVVltFkyZNqvT6qXNWqp8l66+/fpWeGwAAmD8C7AAAAKrJNttsE0OHDs2BdC+99FKMGTMmZyLYcMMN45BDDolll1223GOXXHLJPOTQUkstVeE1Bg4cmPdr165d9O3bt9JlnJfjjz766DwE7iOPPJIbn9Jwtj///HO0atUqVlhhhejTp0/stttuscgii1S6PAAAADUhdY46++yzc6DbQQcdFM8880wsuuiiJdtPOeWUkuFjTzzxxDmO/+WXX+Kqq67Ky6lOtMEGG5TanoL3UgerVG+a3TfffBO77rprTJo0KerVqxeHH354NTxCAABgXgmwAwAAqEaLL754bqSprOuvv36u9rv00kvzNK/m9fjlllsujj/++Hm+LgAAQCFJmbpT56iddtopBg8enIeATR2H0vrnn38+3nzzzbzfMcccEz179pzj+NTp6Mwzz8zLKbvd7AF2Rx55ZIwaNSp69eoVXbt2jc6dO0dRUVEMGTIknnrqqZg6dWre75xzzon11ltvgTxmAABg7giwAwAAAAAAoM7bbrvt4oEHHojDDjssZ7IrzkiXNGrUKGex+/vf/z7PQ9DecccdOVivLCnD+XnnnRd//OMf6/zzAAAAhUaAHQAAAAAAAETEDjvsEFtttVUeInbo0KE5s9ySSy4ZW2+9dbRv377ce9S6des4/fTT8/KGG244x/Yrrrgi/vWvf+VMeJ9//nmMGDEiZsyYER06dIi11lor1l577Tw8LAAAUHgE2AEAAAAAAMD/ady4cWy77bZ5mlspwO7cc8+tcJ8GDRrk4LuyAvAAWLi1OvAvMfPXX/PyV+OaRrxQ0yUCoCoJsAMAAAAAAAAAmEdNem1asvzrZyHADqCWqV/TBQAAAAAAAAAAAIBCJMAOAAAAAAAAAAAAyiDADgAAAAAAAAAAAMogwA4AAAAAAAAAAADK0LCslQAAAAAAAAAA/L7xN1wdM8ePy8uLzmwdEUe6bQC1iAA7AAAAAAAAAIB5NOXVF2Lm6FF5uWmbThGtBNgB1CaGiAUAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDI0LGslAAAAAAAAAAC/r/VJ50TRtKl5+ZNRjSMedtcAahMBdgAAAAAAAAAA86jxSquWLE9t4jYC1DaGiAUAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyNCxrJQAAAAAAAAAAv2/cxWfHjJ9/ysttG7aNiL+7bQC1iAA7AAAAAAAAAIB5NPWTj2Pm6FF5uXGbThGt3EqA2sQQsQAAAAAAAAAAAFAGAXYAAAAAAAAAAABQBgF2AAAAAAAAAAAAUAYBdgAAAAAAAAAAAFAGAXYAAAAAAAAAAABQBgF2AAAAAAAAAAAAUAYBdgAAAAAAAAAAAFAGAXYAAAAAAAAAAABQBgF2AAAAAAAAAAAAUAYBdgAAAAAAAAAAAFAGAXYAAAAAAAAAAABQBgF2AAAAAAAAAAAAUIaGZa0EAAAAAAAAAOD3tbvo6iiaMSMvf/htg4hb3DWA2kSAHQAAAAAAAADAPGrQvmPJ8ozxbiNAbWOIWAAAAAAAAAAAACiDADsAAAAAAAAAAAAogwA7AAAAAAAAAAAAKIMAOwAAAAAAAAAAAChDw7JWAgAAUPVmzpwZ9evXr5LzTJ06da73b9SoUTRo0KDUumnTpsWMGTMqPK5evXqxyCKLLPDHBwAAAAALkx9P/kvM+HFMXu7QvENE/KemiwRAFdLyAQAAUI1effXV2HXXXaN9+/bRsGHDaNOmTWy99dbx5JNPzvM577333mjatOlcT7fddtsc59h9991/97jmzZvXyOMDAAAAgIVJCq6bOXpUnhqM+99AOwBqDwF2AAAA1eTKK6+M3r17xwMPPBA//vhjzgg3bty4HHyWgtBOP/30eTpvykaXMstVNM2asW7ttdeep3M1adKkRh4fAAAAAABAoRBgBwAAUA0GDhwYxx13XB42tW/fvvHRRx/lYVm//PLL2HvvvfM+559/ftxzzz2VPnfKPjdlypQKpzXWWCPvu+6660b37t3LPddRRx1V7jkmTpxYI48PAAAAAACgUAiwAwAAqAannHJKFBUVxWqrrRYPP/xwrLLKKlG/fv3o1q1bHrJ1s802K9kvBalVpffeey/eeeedvHzwwQdHbXt8AAAAAAAAC4oAu1mkxqGHHnoo9ttvv9hoo41i4403zo1Rzz777AJ7QgAAgIXfJ598EoMHDy4JMEvDrc4qDaV65pln5uVvvvkmXnrppSq9fr9+/fK8efPmseeee0Zte3wAAADV6bXXXosjjzwyNt100+jZs2f86U9/ijvvvDNmzJixUJwfAACoWg2r+HwLrQkTJsQuu+wyRzBdagi64YYbYq+99opbbrklGjZ0ywAAgIo999xzJYFmW2+9dZn79O7dO1q1ahXjx4/P9ZBNNtmkSm5rGto1NcwUDyXbsmXLuTouZZlLGegK/fEBAABUZyKGY445Jv7973/PERB3zz33xBVXXBGPPfZYdOjQoSDPDwAAVA8Z7P5P6h2UGn0aNGgQJ5xwQrz44ou50eiQQw7J21MD1bHHHltNTwMAAFCbfPTRR3m++OKLR7t27crcJ9U90rCqs+5fFe6///74+eef53p42AcffDC6dOmSOxOlTHQrrbRSHHXUUfHpp58W5OMDAACoLuecc05J8FvqsPTkk0/Gyy+/HH//+99zfenNN9+MHXfcMXdQKsTzAwAA1UM6tojcG+jxxx/PN+Saa64pCapLNttss+jcuXOu3Pz3v/+Nww47LFZfffVqejoAAIDaYOTIkXm+xBJLVLhfCmybdf+qHB42Bcr16tXrd/f/9ttv8zxlr5s6dWoe/jVN119/fVx55ZW5DrQgH1/KigcAALCgffPNN3HRRRfl5UMPPTSuvfbakm2pbtWjR4/YYYcdcra5W2+9Nfbff/+COj8AAFB9ZLCLyA1Hycorr1wquK7YaaedltNxpx5DxY1VAAAA5Zk0aVKeN2/evMKbVLx94sSJVXIzv/jii5yNOznooIMq3HfppZeO8847LwYPHhw//vhjzJgxI77//vu44YYbcuBcCrY74ogj4pFHHimYxwcAAFBdUlDbb7/9Fk2aNInzzz9/ju3bb799bLrppnn5uuuuK7jzAwAA1afOB9hNnz49nnnmmXwzdtlllzJvUkrLve222+blAQMGVOPTAQAA1CZFRUUVbq/qYX9ScFy6ZqNGjWK//farcN/LL788/va3v8U666wTbdu2zes6deoUBx54YLz11ls5+1w610knnbRAH186Z0UTAABAdSge6WjjjTeOdu3albnPrrvumuevv/56/PTTTwV1fgAAoPrU+QC7zz//PH799dd8M1LDUnnWXXfdPP/qq69kXwAAACrUokWLUpneylO8vWXLlvN9R1MGultuuSUvp2GFOnbsOM/nSoF2Kfgu+fTTT2PYsGE1/vgAAACqS+rMM2TIkLluK5p1/0I4PwAAUL3qfIDdN998U3IzunbtWu6NKt6WKjXffvttNT8tAADAwiwNsZoMHz68wv2KtxfvPz9SNoSRI0fm5YMPPni+z9ezZ8+S5dTRqKYfHwAAQHVJ2eImTpw4121Fyddff10w5wcAAKpXw6jjJkyYULJcUVaFWbfNeszvqVevXpXuB/zO/9LV7hAUgj/VdAGAiH18v6xJq622Wp6PGjUqfvjhh1hsscXm2Gf69Onx8ccfl9p/foeHTZZccsnYcsst5/t8sw7HWr9+/Rp/fLNTh6KQqIdQqK7+1zk1XQQokzojhape3FzTRaCWthVpi6q71tVmAXVY3ft91nse1C3r1rHvOXU+g11q9CnWoEGDcm9Uw4YNyzwGAABgdltssUXJ8mOPPVbmDXr++edLMhjMuv+8SIFuAwYMyMsHHHDAHAFx8+Kll14qWV5mmWVq9PEBAAAszG1F2qIAAGDhVucz2DVv3rzkZkyePLncGzXrthYtWsxT1gfqruLsGl4P4P8Q6jqfidQVKSAtDbH66quvxkUXXRR77rlnNGvWrGT7zJkz4+9//3teXm655UoNx1ps2rRpMWPGjBws17hx4wqvd8stt+QGm/Q/duCBB853+dNQROeff35eXmWVVWL55Zev8sc3r3ynLhze0ylEXpcUKq9NCpXXJiyYtqK63Bblfabu8tzXXZ77ustzX3d57uumenUsDqbOZ7Dr2LFjqawP5fn+++9Lljt06FDtTwwAALBwu+SSS3Lmg88++ywP2ZqC0X755Zd47733Yqeddsp/F+9XVsa57bffPpo2bZoD3H7PjTfemOd9+vSJpZZa6nf3/9e//hXbbrttPm7w4MExfPjwPGTRJ598Epdddlmss846eejXVK60b3U8PgAAgELRrl27ksx11dFWVN3nBwAAqledz2C38sorl9yMoUOH5gapsqRtSevWraNz587V/LQAAAALu5S17dprr40jjjgiXnnllejVq9ccvbtSlrgUjDa/Q7l++umnefnggw+eq2NSdrzHH388T+Vp2bJlLn/fvn1r9PEBAABUt5Q1fNlll811q+L2oLLMum3VVVctmPMDAADVq86nEWjVqlWsscYa+WY888wz5d6op59+Os979+5dzU8JAABQWxx00EHx1ltvxf7775+HVU1Ba127do0//vGPMWjQoDj11FMrbIBZZJFF8lSR2267Le+TOgLNbTDb8ccfn+s/Rx99dGy44Yax5JJL5uGH0jk22WSTOPfcc3PDTxr6tboeHwAAQCEpbv95/vnnY+bMmWXu89RTT+X5EkssketAhXR+AACg+tQrqiuD4VbgggsuiL/97W85PfeHH35YKqtdcWVn8803z8t33HFH7LXXXjVUUhZWdW3saShE/g+hMPhfBKg9vKdTiLwuKVRemxQqr034/1InpC233LLctqCRI0fGiiuuGBMnToy//vWv8a9//augzl+ovM/UXZ77ustzX3d57usuz33dVK+OxcHU+Qx2yVFHHRUdO3aMGTNm5IwPKciu2Kuvvhr77LNPXl5ttdVyJgYAAAAAAABqjy222KIky9yRRx4Zjz/+eMm2r776Knbccccc/Jayf59yyilzHJ8C5NZcc8083XrrrVV+fgAAoObIYPd/Xn755dxz6Ndff81/d+vWLQfcffvtt/nvdu3a5X1WWmmlmnu2WGjVtchdKET+D6Ew+F8EqD28p1OIvC4pVF6bFCqvTSht+PDh0bNnz/juu+/y34svvng0b948vvzyyzysa8OGDePBBx+M7bbbbo5b9/XXX+e2peSSSy6JE088sUrPv7DyPlN3ee7rLs993eW5r7s893VTvToWByOD3f/ZaKON4p133okddtghGjVqlHsLpeC6pk2b5jTd77//vuA6AAAAAACAWqpLly65reiQQw6JVq1axffffx+ff/55bjzs06dPvP766/MV/Fbd5wcAAKqHDHZlmDx5cowYMSIaNGgQSyyxRCyyyCLVdPsBAAAAAAAoNNOmTcttRVOnTo3OnTvnoVsrkvb7+OOP83JqW+rQoUOVnh8AABnobxkAAFIWSURBVKg5AuwAAAAAAAAAAACgDIaIBQAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwY6H2yCOPxJprrhnbbLNNTRelIO/LDjvsUNNFgejXr19+PR588MHuBhSgH3/8Mf+Ppum7776r6eIAAHXQoEGD8neRnj17Vul5H3zwwRr7zaC6HhO1i/oytc2pp56a3/vOO++8mi4KAAAAVKmGVXs6WLB++umneP/992PcuHEFcevvueeeuOCCC6Jbt275h/yavi8TJ06c53N8+umn8corr+QpnWvatGmVflxvv/123HrrrTFkyJCYPHlyLLnkkrlhY++9945GjRrNc9lYuIwaNSq/hlq3bl3TRQHKkN7f0/9o8ttvv9XKe/TUU0/F/fffH5988klMnTo1llhiiejbt2/ss88+0bRp03KPS4HBb731VoXnbtOmTQwcOLDMbelz+PLLL4/nnnsu39u11lorTjzxxFh66aXLPd/333+fA+TTZ2b//v2jfn39YQCKff3117HTTjvl5aeffjo6duzo5tQSv/zyS/4+0rx58yrvSFBTvxlU12OidlFfpjZ+Vqf3vg022KCmiwJUg88++yxuuumm/Lt/+q6z2GKLRZ8+feLPf/5ztGrVquDPz7xL7UR33HFHDB06NP+21rVr19hxxx1j9913jwYNGszzeT/88MN45pln8nzEiBH5e3v6rS0Fa//xj3/Mv6VRc4qKiuKBBx7I7YLpM75hw4axwgorxF577RWbbLJJlV/vv//9b56Szp07x+OPP17l12DuTJo0KW677bb8u3r6vbply5bRo0ePOOCAA2LllVeusmuk37/T7zup43/6HXzZZZfNv9un95dFFlnE01UDRo4cmT+L0/t++k2lffv20bt37zjwwAPn+3e4GTNmxEMPPRRPPvlk/sxP7SfptbX88svHtttuG9tvv732kBry+eefx6uvvpqf93fffTd/1qfvYek9oKp88MEHccstt5TEsKR2uq233jq30zVp0iQWGkWwELvpppuK0st4qaWWKioEV111VS7PiiuuWBD3Zdlll52n4xdbbLF8/OxTZR7XaaedVlSvXr0yz7P66qsXfffdd/NUNhY+//jHP/LzvvHGG9d0UYAyfP/99yXvz5999lmtukc///xzUd++fcv8LErTcsstVzRkyJByj0/vW+UdWzy1a9euzGMnTJiQP+9m379NmzZFH3zwQbnX3GWXXfJ+Tz75ZJXcA4DaZOjQoSXvp+oTtcujjz6an9fmzZtX6Xmvv/76GvvNoLoeE7WL+jK1zR//+Mf83nfYYYfVdFGAKnbNNdcULbLIImX+NtK1a9eid999t6DPz7yZOXNm0eGHH17u72Ibbrhh0dixY+fp3Ouss87v/u627777Fk2ZMsXTV0O/q/7hD38o97k54IADiqZPn15l1/viiy+KmjVrVnL+Qmn3rYvS7+XLLLNMmc97o0aNii6//PL5vkb67btLly7lvr5qup29rnrooYeKWrVqVeZz0rZt26Knn356ns/95ZdfFnXv3r3C9/y11167aPjw4VX6mPh9KZ6krOdjiSWWqNLfPho0aFDmdVZaaaX8GbCwkBIDmEOKSE+9UPbff/+4/vrrY7vttqvUXbrssstyJr/UuyUdO2DAgBzxfOGFF+be+ylCOWWyS9HPAFAd0mfQbrvtlnvYpN5vf/nLX3KP2Ndffz2uvvrq3Dsm9cpJPeLGjh1b4bkOO+yw3GunrOmFF14o85jzzz8/f96lbHX33Xdf7om36aabxs8//xyHHHJImcc8/PDDuVdoyvSaygUAAABQ0x555JH8u0rKzp8yVKaMQ6+99lr85z//iU6dOsW3336bM5CMGTOmIM/PvDvjjDNKMoqljHLpd7ZBgwbF2WefHY0bN87PU8oynn6Hq6z0G1mSMladeeaZOaNRakdK2W3WXXfdvC1l0EoZs1jw0u+qL730Us5ad8opp+TlZ599NrcbJinD1cknn1xl10u/l6aRsNJIWtSc9H+51VZbxZdffhlt27aNK664Ime1SlnHUtbCNBrOcccdF/fee+88XyO956dMZcOHD49VVlklrrzyynjxxRfz7/Z33XVX/m08ZTpjwUoj+aT3+fHjx+cshen9N73H33zzzTnDXBo9b+edd86ZTCtr5syZOSthylaa2mrS+3p6HaTnPL22UlbMJGWwTe89LPi4kPRZvO++++bP/JSdtipdf/31+XM+/V9vscUW+blPn/f//Oc/Y9FFF41hw4bl9530GbBQqOkIP5gfMthVTwa72XscHXTQQXPdY2D06NFFLVu2zPunXquze+GFF4rq16+ft1922WXzVD4WLnrkQ2GrrRns+vfvX/K4/vnPf86xPWU+Kv68OvLIIyvMYHf22WdX+vqpl2U69pVXXimV1S719ErrZ++RM378+NxrL21Pn6UAzEkGu9pLBjvqKvVlahsZ7KD2mTp1alG3bt3ybxkbbLBB/ntWn376aVGLFi0q/H2lJs/PvEu/XTVu3Djf+7/85S9zbH/ggQdKfnu77bbbKn3+jTbaqOjWW28tMwtaypy39957l5z//fffn+fHQeXdd999Jfc+PUezO+WUU/K2lI0o1dPn17XXXpvPt/322+f/cxnsas4JJ5yQ73+TJk3mGIUl/a9uttlmJZmtfv3110qff8SIEXmEl3SO3XffvWjatGll7jdx4sR5fgzMm169euXnJWUvHDduXKltY8aMKercuXPevu2221b63K+99lrJe0qqA5el+H0lTRWNOkTVmz0u5Nhjj62yDHbjxo0raRPbbrvtimbMmFFq+xtvvFHUsGHDCl8bhaZhTQf4UfukXkQpmjlFnqZx2dOYyV26dIlVV101RyCnTC7lSb0fHnzwwRypmsZeTmN5p2wvBx98cLRq1WqeyzQ/502ZaVL09EcffZSjthdffPFYccUVcwT96quvnveZMGFCHn+8OAPO119/HWuuuWap8xx77LFl9rSZ17L98MMPcc0118TLL7+cj0vlStluinuPzI927drN87F33313vh8NGjSIiy++eI7tG2+8cY5ST4/5uuuuyz0dKFvK+JfuZ3o9pKyAzz//fO4V8umnn8a4ceNy76A//elPed/Ro0fHo48+mnsTpF4fqSdBmzZtokePHjniPP3/zc01Uu+z22+/Pb8eU4bBFLGe/m9TxsGKfPXVV/n1OHjw4JgyZUosueSSOXth+j+ZG7/88kt+30j/b8XvGymL4q677lpuFqfZy56yQ6W/U0aq1GMu9XxJWafWWmutkmPSPUqZpFJ569WrF+utt14cf/zx0blz50q/DBfU9dO5UjR/OjadO93ftP+WW26Z/9/LG5e+pl8TdcEnn3ySe/Qk6bU7adKk/DpOPbrSZ2HqbXfPPffk7alnV+qFlXpipud/xIgRscgii8QyyyyT7+Uuu+yS3zd/7xrpOejXr19+XlKvkvR+vdlmm8Xhhx8eLVu2LLes6fqpbI899liMGjUqvxZSj+CDDjooGjVqNFeP94knnsjZ1T777LOS12F6naTXYVnXnr3s6bMhlT31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AAhAgB0AAAAAAAAAAAAAAAAAAAEIsAMAAAAAAAAAAAAAAAAAIAABdgAAAAAAAAAAAAAAAAAABCDADgAAAAAAAAAAAACADDZ//nxTr14906hRoxNq26Z3vd30TZs2Ncez7Dw+TpRtnJN8/vnnJ+T1ADhenJTdCwAAAAAAAAAAAAAAwPFmz549ZsWKFaZQoULmRJLe9XbTFylSxBzPsvP4OFG2cU6yd+/eE/J6ABwvyGAHAAAAAAAAAAAAAAAAAEAAMtgBAAAAAAAAAAAAAJDBmjVrZpYvX27y5MlzQm3bE3W9AQDHLwLsAAAAAAAAAAAAAADIYKeeeqqpV6/eCbddT9T1BgAcvwiwAwAAAAAAAAAAAAAgjj179pjx48ebefPmmR9++MGcfPLJply5cqZ69eqmc+fO9l+/+fPnm969e5uCBQuazz//PHCeO3fuNKNHjzaffvqpnX+pUqXMX/7yF9OjRw+bBU7Tn3LKKXZeftOmTTOPPfaYKVOmjJk5c6b97rhx48yqVavMoUOHzJlnnmluvPFGc8UVV3jTLFq0yLz66qtm7dq19ju1atWy8z/nnHNC1/nIkSPmzTffNLNnzzabN282uXLlMhUrVjStWrUynTp1MiedlDrkIJH13r17t11vbctff/3VW+9bbrklW/aVn7bTlClTzIoVK+x8SpYsaapVq2bXt2HDhqm+v3//frt9PvnkE/P999+bn376yRQuXNhu3/bt25vmzZuna10WLFhgJk+ebFavXm327t1rTj/9dJslUMeI/g6TWds4+thbuXKlee211+z20vF8zTXXmEceeSTFcui403L8+OOPdl/UqFHD7ocWLVqk2pZatz///NMedzVr1kzxufbLoEGD7N/33nuvueGGG1J8rmO0TZs29jj9+OOPTbFixbzPvvzyS/POO+/Y43/Lli3m6NGjpmzZsvb3brrpJhsYGqRp06Zm3759ZsiQIaZx48Z2XebMmWO2bdtmDh48aI93naOi40XbXMeC2+Zax4w6rgFkowgAAAAAAAAAAAAAAAi1fPnySOnSpSPqYg97jRs3LsU07777rn2/UKFCgfNcvHhxpESJEoHzKl++fGTIkCH27yJFiqSa9uWXX7afVaxYMfLMM89EcufOHTiffv362e/36dMn8PO8efNGZs6cGbh869ati1SvXj10fWvXrh357rvvUk0X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" alt="scaled_suite_summary" style="max-width:100%;height:auto;" />
````

````raw
{
  "diversity": {
    "candidate_mean_distances": {
      "knn-k1-distance": 5.663065519179529,
      "knn-k1-uniform": 5.663065519179529,
      "knn-k11-distance": 5.873474461538602,
      "knn-k11-uniform": 6.036819512472147,
      "knn-k15-distance": 6.054461821746522,
      "knn-k15-uniform": 6.2400898623592225,
      "knn-k21-distance": 5.891462375280381,
      "knn-k21-uniform": 6.438797327632664,
      "knn-k3-distance": 5.761358631992751,
      "knn-k3-uniform": 5.715456684760738,
      "knn-k31-distance": 5.775765953712313,
      "knn-k31-uniform": 6.207716663287248,
      "knn-k45-distance": 5.767314436192476,
      "knn-k45-uniform": 6.179846314189413,
      "knn-k5-distance": 5.933124469581267,
      "knn-k5-uniform": 5.861459314171153,
      "knn-k7-distance": 6.056149964835115,
      "knn-k7-uniform": 6.150465373833578,
      "knn-k9-distance": 6.0138466635320675,
      "knn-k9-uniform": 5.984327499385048,
      "linearsvc-c0.001-b0": 4.18231570314148,
      "linearsvc-c0.001-b1": 4.3384014775711215,
      "linearsvc-c0.0020691-b0": 3.1225993075639704,
      "linearsvc-c0.0020691-b1": 3.696988617058392,
      "linearsvc-c0.0042813-b0": 3.197350181691691,
      "linearsvc-c0.0042813-b1": 3.667527589204746,
      "linearsvc-c0.0088587-b0": 3.1063532185626097,
      "linearsvc-c0.0088587-b1": 3.2812707220353676,
      "linearsvc-c0.01833-b0": 3.282145422912684,
      "linearsvc-c0.01833-b1": 3.715982639514595,
      "linearsvc-c0.037927-b0": 3.9317038601829073,
      "linearsvc-c0.037927-b1": 3.632074067213309,
      "linearsvc-c0.078476-b0": 3.805783643798295,
      "linearsvc-c0.078476-b1": 3.803002229477938,
      "linearsvc-c0.16238-b0": 3.5086666686323396,
      "linearsvc-c0.16238-b1": 3.57291112416725,
      "linearsvc-c0.33598-b0": 3.3240293562921814,
      "linearsvc-c0.33598-b1": 3.3304460335764836,
      "linearsvc-c0.69519-b0": 3.191166900760959,
      "linearsvc-c0.69519-b1": 3.190262683817222,
      "linearsvc-c1.4384-b0": 2.790003122298278,
      "linearsvc-c1.4384-b1": 2.797290174473942,
      "linearsvc-c1000-b0": 2.931249395436089,
      "linearsvc-c1000-b1": 3.0142670628342976,
      "linearsvc-c112.88-b0": 2.9271585814977192,
      "linearsvc-c112.88-b1": 3.0138684150500477,
      "linearsvc-c12.743-b0": 3.2599821812364866,
      "linearsvc-c12.743-b1": 3.265954217059642,
      "linearsvc-c2.9764-b0": 2.9720566628976117,
      "linearsvc-c2.9764-b1": 3.078731418440136,
      "linearsvc-c233.57-b0": 2.9279174764347817,
      "linearsvc-c233.57-b1": 3.011516298493938,
      "linearsvc-c26.367-b0": 3.124207547343906,
      "linearsvc-c26.367-b1": 3.1267407281916793,
      "linearsvc-c483.29-b0": 2.925902653589594,
      "linearsvc-c483.29-b1": 3.0086810449091166,
      "linearsvc-c54.556-b0": 3.01102452832436,
      "linearsvc-c54.556-b1": 3.1267407281916793,
      "linearsvc-c6.1585-b0": 3.16311478960006,
      "linearsvc-c6.1585-b1": 3.0770319198572476,
      "logreg-c0.001-b0": 4.60719619113778,
      "logreg-c0.001-b1": 4.039511444286625,
      "logreg-c0.0020691-b0": 4.055858350487089,
      "logreg-c0.0020691-b1": 3.272215126054243,
      "logreg-c0.0042813-b0": 3.1635239416681418,
      "logreg-c0.0042813-b1": 2.7633596387151225,
      "logreg-c0.0088587-b0": 2.75026161697234,
      "logreg-c0.0088587-b1": 2.832338477776237,
      "logreg-c0.01833-b0": 2.8026258635316292,
      "logreg-c0.01833-b1": 2.9275197443718364,
      "logreg-c0.037927-b0": 2.6880122197052003,
      "logreg-c0.037927-b1": 3.0773304826137906,
      "logreg-c0.078476-b0": 2.878671139028318,
      "logreg-c0.078476-b1": 3.0489844251320752,
      "logreg-c0.16238-b0": 2.7580233051873893,
      "logreg-c0.16238-b1": 3.0794847357137227,
      "logreg-c0.33598-b0": 2.7643189913076056,
      "logreg-c0.33598-b1": 3.0835163947691937,
      "logreg-c0.69519-b0": 2.752317601836064,
      "logreg-c0.69519-b1": 2.7303289823747248,
      "logreg-c1.4384-b0": 2.73656352201972,
      "logreg-c1.4384-b1": 2.876537420916255,
      "logreg-c1000-b0": 2.8107074643615366,
      "logreg-c1000-b1": 2.803089192116411,
      "logreg-c112.88-b0": 2.667962677955079,
      "logreg-c112.88-b1": 2.7987168209351387,
      "logreg-c12.743-b0": 2.594526549561892,
      "logreg-c12.743-b1": 2.5967632208590867,
      "logreg-c2.9764-b0": 2.6294576581792533,
      "logreg-c2.9764-b1": 2.6244012891840924,
      "logreg-c233.57-b0": 2.670792531062626,
      "logreg-c233.57-b1": 2.803089192116411,
      "logreg-c26.367-b0": 2.6599553779446534,
      "logreg-c26.367-b1": 2.6637298226230035,
      "logreg-c483.29-b0": 2.670792531062626,
      "logreg-c483.29-b1": 2.803089192116411,
      "logreg-c54.556-b0": 2.6618265888506514,
      "logreg-c54.556-b1": 2.6618265888506514,
      "logreg-c6.1585-b0": 2.5793678709764305,
      "logreg-c6.1585-b1": 2.561141043072963,
      "sgd-hinge-a0.000215": 3.138731384166419,
      "sgd-hinge-a0.000774": 3.2254044941559425,
      "sgd-hinge-a0.00278": 2.796822616175897,
      "sgd-hinge-a0.01": 3.481241224115982,
      "sgd-hinge-a1.29e-06": 5.577922917392722,
      "sgd-hinge-a1.67e-05": 3.589043687538677,
      "sgd-hinge-a1e-07": 3.383875013279164,
      "sgd-hinge-a3.59e-07": 3.4418272016681595,
      "sgd-hinge-a4.64e-06": 3.443511038395062,
      "sgd-hinge-a5.99e-05": 3.337962299212947,
      "sgd-log_loss-a0.000215": 2.9221023567522773,
      "sgd-log_loss-a0.000774": 2.8737578938660704,
      "sgd-log_loss-a0.00278": 2.964150639797383,
      "sgd-log_loss-a0.01": 3.598071284893172,
      "sgd-log_loss-a1.29e-06": 4.731290140592902,
      "sgd-log_loss-a1.67e-05": 5.923319725103103,
      "sgd-log_loss-a1e-07": 3.6578349902460165,
      "sgd-log_loss-a3.59e-07": 3.448745005165697,
      "sgd-log_loss-a4.64e-06": 3.861498948402675,
      "sgd-log_loss-a5.99e-05": 3.325417089590722
    },
    "empirical_p_ge_selected": 0.0001999600079984003,
    "random_max": 5.6573919639484425,
    "random_mean": 3.6948990845312992,
    "random_p95": 4.828126579940782,
    "random_sd": 0.7429894804922337,
    "random_trials": 5000,
    "selected_ids": [
      "sgd-log_loss-a1.67e-05",
      "knn-k31-uniform",
      "sgd-log_loss-a0.01",
      "knn-k1-uniform",
      "logreg-c0.001-b0",
      "linearsvc-c0.001-b0",
      "sgd-log_loss-a1.29e-06",
      "knn-k15-distance",
      "linearsvc-c12.743-b1",
      "sgd-log_loss-a4.64e-06"
    ],
    "selected_mean_pairwise_distance": 5.847667329424862,
    "selection_method": "standardize per training task, then deterministic maximin farthest-point sampling"
  },
  "environment": {
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
    "python": "3.13.14 (main, Jun 10 2026, 12:24:04) [Clang 21.0.0 (clang-2100.0.123.102)]",
    "sklearn": "1.9.0",
    "wall_seconds": 5.355676625000342
  },
  "generalization": {
    "agent_mean": 0.895810359231412,
    "agent_minus_global_mean": -0.006666666666666665,
    "agent_ties_vs_global": 4,
    "agent_wins_vs_global": 2,
    "feature_columns": [
      "task_log_samples",
      "task_log_features",
      "task_classes",
      "task_imbalance",
      "task_sparse",
      "task_mod_tabular",
      "task_mod_image",
      "task_mod_text",
      "cand_fam_logreg",
      "cand_fam_linearsvc",
      "cand_fam_sgd",
      "cand_fam_knn",
      "cand_strength",
      "cand_balanced",
      "cand_variant_distance",
      "cand_variant_log"
    ],
    "global_best_candidate": "sgd-log_loss-a0.01",
    "global_mean": 0.9024770258980785,
    "heldout_tasks": 10,
    "modalities": [
      "image",
      "tabular",
      "text"
    ],
    "oracle_mean": 0.9132289055973265,
    "random_mean": 0.8484507414369256,
    "selected_candidates": 10,
    "training_tasks": 9
  },
  "protocol": {
    "candidate_actions": 120,
    "heldout_tasks": 10,
    "label": "scaled_independent_reproduction",
    "offline_text_smoke": false,
    "openreview_id": "kcPPWaoegr",
    "paper": "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
    "random_subsets": 5000,
    "scope_boundary": "offline pipeline-action environment; no unreleased ML-Agent checkpoint, trajectories, or PPO weights",
    "seed": 17897,
    "selected_actions": 10,
    "training_tasks": 9
  },
  "reward": {
    "actual_error_feedback": "ValueError: not-a-model",
    "corner_latency_budget_seconds": 0.1949065562498617,
    "error_rewards": [
      0.0
    ],
    "neutral_rewards": [
      0.5
    ],
    "successful_edit_max_reward": 1.0,
    "successful_edit_mean_reward": 0.4,
    "successful_edit_min_reward": 0.0,
    "successful_edits": 10
  },
  "task_manifest": [
    {
      "classes": 2,
      "features": 30,
      "modality": "tabular",
      "name": "train-breast-cancer",
      "samples": 569,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 64,
      "modality": "image",
      "name": "train-digits-low-high",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 10,
      "features": 64,
      "modality": "image",
      "name": "train-digits-multiclass",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 64,
      "modality": "image",
      "name": "train-digits-parity",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 3,
      "features": 4,
      "modality": "tabular",
      "name": "train-iris-multiclass",
      "samples": 150,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "train-text-autos-vs-motorcycles",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "train-text-comp-vs-space",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "train-text-politics",
      "samples": 1400,
      "split": "train"
    },
    {
      "classes": 3,
      "features": 13,
      "modality": "tabular",
      "name": "train-wine-multiclass",
      "samples": 178,
      "split": "train"
    },
    {
      "classes": 2,
      "features": 15,
      "modality": "tabular",
      "name": "heldout-cancer-half-features",
      "samples": 569,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 10,
      "modality": "tabular",
      "name": "heldout-diabetes-above-median",
      "samples": 442,
      "split": "heldout"
    },
    {
      "classes": 10,
      "features": 64,
      "modality": "image",
      "name": "heldout-digits-noisy",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 64,
      "modality": "image",
      "name": "heldout-digits-noisy-parity",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 10,
      "features": 64,
      "modality": "image",
      "name": "heldout-digits-shifted",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 4,
      "modality": "tabular",
      "name": "heldout-iris-setosa",
      "samples": 150,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "heldout-text-computer-systems",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "heldout-text-religion",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 2500,
      "modality": "text",
      "name": "heldout-text-science",
      "samples": 1400,
      "split": "heldout"
    },
    {
      "classes": 2,
      "features": 13,
      "modality": "tabular",
      "name": "heldout-wine-class-zero",
      "samples": 178,
      "split": "heldout"
    }
  ]
}
````
