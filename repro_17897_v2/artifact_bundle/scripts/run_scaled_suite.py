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
