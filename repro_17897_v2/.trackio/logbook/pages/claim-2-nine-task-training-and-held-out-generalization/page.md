# Claim 2: Nine-Task Training and Held-Out Generalization


---
<!-- trackio-cell
{"type": "code", "id": "cell_9d168dfb327b", "created_at": "2026-07-17T19:34:26+00:00", "title": "Full scaled execution: 2,280 fresh pipeline fits", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "scripts/run_scaled_suite.py", "--output-dir", "outputs/scaled_suite", "--data-home", "/tmp/ml-agent-20news"], "exit_code": 0, "duration_s": 89.227}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_scaled_suite.py --output-dir outputs/scaled_suite --data-home /tmp/ml-agent-20news
````

exit 0 · 89.2s


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
    order = np.argsort(-mean_distance)
    selected = pivot.index[order[:SELECTED]].tolist()
    selected_idx = order[:SELECTED]
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
    parser.add_argument("--offline-text", action="store_true", help="use only for a deterministic local smoke test")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tasks = load_tasks(offline_text=args.offline_text, data_home=args.data_home)
    candidates = candidate_pool()
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
PROGRESS=120/2280 last_task=train-breast-cancer
PROGRESS=240/2280 last_task=train-digits-low-high
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=360/2280 last_task=train-digits-multiclass
PROGRESS=480/2280 last_task=train-digits-parity
PROGRESS=600/2280 last_task=train-iris-multiclass
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklea
... [3460 chars elided] ...
rivate/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=840/2280 last_task=train-text-comp-vs-space
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=960/2280 last_task=train-text-politics
PROGRESS=1080/2280 last_task=train-wine-multiclass
PROGRESS=1200/2280 last_task=heldout-cancer-half-features
PROGRESS=1320/2280 last_task=heldout-diabetes-above-median
PROGRESS=1440/2280 last_task=heldout-digits-noisy
PROGRESS=1560/2280 last_task=heldout-digits-noisy-parity
PROGRESS=1680/2280 last_task=heldout-digits-shifted
PROGRESS=1800/2280 last_task=heldout-iris-setosa
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=1920/2280 last_task=heldout-text-computer-systems
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=2040/2280 last_task=heldout-text-religion
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
/private/tmp/uv-cache/environments-v2/run-scaled-suite-aad0a18318225258/lib/python3.13/site-packages/sklearn/svm/_base.py:1298: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn(
PROGRESS=2160/2280 last_task=heldout-text-science
PROGRESS=2280/2280 last_task=heldout-wine-class-zero
RESULTS_JSON=outputs/scaled_suite/results.json
TRAIN_TASKS=9
HELDOUT_TASKS=10
CANDIDATES=120
SELECTED=10
DIVERSITY_P=0.93321336
AGENT_HELDOUT_MEAN=0.794039
GLOBAL_HELDOUT_MEAN=0.805196
AGENT_MINUS_GLOBAL=-0.011157
WALL_SECONDS=88.109

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_b362aa65375f", "created_at": "2026-07-17T19:34:26+00:00", "title": "Artifact: execution_matrix.csv", "path": "outputs/scaled_suite/execution_matrix.csv", "size": 562538, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite/execution_matrix.csv` · dataset · 0.6 MB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite/execution_matrix.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_cb51bb040776", "created_at": "2026-07-17T19:34:26+00:00", "title": "Artifact: execution_matrix.partial.csv", "path": "outputs/scaled_suite/execution_matrix.partial.csv", "size": 562538, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite/execution_matrix.partial.csv` · dataset · 0.6 MB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite/execution_matrix.partial.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_3eb25bde01a7", "created_at": "2026-07-17T19:34:26+00:00", "title": "Artifact: heldout_results.csv", "path": "outputs/scaled_suite/heldout_results.csv", "size": 1713, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite/heldout_results.csv` · dataset · 1.7 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite/heldout_results.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_3411b332815c", "created_at": "2026-07-17T19:34:26+00:00", "title": "Artifact: reward_events.csv", "path": "outputs/scaled_suite/reward_events.csv", "size": 1698, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/scaled_suite/reward_events.csv` · dataset · 1.7 kB

https://huggingface.co/buckets/binzhango/repro-ml-agent-autonomous-ml-engineering-v2-artifacts#logbook-files/outputs/scaled_suite/reward_events.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_f1311fda093a", "created_at": "2026-07-17T19:37:13+00:00", "title": "Fail-closed verification of corrected evidence bundle", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "scripts/verify_scaled_suite.py", "--results", "outputs/scaled_suite_corrected/results.json"], "exit_code": 0, "duration_s": 0.045}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache uv run scripts/verify_scaled_suite.py --results outputs/scaled_suite_corrected/results.json
````

exit 0 · 0.0s


````python title=verify_scaled_suite.py
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

````


````json title=results.json
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


````output
PASS: exactly 9 training tasks
PASS: exactly 10 held-out tasks
PASS: at least 100 candidate actions
PASS: exactly 10 selected actions
PASS: selected-id manifest is complete
PASS: selected actions are unique
PASS: task manifest has 9 training rows
PASS: task manifest has 10 held-out rows
PASS: held-out tasks span image, tabular, and text
PASS: 5,000 random diversity controls
PASS: selected diversity is finite
PASS: selected set is more diverse than random mean
PASS: selected set exceeds random 95th percentile
PASS: ranker train/test counts are exact
PASS: ranker evaluation covers all modalities
PASS: agent accuracy is bounded
PASS: trained ranker beats random selected actions on held-out tasks
PASS: invalid/error trajectories receive reward 0
PASS: non-edit/corner outcomes receive reward 0.5
PASS: sigmoid reward evaluated on all held-out edits
PASS: success rewards are sigmoid bounded
PASS: error branch comes from an executed invalid action
PASS: evidence run uses real 20 Newsgroups text, not smoke corpus
ALL_VERIFICATIONS_PASS

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8dd93aad3a98", "created_at": "2026-07-17T19:38:08+00:00", "title": "Claim 2 verdict: scaled independent proxy"}
-->
Exact claim: ML-Agent is trained via exploration-enriched fine-tuning and step-wise reinforcement learning on only 9 ML engineering tasks (4 from MLAgentBench, 5 from MLE-bench), yet generalizes to 10 held-out MLE-bench tasks spanning image, text, and tabular modalities (Section 4).

Verdict: PARTIAL/PROXY. A fresh executable run trained a task-conditioned ExtraTrees action ranker only on exactly nine public classification tasks and evaluated it on exactly ten held-out tasks spanning image, text, and tabular data. It evaluated 120 pipeline actions on every task (2,280 fits). The ranker achieved 0.895810 mean held-out accuracy versus 0.848451 for random action choice, but 0.902477 for the single global-best training action. This supports cross-task transfer over random under the scaled protocol, while slightly falsifying improvement over the stronger global-best baseline. It is not ML-Agent's 7B EFT plus step-wise RL and does not use MLE-bench/MLAgentBench task instances.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_dd6b9b787ed5", "created_at": "2026-07-17T19:38:22+00:00", "title": "Held-out task results"}
-->
````html
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAACdgAAAL0CAYAAAAVnMcMAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAbrwAAG68BXhqRHAABAABJREFUeJzs3QeUU1X39/FN70MvglRRQBRBAZGiqCggCqJiF2mKCoryIHbE8oiCiAV7QRQsoCBFBRSRKk1BqQoCUqT3Xuddv/P8b97MTJJJZjIzmZnvZ62s3Elubs69CXpOzj5754iPj483AAAAAAAAAAAAAAAAAACQQM6EfwIAAAAAAAAAAAAAAAAAAALsAAAAAAAAAAAAAAAAAAAIggx2AAAAAAAAAAAAAAAAAAAEQIAdAAAAAAAAAAAAAAAAAAABEGAHAAAAAAAAAAAAAAAAAEAABNgBAAAAAAAAAAAAAAAAABAAAXYAAAAAAAAAAAAAAAAAAARAgB0AAAAAAAAAAAAAAAAAAAEQYAcAAAAAAAAAAAAAAAAAQAAE2AEAAAAAAAAAAAAAAAAAEAABdgAAAAAAAAAAAAAAAAAABECAHQAAAAAAAAAAAAAAAAAABNgBAAAAAAAAAAAAAAAAABAeMtgBAAAAAAAAAAAAAAAAABAAAXYAAAAAAAAAAAAAAAAAAARAgB0AAAAAAAAAAAAAAAAAAAEQYAcAAAAAAAAAAAAAAAAAQAAE2AEAAAAAAAAAAAAAAAAAEAABdgAAAAAAAAAAAAAAAAAABECAHQAAAAAAAAAAAAAAAAAAARBgBwAAAAAAAAAAAAAAAABAAATYAQAAAAAAAAAAAAAAAAAQAAF2AAAAAAAAAAAAAAAAAAAEQIAdAAAAAAAAAAAAAAAAAAABEGAHAAAAAAAAAAAAAAAAAAABdgAAAAAAAAAAAAAAAAAAhCd3mPsBAIIYPny4/fLLL2574MCBFhcXl2Wu1fjx4+27775z2/369bPy5ctbVpIR5/fGG2/YsmXLLE+ePG47FtqUmXB9AAAAEInff//d3n77bbd92223WbNmzTL8Aq5fv95eeOEFt92uXTtr3bp1io81c+ZMGzlypNt+4IEH7Oyzz45aOxGbXnnlFfvrr7+CjikBAACAzOjIkSP27bff2qJFi2z37t128uTJNJt3W7t2rb300ktuu3379tayZUvLTH7++Wf74osv3PaDDz5oNWvWzFLnh8iNGDHCZs2a5bYHDx5shQoV4jICaYAAOwCZgv+kQSB58+a1okWLuk7kJZdcYqeffnq6tW3atGkuyE769++fpQLs5s+fb++++67b7tmzZ5YL9sqI89MAcfLkyZYvX76AkyFZ/ZqnFtcHAAAgYyc8lixZ4iY8tGjk6NGj7vEbbrjBWrRoEfHx9Prvv//eHXPHjh1WqlQpO/fcc13AmfrL0aCJBa9/Xbdu3ZgIsNu2bZuvTRq7pibATp+Dd6xrr72WALtssuho+vTpQceUAAAAWdEff/xhc+fOtXXr1tn+/fvdPIzGD2XKlHH9fC00yZEjR0Y3Eymkz/b666+3f//9N8lzaTHvtnXrVt84qkqVKpkuAG3p0qW+9ms8njjALrOfH1IWdPnhhx+67eeff54AOyCNEGAHIFPwnzRITs6cOV0WgNdee80qVqyY5m0DEDtGjx5tU6dOddvPPPOMlS1bNqObBAAAkOnpx9kvv/zSVqxY4csi4K969eoRB9h9+umnbqX9rl27kjxXvHhxGzJkiN15552pajeyhnfeeccWL17stocOHWq5c+eO6eMCAAAgeqZMmWIPP/ywC7ALpVixYta4cWObMGGCmyNC5rFv3z63WEhBYUgfK1eutFdffTVVC+YAIDvilyMAWc6pU6ds7NixtmDBApszZw5BdqmgQEUvG2CFChWi9RGBa55m38nZs2f7gnE1YUuAHQAAQHRWQmuFfLQMGDDAHn/88aDPqxxQp06dbOPGjfbEE09E7X2ROU2aNMnGjRvntjUJFK1AuLQ6LgAAAKLj5ZdfdsF14dizZ4999913bn6IALvMRaVOveC60qVLu7FgpUqVfP1zVa9CZKpVq2Zvv/22227UqFGS5zXW9uZRUrJgDgCyK345ApDpXHbZZdahQ4cEjx07dsw2bNjgSm+qvJDXQfzPf/5jo0aNyqCWZn4NGjRwN3DNYwXfSQAAgIyTP39+O+ecc6xevXpuvKXyrikJ1vMPmitfvrzdfPPNbgJFx/z8889t06ZN7rknn3zSTQZcfvnlUT0PAAAAALFtxowZCYLr8uTJ48YF9evXd2Vhjxw5Ytu2bXM3ZSXWgiAF1yHzmTVrlm/7m2++cZkIkTr6N3LPPfdwGQEgygiwA5DpnHvuuUE7hoMGDXJlIfv37+/+1mr0gwcPUmseAAAAAFLoqaeechm+atas6csi8OKLL6YowK5v374WHx/vti+55BIbP368xcXF+Z7v16+ftW/f3qZOnerb/9dff+WzAwAAALKRl156ybddp04dV7VIWbmC2bt3r6tolCtXrnRqIaLFW2ClsWagbGsAAMQKitADyHI0IVOuXDlfZrv169dndJMAAAAAINNSIJwy16W2hObvv/9uCxYscNuFChWyL7/8MkFwnRQpUsRlsdO9/Pbbb+4GAAAAIPtQ5muPxgehguu8MqKtW7e2HDlypEPrEE2HDh3yjREp7wsAiGVksAOQ5WgAVbFiRduyZYuvjFE4VqxY4QZtKjV74MABK168uJ133nku7bgGZ5FQKvIpU6a4yaMdO3ZYyZIlXVprlbcNZ4Cg18ycOdO1ReehLHxqT+3atd0xdLxAlNnh/fffd9tXXXWVtW3bNtn3+uGHH+zrr79227fffrs1bdrU95yySXz33Xe+wEWVbwrm8OHDLm37woULbfv27ZY3b16rUKGCm4yrW7duyDYMHz7cfvnlF7c9cOBAN8l28uRJdw0XLVrkroGyXLzxxhtRuU5p6d9//7UJEybY33//7f6uUqWKtWvXzl2LcAS75tH+bKP1/U/pZ/fXX3/Z9OnTbc2aNe799LqyZcu60mD67BJPtCZ3fZQ6ftKkSW6VokeZLIsVK5bg9VrteN9997m2vfvuu+6xVq1a2bXXXpvs9dQ5jRkzxm137NiRVPUAAAARUp/No/6p+n+BlC5d2u68804bOnSo73Xnn39+VK/3smXLXP9RZWkLFCjgxhBXX311RGM/jXt++uknW7lype3cudNNCGniT33o5CYAw6Vxltqp/uv+/fvttNNOc2OsCy+80NLavn37bNq0aS4wUudXsGBBX3+9Ro0aIV/bp08f18/XdVAWwlBUMljH15hJ2RI9H374oRtT6/09999/f5KsJJdeeqnddNNNYZ9Xao+rDBuzZ8929xrvHD161I09Ne5t3ry5Lzg0OSplpvHLP//84yYVNQbTvwldMx1H38vUUGCqxpAaj+XLl89dZ5WKSs82AAAApIT6JV7QlfrnZ599dqou5MSJE91NHn/8cdenDebHH3+0r776ym337t3bzjrrrJDHVj9Wv3NrfKFt9Z/Ur1X2b/XbVdo2OdE4hkrlemOTXbt2+cYmLVq0sKpVqyb7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rUSJEtaoUSPr0aOH1a1bN1XHPnnypH3++ec2atQoW758uR05csROO+00u+KKK9zxvfJFgSxdutR++OEHmzVrlm3YsMH+/fdfy5s3r1WtWtVatGhh3bp1s9Kl9WNgYB07drTvvvsuZPs08bRly5ZUnSMAAAAApLe3337b/v77bzfm0jjul19+sTVr1ljJkiXto48+snPPPTdVx+/evXuSx5QV77///a916dIlVccGAGQMBded2va/38FyFTezOD4JAMhKCLADAAAAgDSiwLdOnTrZ4cOHfY8pkE3BbR9//LG99tprdt9996Xo2Dt27LB27drZnDlzEjy+adMmW7hwob311lv25ZdfWsuWLZO89o033rAHHngg4HHXrl1rP/30k8vO8Omnn1qbNm0C7rdv3z7buXNnyDbmypUronMCAAAAgFigsZQWSfmrUaOGC65r3Lhxqo5du3Ztu+CCC9ziphw5crjx4cSJE93ipK5du7ox2XPPPRfRMXUcAAAAAGknZxoeGwAAAACyrV9//dXuuOMOF1x3/vnn2+TJk12mOJUCuuyyy+zEiRPWs2dPmzJlSorKE11//fUuuE5Z4vr16+cmZTZu3OiyytWvX9/27t3r9vnzzz+TvF5ZGKRUqVL2yCOPuImj9evX22+//WYDBw60YsWK2e7du+26665zx00u84IyOgS6bd26NeJzAwAAAICMdu+997pFR8ood/fdd1vNmjXd2KpJkyb24IMP+sZUkYiLi3NjLo2xhg8fbv3797enn37aRo8e7bLlKdO5PP/88zZv3rw0OCsAAAAAKUUGOwAAAABIAw8//LAdO3bMqlSpYj///LMVKVLEPX766afbpEmTXNYDZZp76KGH3ARLJBkHxo8f7wL15OWXX7ZevXr5nlNZ2EsuucRq1arlgubUDu3vL3/+/G4ip2/fvlawYEHf4xUrVrR69erZVVdd5TIqHD161F588UUbMWJE0LboWArUAwAAAICs4qabbkrw96lTp+zdd9+1Hj16uEzkZ5xxht1///0RHVNjL423AilfvryNGzfOqlevbvv377d33nnHLrzwwogWYYVChjsAAAAgdQiwM7N169a59NvJadGihVulBAAAAAChKLBt2rRpbvvRRx/1Bdd58uTJ47IVXH311bZ8+XKbP39+RJMnY8eOdfdly5Z1EzyBJm70vio/++23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AwAAABC7ateubblz57a//vrL91i9evWsaNGiLkvdBx98YHfddZcv4O7TTz912+ecc06GtRkAAAAAAAAAAACxJ0MC7AoUKBDR44HcdNNNUWwRAAAAgKykWLFidskll9jUqVPtl19+sYsuusiVi7333nvdop27777b3n33XffYwoUL7dixY25RzQ033JDRTQcAAAAAAAAAAEAMyXQlYvPnz2+dO3d2meUAAAAAIJhHH33U6tatawcOHPA91r9/f1uxYoWNGzfOfv31V9/jJUuWtDFjxliRIkW4oAAAAAAAAAAAAMjYALuZM2cm+LtZs2YBH/eXK1cui4uLc6VhlWUiPTNfVKhQwcqUKZNu7wkAAAAg9Vq0aOFu/jSW+Oabb9zYQ7eDBw/aGWecYddff70rHwsAAAAAAAAAAABkeIBd06ZNE/xduXLlgI/HgieffNLdAAAAAGQe27Zts7///tstlFEAXWJa5OMt9AEAAAAAAACA1Nj34Zt2at8et130VDEz68EFBYAsJKfFgHXr1rkbAAAAAETD2LFjrXHjxjZo0CAuKAAAAAAAAIA0dWTOz3bkp0nuVuD3n7naAJDFZEgGu0idOHHCli5daqdOnbJzzz3X8uTJk9FNAgAAABDDChUq5O7j4+MzuikAAAAAAAAAAADIxGIig93+/fvt0UcftRdeeCHJcwsXLrRq1apZvXr17IILLnDlZKdNm5Yh7QQAAACQOVSqVMndb9myJaObAgAAAAAAAAAAgEwsJgLsRo8ebS+99JKtXbs2weNHjx616667zjZs2OB7bPPmzdauXTvbtGlTBrQUAAAAQGbQqFEjK168uFucs2vXroxuDgAAAAAAAAAAADKpmAmwkxtvvDHB41999ZULritbtqzNnj3bVq5caeedd57LePfqq69mUGsBAAAAxLq8efO6RTwaO3To0MF27NiR0U0CAAAAAAAAAABAJpTbYsDq1avd/Zlnnpng8YkTJ7r7Xr16WePGjd32gAED7KqrrrIffvghA1oKAAAAIDP47bffbOvWrVazZk376aefrEqVKtaqVSurUaOGFShQIOjrLrjgAmvdunW6thUAAAAAAAAAAACxKyYC7Lxyr+XLl0/w+Jw5c9y9/wRXw4YN3f3ff/+drm0EAAAAkHnMnz/fnnrqKd/fBw8etK+//jrZ13Xv3p0AOwAAAAAAAAAA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7Z9//rEOHTq4cdx///vfDGotAAAAgOwkJgLs5s6d6+7vvffeZPctX768u9++fXuatwvA/2PvPqCkKtKGAb9EyZJFRBRzQsUsyJpQzFl3jWsOa1yza9xvzbprWtdVMWfFLGZFxYxZFMwJEAEVSSJp/lP1fTM/AzMjAzNMM/M859xz79xYfbunu6vrrbcAABZeZ511Vs5kN3369LzcoUOHWHfddWOrrbaKjTbaKNq3bx8HHnhg/PLLL7HEEktEv379arrIAAAAddbnn3+e58svv3yp9QMGDMjz1DkqBd+lRAxXXnllNGjQIAYNGhRTpkypkfICAAB1R0EE2P3888+lgueKpSFjZ1e/fkEUGQAAKHBNmjSJ5557Lvbff//c8PLbb7/FW2+9lYcbeuWVV2L8+PF5v759++ahhmavjwAAALDgjBgxIs9nr5u9+uqreZ6yjhfr1KlTLLnkkjFjxoyc0Q4AAKA6NYwC0Lp16xg7dmyMGjUqunTpUuG+X375ZZ537NhxAZUOAABYWLVo0SIPA5sy2KWsBx999FGMGzcur09ZEVJw3RprrFHTxQQAAKjz0pCvSXFnqNkD7DbYYINS61O9Lpk6dWqdv3cAAEAdCLBba6214umnn477778/jj322FIZ7IorVMXSPmVVpAAAAMrTrVu3OOqoo9wgAACAApWGfk0GDx4cW221VV5OWchTcoY2bdrEKqusUmr/tL44mx0A1LSON9xXsvzWZxFxRY0WB4AqVhDjre611155nrJKDBo0qNz9Pvzww7jkkkvy8r777rvAygcAAAAAAED16dOnT55ffvnl8eSTT8Ynn3wSf/3rX/O6bbbZJurX//9NWlOmTIkff/wxmjVrFh06dPC0AAAAtT/Abp999skZ6VLa70022SQHz91yyy0l21988cU4/fTTo2fPnjFx4sTYcsstY7vttqvRMgMAAAAAAFA19txzz1hhhRXip59+iq233jpWWmmlePnll6NRo0Zx0kknldr3hRdeyCMgpTYlAACAOhFg16BBg3j44YdzkN3MmTPj9ttvj/333z8vJ6mCdP755+fgul69esVdd91V00UGAAAWApMnT87ZD3r37h3t2rXLdY969eqVOx1++OE1XWQAAIA6qUmTJvH000/n4LqUrS7V0VZcccV45JFHYo011ii1b2pTSrbffvsaKi0AAFCXNIwC0bFjx3jppZfixhtvjH79+sW7774bM2bMyNtSJap79+5xyCGHxKGHHhqNGzeu6eICAAAFbuzYsbHpppvGkCFDarooAAAAzIWllloqHn/88TwEbGojat68eZn7nXDCCXHYYYfF8ssv774CAAB1J8AuSWm+U4UoTSnTxA8//JArUCn4rlWrVjVdPAAAYCFy5JFH5uC6Dh06xNlnnx2fffZZXHHFFTnDQdqW/r755pvj7bffjvbt28f1118fq6yySk0XGwAAoE768ssvY5lllinJZleR5ZZbLs8HDx6c249SpjsAAIBaPURsWZo1axbdunXLlSTBdQAAQGWMHj06+vfvn5fvv//+HFC30kor5b87d+4cffv2jaOOOirefPPN+Otf/5qz3V133XWxwgoruNEAAAA14OCDD4533nlnrvdPnaW23HLLmDBhQrWWCwDmxuiDdo9R2/fOU6d/7O6mAdQyBRtgN6vp06fHe++9lytW06ZNq+niAAAABS5lMZg5c2asscYa0bt373L3q1+/flx66aXRvXv3eOKJJyrVmAMAAEDVGT9+fGy99dbxxRdf/O6+7777bmyxxRYxbtw4TwEAAFA3AuxS76JTTz01zj///Dm2vfXWWzkleI8ePWLttdeOpZZaKgYOHFgj5QQAABaeDHbJrMMENWjQIM9/++23OYLsdt5557z8/PPPL9ByAgAA8L/WX3/9XJdLGcd/+OGHcm/LBx98kIPrfv7559h8881jtdVWcwsBAIDaH2B33333xUUXXRRfffVVqfWp4WuXXXaJ7777rmTd999/HzvuuGOMGDGiBkoKAAAsDJo1a5bniyyySMm6Fi1alAq+m1WbNm3yfOTIkQusjAAAAPx/V1xxRUkGu2222abMoV+HDBmSg+p+/PHH2GSTTeKRRx6JJk2auI0AAEDdCLBL9thjj1Lr+/fvn4PrFltssXjllVdi2LBheYinVKm6/PLLa6i0AABAoevSpUueDx8+vGRdt27d8vy9996LoqKiUvunukZSr169BVpOAAAA/lfDhg1ze9F6660X77zzTs40PnXq1JLb8/HHH+fgurFjx0bv3r3jscceK+lcBQAAUOsD7D7//PM8X3755UutT5Wj5Nhjj42ePXvm4Z0uuOCCvO6ZZ56pgZICAAALg1VXXTU3znz66acl63r06BGLLrpozlLXr1+/kvUp4O62227Ly4YWAgAAqDnNmzePAQMGxAorrBDPPfdc/PnPf84dpFKnqM022yxnJE/tRY8//njeFwAAoM4E2BUP99q5c+dS61999dU8TynBi6WeS0lKEQ4AAFCW1q1bx8Ybb5zrGq+99lrJcLFHHHFEXj700ENjnXXWiV69esX6668fkydPjg4dOsRuu+3mhgIAANSg9u3bx1NPPRWdOnWKu+++O/bff/8cXPfDDz/EBhtsEE8++WS0aNHCcwQAACwwDaMAFA/PNH78+FxxSlJD2Lfffpt7IHXv3r1k3+JK06xpwQEAAGZ36qmnxpprrhkTJ04sWXfOOefE0KFD4+GHH4633367ZH27du3igQceiJYtW7qRAAAANWzppZeOJ554InecuvXWW/O6ddddNwfXqbcBAAB1MsCua9eueeimwYMHl2Sre/TRR/M8ZZNo0KBByb6jRo3K89RzCQAAoDx9+vTJ06xSFruHHnooBg0alKdJkybFsssuG7vuumsePhYAAIDCkDpMpfrbVlttFauttlo8/fTT6m0AAEDdDbBLjV4pwO7kk0+Ojh07xvTp0+P888/P27bffvtS+w4fPjzPl1pqqRopKwAAsPDr3bt3ngAAAFjw0lCvb7311lztO2PGjHjvvfdKRkCa3RtvvBFrr712FZcQAACgwALsTjzxxJzie8iQIbHOOuuUrE9Z6g488MBS+w4cODDPN9100wVeTgAAAAAAAOZPSrSQAufm1syZM8vdVlRU5OkAAABqf4Bdt27dcmrvo446Kt555508JGzqvXTttddGq1atSu378MMPl5nZDgAAAAAAgMJ31VVXxS+//FIl51phhRWq5DwAAAAFHWCXbLjhhvH222/HxIkTo3Hjxnkqyw033JB7Kq2xxhoLvIwAAAAAAADMf5sQAADAwqJgAuyKtWjRosLtq6+++gIrCwAAAAAAAABARRq061CyPKN5h4jyRzcHYCFUcAF2AAAAAAAAMLthw4bloWVXWWWVaNmypRsEQMFod/F/Spbf+iwirqjR4gCwsAfYjRs3rmS5devWc6yrjOLjAQAAAAAAWLhdcMEFOYDu9NNPLxVA9/3338dOO+0Ub775Zv67WbNmcdlll8Whhx5ag6UFAADqigUeYNemTZuS5aKiojnWVUbx8QAAAAAAACy8Pvzww/jb3/4Wa665Zlx44YWlth100EElwXXJ5MmT4/DDD49lllkm+vTpUwOlBQAA6hJDxP6f0aNHx1NPPZUraCNGjIixY8dGixYtYoUVVoi+ffvmqX79+jX7bAEAAAAAANRC9957b57vsccecwwL+8QTT0TDhg3jjjvuiA022CBOPPHEuO++++K8884TYAcAANS+ALt33313rtYtSKknVOoVVVZGvFRpu+KKK6JHjx7Rv3//3BsKAAAAAACAqvP555/n+fLLL19q/YABA/J8hx12KAm+u/LKK+OBBx6IQYMGxZQpU6JJkyaeCgAAoPYE2KXU3nOzbkFK2eqWXnrp2GKLLXJZllxyyWjevHmMHDkyHn744bj//vtzEOCWW26ZU5Q3bdq0RssLAAAAAABQm6TRhZLOnTuXWv/qq6/m+dZbb12yrlOnTrkt5+uvv45vvvkmVlxxxQVcWgAobcbY0VE0Y0ZebjCuQUR0dIsAahFDxEbk7HWXXnppmTdo7733zhnsjjvuuPjiiy9yT6nddtttQT9PAADAPJg5c2Y89dRT8eKLL+bGmpTZoKzM1cVSp5pDDz3UvQYAAFjAiutq48ePLzPALg0NO6sWLVrk+dSpUxdYGQGgPD+ecmTMHD0qL3do0ymi1X1uFkAtUhABdqnX0WabbRabbrppHoq1QYMU0b3gtG3btsLt+++/fw6wS7799tsFVCoAAGB+pIC6HXfcMd5+++25PqZ9+/ZuOgAAQA3o2rVrng8ePDi22mqrvPzWW2/FqFGjok2bNrHKKquU2j+tL85mBwAAUOsD7J588sk8JYsuumj84Q9/yMF2aVpjjTWiXr16NVq+4cOHlyyvsMIKNVoWAABg7vzxj38sCa5beeWVY/XVV//dALrevXu7vQAAADWgT58+ceedd8bll18e6667bnTr1i3++te/5m3bbLNN1K9fv2TflJ38xx9/jGbNmkWHDh08XwAAQO0PsEtDtD7//PO5J9Ivv/wSjz76aJ6Ks8ttsskmJQF3q6666gIt2/vvvx8HHHBAXk4VupRtDwAAKGzpe/wrr7ySl6+88so46qijarzjDgAAAOXbc88948ILL4xPP/20VFtMo0aN4qSTTiq17wsvvJCHlE3tRwAAAHUiwO68887L84kTJ8ZLL70UAwcOzAF37733Xvz000/xwAMP5Cnp2LFjScDd4YcfXuVlSUPVTp48OaZOnZrTi3///ffRtGnTOPDAA+Oyyy5b4MPXAgAAlffBBx/k+dprrx1HH320WwgAAFDgmjRpEk8//XQcccQR8dRTT+UAujSqUMpol0Y7mtXDDz+c59tvv32Vl+Ozzz6LDz/8MEaMGBE///xzHp42Xb9Xr15V1kaU2qCee+65+Pjjj+O3337Lw+OmoMJ27dpVyfkBAIBaGGBXrEWLFjnNd5qScePG5V5IKdguBd199NFHMXr06Lj33nvzVB0Bdm+++WZMmjSp1LoePXrElltuGa1atar0+WTJAACABS81xCSrrbaa2w8AALCQWGqppeLxxx/PQ8DOmDEjmjdvXuZ+J5xwQhx22GGx/PLLV9m1TzvttLjrrrvim2++KXP7EkssEZdccknOtDc/BgwYEIccckhO8DCrlKkvjfh09tlna1sCAIACU1ABdrNr3bp17LjjjrH00kvn6Z577skBcNUpBfKlSlsKsvv666+jf//+8eSTT8arr74aTzzxRNx0000qNgAAUOC6deuW57/88ksUgnfffTdefPHFGDNmTLRt2zY22GCD6NmzZ5XULVLHpJT54JNPPokJEyZEs2bNYrnllovNN988ZwAvz8033xxDhgyp8NwpO8NFF11Uo48PAACom9nsKpLqPFXtwQcfLAmuW3PNNWOjjTbKiRc+//zzePTRR3NGu7322ivXe4455ph5ukYKHkztXqkdavHFF49dd901J59IdbrBgwfH3//+9xg/fnz861//quJHBwAA1LoAu2HDhuWsdWlKGex+/PHHkm2NGzeO9dZbLw/lWh3WXXfdUn8fdNBBucdSqjTdcsstscUWW8Tee+9d6cwZ5dHgBAAAVW/DDTeMxRZbLHegSUF2iy66aI3c5pEjR8af//znePbZZ8use9xxxx3znHEhNcicddZZueElZXcoKzguZUW47LLLymyceuihh0qGVZrXALvqfHwAAAALUmqvSQFvqZ61+uqrl9r27bffxg477BDvv/9+nHzyybHbbrtF586dK3X+yZMnx6GHHprrcuuss04888wzOdHErFn5Uv0u1eH22GOP3HEJAAAoDAURYJcyxRUH1KUGsNRIM2uDTnFA3aabbpp7DKWMDAtSSvd93XXX5WC/FGRXmQA7AABgwWvYsGFcfvnl+bv8PvvsE3fffXe5QwtVlxTYlzrofPzxx3mon9QAs8oqq8RXX32V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23TTTXO9KA0JO3DgwFhyySXn6VzFnY+SH3/8sdz9fvrpp1KJKAAAWDik4LqD61hgbb9jI9ZZPuqMggmwS8OxFgfUpUrKqFGjSm1PAW/du3fPlZnNNtssZ1moasVBdWkCAAAWfi+//HLO5Pbdd9/Nse29996Lhx56KM4888w8pNA+++xTZdctzk6QhhCqSHHgWWUzIKSGpK233jo3+KShYVOg3KyNLykA7uijj45+/frlILwUWJiGkZ3VuuuuW+a5U2em0047LR9zyy235KFoUya8li1bLpDHJ/MCAADUTcVZ6FK2uGJrrrlmPPzww/HYY4/FIYccUqo+V7xfcRBeVUptVKktaujQoTm4LiWCSPN5lTp5Lb744jnBQ0UjNH3++eclyyuuuOI8Xw8AAKhaVZMGbj6lSklK5Z0qR3fddVdJcN1KK60URxxxRNx3330xevToeP/99+Pyyy/Pw8bquQMAAFTk7bffjr59+5YE162zzjrxl7/8JWduO/bYY3NjSYMGDXKwWspid/fdd1fZDS3OTlBRZoJZsxPMms1gbtx44405uC4Fw918881z1I9S5rh///vfufNQCrZL9ajKSvckmTZtWm68WpCPDwAAqHuKM2+nJAzFdtttt5wc4ZFHHonjjz8+d6JKnYB22mmn3DknJWZIgWtVaeTIkTnb+azBdVUxBG2qkybpMZRn0KBBJZ2UVl555fm+JgAAUIsy2H377bd5vswyy+RGruIsdYZnBQAA5tWRRx4ZkydPzvWM1JFnvfXWm2Ofr7/+Ovbdd9/cwHHMMcfkYVZTZoH5VZxp4Jtvvonp06dHw4YNK8xOUNnMBB988EGep8dWXmPSIossEuuvv348+uijubNSZc1aHxs3btwCfXwAAEDds/nmm0erVq1yHWbixIm5brbaaqvljlJXX311XHbZZXkqluohs/5dFVJWvNRGleoyKaguBftVRXBdkoIC02N7991389SjR49S21PnpltvvTUvb7PNNrlOBwAAFIaCCLBLGRdShWX2IYsAAADmxWeffRZvvPFGNG7cOJ566qlYbrnlytwvNZQ88cQTscoqq+RMd08++WTOkDC/iodfTdnj3nrrrdhggw3m2GfEiBHx5ZdflspkMLeKh15NwW0VKd7+e0O5lmXWYXXbtm27QB8fALXHPQPeqOkiUID+uO36NV0EoACl+lsa4WjGjBk5g1uxK6+8MtfdrrnmmlzHSIF1qQ5y3nnnxR/+8Icqu36qA6W2qjSEa7du3XJwXWWGhU0dky699NK8vPXWW0evXr1Kbd9zzz3j7LPPzkF8Bx54YDz77LPRrl27vC1l40sZ+lInsJSx76STTqqyxwXAgjF12EdRNG1qXm48qnFErOrWA9QiBRFgl7JJpJTeTz/9dE0XBQAAqAWKhzRNGRDKC64rlrIi7L333nHhhRfm46oiwC5l5E7Dtv7yyy/Rr1+/MgPQrr/++jxPjUM77rhjpc6//PLLl2SQSw1MKZPd7FLGh9deey0v/949KMu1115bMtzs7JkVqvvxAQAAdVPTpk3nWFe/fv048cQT85Q6EaW/01QdQ9Sm4Lpko402KqnTlCV1IkoZ6WYPsEtBf0nr1q3nCLBLjy3Vn1Lm9FT3XGmllWLnnXeO5s2bx/PPP1+SqTw9zrIysANQ2MZdck7MHD0qL7dt0ymi1X01XSQAaluAXeqlk3okAQAAVIUJEybMMcxpRYr3Gz9+fJVlXjj22GPjf/7nf+LGG2+MrbbaqlTg3qBBg+Kiiy7KywcddFC0b99+jnOkhpdhw4blbaeeemqpbelc//jHP3KWg3322Sfuv//+UkPF/vTTT/m8xUO7/vGPfyx1/Ouvv56PTYFxKTvCrFJWunS9Bx98sKR8s2aPqKrHBwAAUFmpA091SR2Yit12220V7pvqObMH2M2Nvn37xmOPPRaHHHJIzpg3axBf6tx0xhlnxN/+9rdKnxcAAKgDAXYpBfbo0aNzQ06qQAAAAMyP4iFN01Cxc+PTTz8tqZtUlVNOOSUPPzt48ODYY489YpNNNolVV101vvrqqzwUbepktMIKK5RkOJhd//798/C2yy677BwBdt27d48TTjghDz+UstSl86Tzd+3aNUaOHBkvvvhi/Pzzz3nfP/3pT3l4olm98MILcdppp8USSyyRy5SC81JGunRsGgbpxx9/LBkKNmX2q47HBwAAMKvTTz89Z5A7//zzy8zSPb/7/55zzjknZ8ibG7Nn+U7atGmTO0IVZ8CrKMgu1ZtSvWzo0KExderUWHLJJWPLLbfM9TIAAKDwFESA3dprr50bZlJK7LKGFgIAAKiM9ddfP2dme+WVV3KjRQr+Ks/XX38dt99+e17ecMMNq+xGp6xvKdDs6KOPjrvuuisHrqWp2DbbbJOzFcxrUN/FF1+cA+rOPffc3GEpZUGYVWqYOe6443IGhNmlx7nxxhvHyy+/HCNGjJhjexrO6Igjjoizzjqr3E5Q1f34AACAuiV1MHr77bfzEKlzEzBX2f1/z+wdmyor1cHKqn+VpUGDBrH55pvnCQAAKHwFEWB32GGH5QC7s88+Ox5//PFcsQAAAJhXKSPbjjvuGA899FDssMMOOdPbfvvtVypYbObMmTko7aijjspDwy6//PKx2WabVXkmvTvuuCMPl5qGTR0zZkzOapAC3JZbbrkKj01DBvXp0yfvX5YUQJiC2w4//PB4880383CyaWjc5s2b53OnIMPZh3YtloLrUuBhGko2dXQaPnx4znjXqlWrnHVunXXWiUUWWaRaHx8AAMD8KCoqKqkbAQAA1PoAu9Twdfzxx8e//vWv2HTTTfNQRT179pQKGwAAmGdXXnllHr40ZWhLnXpSnWO11VbLAWCTJk2Kjz76KAeYJU2bNo2bb745GjasnipSly5dYs8996zUMbvuuutc7deoUaPo1atXniorBchVRVDhvDw+AACA+fHDDz/keepkBAAAUOsD7GZtxEpZD9KU1K9fv8KeR9OnT18g5QMAABY+Sy65ZLz22mtx8MEHx9NPP52D6t5444059lt99dXjhhtuyFnbAAAAWDCmTp2aM4vPnpEurZ8yZUq5x6UM5Pfdd1/uTJXal5ZeeukFUl4AAKDuKogAuxkzZpS5ftaKFQAAwLwE2T311FPx4Ycf5iC7Tz75JDfGpAwHqRFmk002id69e7uxAAAAC1gayejtt9+eY31lsnPvsMMO0aRJkyouGQAAQAEG2D3xxBM1XQQAAKAW6969e54AAABY+KURkLbddtu45pprarooAABAHVAQAXZbbbVVTRcBAAAAAACABeSGG26ICRMmlPx98MEH56zj/fr1ixVXXLHcwLpmzZrFsssuGy1btvRcAQAAdSfADgAAAAAAgLpjjTXWKPX3CiusEFOmTIn11ltPBnIAAKCgCLADAAAWaq+++mrce++9eblXr16x++67l1pXGcXHAwAAsGA98sgjbjkAAFCQFniA3bhx40qWW7duPce6yig+HgAAqLs++OCDuOKKK/JyynaQAuRmXVcZxccDAAAAAABAjQTYtWnTpmS5qKhojnWVUXw8AABQd3Xr1i123HHHvLzmmmvOsa4yio8HAAAAAJhbTXpuEjPH/29ioTEzW0d85d4B1CaGiAUAABZqffv2zdPvrQMAAAAAqA6tDjqyZPnzzyKi8oNrAFDAFniA3bvvvjtX6wAAAAAAAAAAAKBOBdiVNeSSYZgAAICqNG7cuBg1alS0adMmFltssSrfHwAAAAAAgLqhfk0XAAAAoKrdfffdsfLKK8fZZ59dLfsDAAAAAABQNwiwAwAAAAAAAAAAgEIYIvb3DBs2LF5//fU8PNOUKVMq3Pecc85ZYOUCAABqr19//TXPGzduXNNFAQAAAAAWMlNeGRgz/+83xqbjmkbEpjVdJABqY4DdN998EwcccEAMHDhwro8RYAcAAFSFt99+O8/bt2/vhgIAAAAAlTL+xv/EzNGj8vKibTpFtBJgB1CbFESA3bhx42KTTTaJr7/+Oho1ahQ9evSIN998M2/r3bt3jB49Oj799NMoKirK69Zff/1o0qRJDZcaAAAoFG+88UY8+OCDJX+/9957JetPPfXUco+bNm1azqL9xBNP5L833HDDBVBaAAAAAAAAFhYFEWB35ZVX5uC6JZdcMl544YVYZpllol69ennbSy+9VJLh7q9//WtuNGvVqlVJAxgAAMC7774bF1100Rw3IgXaFQfb/Z5NN900Nt98czcTAAAAAACAwgqwe/TRR/P8jDPOyMF1ZVlqqaWif//+se2228aTTz4ZN9xwQxx66KELuKQAAEAh6tKlS6nguBEjRuTMdEsssUSstNJK5R7XuHHjWHzxxWPjjTeOPffcM+rXr7+ASgwAAAAAAMDCoCAC7NLwr0nfvn3n2DZjxoxo0KBBXk6NXWeddVYOsLvzzjsF2AEAANl2222Xp2L//e9/44gjjsjr0jIAAAAAAAAstAF2U6ZMyfNOnTqVyiQxderUmDBhQrRu3bpk/ZprrpnnQ4YMqYGSAgAAC4M03OtNN90UK664Yk0XBQAAAAAAgIVYQYx/VBxY9/PPP5es69ChQ55/9dVXpfZNAXfJ+PHjF2gZAQCAhUcKrNt///1jww03rOmiAAAAAAAAsBAriAC7pZdeOs+HDx8+R6a6xx57rNS+xX8XB+ABAAAAAAAAAABArQ2w69OnT54PHDiwZN3uu++e5xdccEFcdtll8fLLL8cVV1wRxx57bF7ft2/fGiotAACwsJg8eXJcfvnl0bt372jXrl00aNAg6tWrV+50+OGH13SRAQAAAAAAKCAFEWC388475/ldd91Vsm6fffaJXr16xa+//hrHH398bhA77rjjYuLEidG+ffs4++yza7DEAABAoRs7dmysv/768de//jV32Pnpp59i5syZNV0sAAAAAAAAFiIFEWC32mqrxYQJE2LQoEEl61JmiccffzxnrOvYsWNe17Rp09hpp53itddei6WWWqoGSwwAABS6I488MoYMGRIdOnSIf//73yXZsLfffvt48skn46qrroq11147r0udeB588MHcuQcAAAAAAAAKKsAuadGiRTRv3rzUulatWuXhnH744Yf47bff8vBOqdFrueWWq7FyAgAAhW/06NHRv3//vHz//ffnYLuVVlop/925c+fo27dvHHXUUfHmm2/mDHcp2911110XK6ywQg2XHAAAAAAAgEJSMAF2v6dx48Y1XQQAAGAhMXjw4Dwc7BprrBG9e/cud7/69evHpZdeGt27d48nnngi3nnnnQVaTgAAAAAAAApbw5ouAAAAQHVksEtWXHHFknUNGjTI85Qde/Ygu5133jk+/PDDeP7552OttdbyhAAAAAAAc635DrvHzEkT8/KIyS0i3nPzAGqTggqwKyoqylkj7rnnnnjrrbdKGsU6duwY66yzTvzpT3+KrbbaKurVq1fTRQUAAApYs2bN8nyRRRYpWdeiRYs8L65nzKpNmzZ5PnLkyAVWRgAAAACgdmi+4x4lyxM/CwF2ALVMwQTYDR8+PPbcc894+eWX59g2duzY+Pjjj+PWW2+NjTfeOO68887o3LlzjZQTAAAofF26dCmpZxTr1q1bnr/33nu5c8+sHXeGDRuW5zrzAAAAAAAAUHABdr/88ktssskm8cUXX+S/119//ejTp0+pRrFnn3023njjjXjxxRfzvinDXatWrWq45AAAQCFaddVVo2HDhvHpp5+WrOvRo0csuuiiOUtdv3794pBDDikJuLvtttvy8mqrrVZjZQYAAAAAAKDw1I8CcP755+fguubNm8fDDz8cr7/+epx77rlx+OGH5yktp3VpW9rns88+iwsuuKCmiw0AABSo1q1b5+zXI0aMiNdee61kuNgjjjgiLx966KGxzjrrRK9evXIHn8mTJ0eHDh1it912q+GSAwAAAAAAUEgKIsDu/vvvz/OLLroodthhh3L3S9suvPDCvNy/f/8FVj4AAGDhc+qpp8YJJ5wQEydOLFl3zjnnxI477piX33777Xj11Vdj6tSp0a5du3jggQeiZcuWNVhiAAAAAAAACk1BDBH73Xff5fkee+zxu/umfY4++uiSYwAAAMrSp0+fPM0qZbF76KGHYtCgQXmaNGlSLLvssrHrrrvm4WMBAAAAACpr0sP3xsxJ/9vRt8XkFimywU0EqEUKIsCuRYsW8dNPP81VtohWrVrlucwSAADAvOrdu3eeAAAAAADm16RH7ouZo0fl5RZtOkW0EmAHUJsUxBCxa665Zp4PHjz4d/d9880387xHjx7VXi4AAAAAAAAAAADqroIIsDvssMPy/IQTTogJEyaUu1/alvaZ9RgAAAAAAAAAAACotUPE7rHHHvHqq6/GFVdckTPTnXrqqbHFFltEly5d8vbhw4fHM888ExdeeGF88cUXcdxxx8Wuu+5a08UGAAAKwOuvvx79+/evknNtuOGG6hoAAAAAAAAUVoBdw4b/vxgpgO6QQw6pcP+rrroqT2WZPn16lZcPAAAoXO+9917885//rJJzpUzZOvMAAAAAAABQUAF2M2bMqNb9AQCA2qtbt26x4447lrntgw8+iK+++iovt2zZMrp37x5t2rSJSZMmxccffxyjR4/O29q2bRu9e/eONddcc4GWHQAAAAAAgMJWEAF2TzzxRE0XAQAAWEj17ds3T7N75plnYvvtt4/WrVvHZZddFnvttVc0bty4ZHtRUVE89dRTceSRR8aXX34Zm2++eRx++OELuPQAAAAAAAAUsoIIsNtqq61quggAAEAtMnPmzDjkkENi2rRpMXDgwNhwww3n2KdevXq5LjJo0KCc2e7kk0+O3XffPTp16lQjZQYAAAAAAKDw1K/pAgAAAFS1N998M7755pvYeOONywyum1Xnzp1jv/32iylTpsQjjzziyQAAAAAAAKCEADsAAKDW+fzzz/N86aWXnqv9u3XrVuo4AAAAAAAASATYAQAAtU5RUVGef/vtt3O1f8p2Vzy0LAAAAAAAABQTYAcAANQ6q6yySp6/8MIL8c4771S475gxY+K2227Ly6uuuuoCKR8AAAAAAAALBwF2AABArbP22mvHWmutFTNmzIitttoq+vfvn5dnN2jQoNh4441zkF27du1i5513rpHyAgAAAAAAUJga1nQBAAAAqsMtt9xSEjy3++67R/v27WPNNdeM1q1bx6RJk+Kjjz4qGUK2UaNGcfPNN+dtAAAAAACV0erAv8TMX3/Ny1+NaxrxgvsHUJsIsAMAAGql1VZbLV5//fU47LDDYuDAgTF27Nh49tln59gvDQt77bXXRq9evaqtLD/99FO8+uqrOdivbdu2sd5668Xiiy9eJecuKiqKDz74ID755JOYMGFCNGvWLJZbbrno0aNHNGzY8HePHTZsWHz33XcxcuTIaNy4cXTr1i1n/1tkkUUqPPapp56Kr776qsJ96tevH4ceeug8PS4AAIAFLWU+f//99+OVV17JdbhU10rrUmetu+++e77OneqkG2200e/ut88++8QZZ5wxX9cCYMFr0mvTkuVfPwsBdgC1jAA7AACg1lp++eXj+eefjw8//DCefvrpHIQ2fvz4HIS2zDLL5Ax3vXv3rrbrT5w4MU466aS44YYbYtq0aaUCz/bYY4/497//nYemnVc33XRTnH322TlAbnYdOnSIU045JY4//vioV69eqW1vvfVWXHrppfHcc8/lRp7ZpSDAY445Jk477bQcdFeWa665Jh5++OEKy9egQQMBdgAAwEJh8uTJsdhii+V63OyqItv59OnTc53094waNWq+rwUAAFQtAXYAAECt17179zwtSFOnTo3tttsuXnzxxfz3uuuuG6usskrO+jZo0KCc/WDo0KHx0ksvRatWrSp9/gsvvDAHwCVNmjTJwYIpK97o0aPz+VO2vBNPPDG+/vrruOqqq0od+/LLL8c999xTMjzuBhtskDPXpWC7N998M8/POeecnLFhwIABFWbCS/e1Z8+e5QbYAQAALAxmzpwZv/76a84GnjKcp3pO6iyVOiZVtccffzzXwcrSpk2bKr8eAAAwfwTYAQAAVIN//etfJcF1N998c/z5z38uNbzq9ttvn4ceOuuss+Lyyy+v1Ll/+eWXHACXrL766vl8nTp1Ktn+888/x6677pqHxk1Z8o477rhYdtll52i0SQF4RxxxRKkGnDTM7F//+tfckJSy/qVMdUcffXS5Zdlss80qXX4AAIBC07x58xg3bly0aNGiZN39999fLddKwXUrrbRStZwbAACoevWjAHsIDR8+PD7++OMYMmRIhRMAAEAhSkP/XHLJJXl5n332KRVcl/Tt2zdOOOGEvJwC2FJAXGV89NFH8dtvv+Xlv//976WC65IUMPfPf/6z5O933nmn1PY0LG4amuhvf/vbHNkRWrZsGdddd11JY89dd91VqbIBAAAsjOrVq1cquA4AAKDgMtiNGDEiTj/99HjggQdyxoS5UVRUVO3lAgAAClsaxvTee+/Ny2kYn913373UusooPn5+vfDCC/HTTz/l5cMOO6zMfQ4//PA8zGsaSvbRRx+N/fbbb67P365du1JZFsrSrFmzMvdP1l577QrPX79+/Tzk7LBhw+Kbb76Z63IBAAAAQF00/oarY+b4cXl50ZmtI+LImi4SALUtwC5lrFtvvfXi+++/r+miAAAAC5kPPvggrrjiirw8ZcqUHCA367rKKD5+fr3xxht5vsgii8T6669f5j5LLbVULL300vH111/n/SsTYLf88stHjx494t13340rr7wyNt988xwUN6viDHbpOhtuuGGlH8PYsWPzvG3bthXuN2nSpHj22Wfju+++iyZNmuShjtZaa61o3Lhxpa8JAABQF/zP//xP7syUEk6krOKpfrfHHntEz549a7poAMyjKa++EDNHj8rLTdt0imglwA6gNimIALtzzjknB9elxpg0TNKOO+4YSy65ZDRsWBDFAwAAClgK6Ep1iGTNNdecY11lFB8/v9Lwq8XBbY0aNaowUC4F2KVMcZWRgunuvvvu2GmnneKxxx6LlVdeOXbZZZc8VOyYMWPyuvfffz/Xq+6///5o2rRppc6fzvHkk0/m5S222KLCffv165enWaUGomOPPTYPQVvR4wcAAKiL7rrrrlJ/v/TSS7mT2M477xw33XRTLLroojVWNgAAYE4FEcFW3HBz8cUXx9FHH13TxQEAABYiffv2zdPvrVuQfv755zxv3759hfsVD906btz/Dh9RGSussELOYHfqqafG5ZdfnoebndVee+0V//3vf6Nly5aVPvdf/vKXnJmuRYsWceKJJ5a7X7169WKVVVaJZZZZJj/WkSNHxiuvvJIff+pINXDgwHjqqadyJr+5lc4JAABQG3Xt2jX23HPPWHvttXPHsFT/GTJkSFx//fW5LvXggw/m+tRzzz03R5byiqhHAQBAHQiwGz16dJ7vtttuNV0UAACA+TZ58uQ8/73AsuLMcimYrbJGjBiRM9i99dZbJUPRpiFn0/o05Oydd96Z5w8//HCsuuqqlRqqqH///nk5Beh17ty5zP0OPPDAuPrqq2OJJZYotX78+PFx/PHHxw033BAvvvhinHnmmbkzFQAAQF3WoUOH+PLLL6NBgwal1qdgu/322y93bvrXv/4VL7zwQtxxxx2x77771lhZAQCA0ua++0s1at26dckwQgAAAAu7Jk2a5Pm0adMq3O+3334rtf/cSufdbLPNcnBdGtb2o48+ysFst9xySzz77LPx2Wef5Qx+X3zxRWy88cbx008/zdV5//3vf8fZZ5+dl9N87733LnffHXbYYY7guqRVq1Z5yNhtttkm/52C8Iof59woKiqqcAIAAFgYpcC62YPrZs1Ad9FFF+VOU8k999xTqXOrRwEAQB0IsEuZFpIPPvigposCAAAw31KQ2dwM/Vq8vXj/uXXrrbfGp59+mpfvuuuuWHbZZUtt79SpU26QSef98ccf8xCyv+c///lPHH300Xn5jDPOyEO8zo80zGxxNr933nlnvs4FAABQ2zVs2DB3pEqGDh1a08UBAAAKbYjYY489Nh577LH4xz/+EY888kjuqQMAADA3Xn311bj33nur5Gb16tUrdt999/k+z3LLLZfnX3/9dc4kUF4dJw0PlCy//PKVOv/gwYPzvGvXrrHSSiuVuc+iiy4aPXv2jCeffLJk//KkLHNHHXVUXj799NNz3Wx+devWrWR57Nix830+AACA2q5p06Z5PnXq1JouCgAAUGgBdn369IkzzzwzN+LsuOOOeTkNc9SoUaOaLhoAAFDgUibsK664okrONWXKlCoJsOvRo0eeT5w4MT788MNYffXV59gnZZb75JNPSu0/t4oz3/3e0LKNGzcutX9Z0r077rjj8vLf/va3OPfcc6MqjB49umS5shn6AAAA6qL33nsvz7t06VLTRQEAAAotwC6lvS726KOP5ilp0KBBhcdNnz692ssGAAAUtpQpLXXUqQqpo09V2HLLLXPwWwrYu+222+KSSy6ZY5/bb7+9JLvdDjvsUKnzL7XUUnn+xRdfxJgxY6JDhw5z7DNt2rR48803S+0/u3/+859x4oknlgwLWxWZ64rdeeedJfW9NdZYo8rOCwAAUBsNGDAgXnnllbzct2/fmi4OAABQaAF2M2bMqNR6AACAYqnhodAaH1q0aBEHHXRQHnr1qquuil133TU22GCDku2fffZZ/P3vf8/Lu+yySx7qdXaPP/54fPvtt3mo1z333LPUthSQd/HFF+c606GHHhp33XVXqWx2qTPSCSecEKNGjcp/lxWAmI4/5ZRT8nLKIv4///M/c/34Uua9VK5OnTqVuf3f//539OvXLy+nx966deu5PjcAAMDCaMSIEbH55pvn5VTXOuCAA0ptP/jgg2PDDTeMbbbZJtelUmer4ozjqf501lln5b/bt28fxxxzTA08AgAAoKAD7J544omaLgIAAECVSgF0KUjuq6++ik033TT23XffWHXVVfPfN998c/zyyy/RsWPHuPTSS8s8/sorr4ynnnoqll122TkC7Hr16hV//OMf45577omHHnooVlxxxZJAvZEjR8Zjjz0Ww4YNy/v+4Q9/yPvO6sYbbywJrltttdWic+fO8d///rfcx3LYYYeVNP4kDz74YJx++um5HN27d4/FF188B9yla6fHnIbtTZZeeum47LLL5uMuAgAALDhHHHFEDBw4sOTvVMdJ3n///VhppZVK1qd6VP/+/efIIp46IyU//vjjHOd+9tln44YbbsjLjRs3zkF2Kat5CsybOXNmXr/YYovlUZ7atm1bTY8QAABYaAPsttpqq5ouAgAAQJVq165dbkDZa6+94o033ojrr7++1PZVVlklZ55LQWjz4pZbbsmNL9dcc03OdHf55ZeX2p4C4tK1//Of/0T9+vVLbSsOgEuGDBmSG5EqkjItpKFei6288so5KG/QoEF5ml26Xspcl4IEy8tyBwAAUGi+++67kiC5WU2ZMqXU+pS1vLJuuummuP/++3PSia+//jrX44rrbqnT1O677x7HHXdcrksCAACFpSAC7ArFl19+GZ9++mnuLTR58uScTWK99daLbt261XTRAACAhdAyyywTr732Wrz00kt5GjNmTLRp0yYPC9SnT59SQWuz23bbbXPwXaqXlGWRRRaJK664Ik477bSc6S5lrJswYUI0b948lltuudhiiy3KDd5LmedSA9Hcmj1ALw05m6Z33nkn3nrrrRg+fHj8/PPP0apVq1hhhRVis802iyWXXHKuzw8AAFAIUmbviRMn/u5+TZs2nWPdEkssEUOHDs3LZdXjUmbzNCXTp0+PUaNGxYwZM/KQsKkeBwAAFK46H2CXUnYfddRR8cwzz+Shmsqy8cYbx1VXXZWHPgIAABYeqePMddddl7MEfPzxxzFu3LiSoXfKGwq1oqFS50XKRpDqFGmqjKOPPnqu9ksZ4v785z9X6twpM0Ka5tdaa62VJwAAgNqgS5cu83xso0aNSg0jW5HU2Wp+rgVA4Wl90jlRNG1qXv5kVOOIh2u6RADU6gC7lHXh9ddfzz13fi+jwjnnnDPf1/vtt99yg1uxNddcM3r06JErN2mopJRt4sUXX8wZJtLwThtssMF8XxMAAKh+Y8eOzdkB0vd6AAAAAIDq0nilVUuWpzZxnwFqm4IJsPvmm2/igAMOiIEDB871MVURYFecUSJlfDj11FNjxRVXLLUtZbbbbbfdYvz48XHggQfmrBcAAEDhO/LII3NwXYcOHeLss8+Ozz77LA+puv322+dt6e+bb7453n777Twkz/XXXx+rrLJKTRcbAAAAAACAAlI/CkAapmmTTTbJwXUphfZ6661Xsq1379456C0FwRVbf/31Kz28UnkaN26cs9TddNNNcwTXJVtssUVJIN/QoUPjww8/rJLrAgAA1Wf06NHRv3//vJyGh00BdcVD9XTu3Dn69u0bRx11VLz55pvx17/+NWe7S5mtV1hhBU8LAAAAAAAAhRVgd+WVV8bXX38dSy65ZB4i9o033ijZ9tJLL+V1X331Vey88855XatWreK5556rsgC7FLBXkRTkV2z48OFVcl0AAKD6DB48OGbOnBlrrLFGqe/zs6tfv35ceuml0b1793jiiSfinXfe8bQAAAAAAABQWAF2jz76aJ6fccYZscwyy5S5z1JLLZUzUGy11VZ52NYbbrhhgZXvhx9+KFlOQ0cBAACFn8EumTVLdYMGDfL8t99+myPIrrgzz/PPP79AywkAAAAAAEBhaxgF4NNPP83zNEzT7GbMmFHSEJYavs4666x48skn484774xDDz10gZQvDR+bdOjQIXr06LFArgkAAMy7Zs2a5fkiiyxSsq5Fixalgu9m1aZNmzwfOXKk2w4AAAAAVMq4i8+OGT//lJfbNmwbEX93BwFqkYIIsJsyZUqed+rUqdTQrVOnTo0JEyZE69atS9avueaaeT5kyJAFUrYHH3ww7r///rx8zjnnRMOGlbtl9erVq6aSAQAA5enSpUueDx8+vGRdt27d8vy9996LoqKiUt/Vhw0blue+vwMAAAAAlTX1k49j5uhReblxm04RrdxDgNqkIIaILQ6s+/nnn0vWpWxxyVdffVVq3xRwl4wfP77ay5Ua3vbbb7+8vM0228QRRxxR7dcEAADm36qrrpo7xxRny05SNupFF100Z6nr169fqe/9t912W15ebbXV3H4AAAAAAAAKK8Bu6aWXniO7RHGmuscee6zUvsV/FwfgVZehQ4fmIWsnTpwY6623Xtx1113zlM0iZcaoaAIAAKpeyoK98cYbx4gRI+K1114rGS62uNPMoYceGuuss0706tUr1l9//Zg8eXKuY+y2226eDgAAAAAAAApriNg+ffrESy+9FAMHDsyNXMnuu+8eAwYMiAsuuCBatGgR6667brz99ttxxhln5O0p+K06g+s23XTTGD16dC7PU089Fa1ayeEKAAALk1NPPTV33EmdZoqdc845+fv+ww8/nOsXxdq1axcPPPBAtGzZsoZKCwAAAAAAQCEqiAC7nXfeOc4666ycJe6kk07K6/bZZ5+4/vrr45VXXonjjz++1P7t27ePs88+u1rK8vHHH8dmm20WP/zwQ6y99trxzDPP5OwXAADAwiV15EnTrFIWu4ceeigGDRqUp0mTJsWyyy4bu+66ax4+FgAAAAAAAAouwG611VaLCRMmlBqCtUGDBvH444+XBN6lbHJNmzbNmesuueSSWGqppaq8HB9++GFsvvnmMWbMmBxc9+yzzwquAwCAWqh37955AgAAAAAAgIIPsEvSMLCzS8OyXn755XmaOnVqNG7cuNqu//777+fsFmPHjs3DwspcBwAAAAAAAAAAULfVj4VEdQbXvfvuu3lY2BRct+666wquAwCAhdzNN98ce+yxRzz66KMxffr0mi4OAAAAAAAAC6mCyWBXLDV+vfXWW/Hee+/FTz/9lDPXnXPOOdV2vR9++CFnrkvXatSoUR6Ctl+/fuXuv8UWW8Qaa6xRbeUBAADm35QpU+K+++7LU4cOHWLPPfeMfffdN2erBgAAAAAAgIUywO6WW26JM888M7777rtS62cNsFt11VXjk08+iWHDhsVyyy0339ccM2ZMDq5Lpk2bFueee26F+19zzTUC7AAAoMD17Nkzttlmm3j66afzd/4rr7wyT6usskoOtNtnn32iS5cuNV1MAAAAAAAAClzBBNidfPLJcckll+TllEkuBc8NHTp0jv223Xbb+Pjjj+Oxxx6L4447br6v2759+zjhhBPmev8111xzvq8JAABUr9VXXz0GDBgQo0ePjrvuuituvfXWeOedd3Jd4rTTTovTTz89Ntlkk9hvv/1i1113jRYtWnhKAAAAAAAAKMwAu2eeeSYH19WrVy8H2p1xxhm5gSv9Pbs0hGva96mnnqqSALtOnTrFpZdeOt/nAQAACk/Hjh3j2GOPzVMKrrvtttvijjvuyFmzn3/++Tz95S9/iZ133jkH2/Xp0yfq169f08UGAAAAAACgQBREy9FVV12V56ecckpceOGFFWaP6Nq1a55/9NFHC6x8AADAwi8ND3vBBRfEN998E88991zsv//+0bJly5g8eXIOukudeaqiEw8AAAAAAAC1R0EE2L3++ut5fsQRR/zuvp07d87zMWPGVHu5AACA2idlyt5ss83ipptuih9++CGuvvrqaNasWd42derUmi4eAAAAALCQaXfR1dG+3715GnP01TVdHABq4xCxP//8c6nguVkbvoqKikqtM1wTAAAwv6ZNmxaPP/54HjL2sccei99++81NBQAAAADmSYP2HUuWZ4x3EwFqm4IIsGvdunWMHTs2Ro0aFV26dKlw3y+//DLPO3b8/x9QAAAAc5s9OwXV3XPPPfHjjz+WrG/fvn386U9/mqus2gAAAAAAANQdBRFgt9Zaa8XTTz8d999/fxx77LEVZrBL+yQbbLDBAi8nAACw8EmddG6//fYcWPf555+XrF9kkUViu+22i/322y+23nrraNSoUY2WEwAAAAAAgMJTEAF2e+21Vw6wO+uss3KwXe/evcvc78MPP4xLLrkkL++7774LuJQAAMDC4pdffom77747B9W98sorpbb17NkzB9Xtscce0aZNmxorIwAAAAAAAIWvIALs9tlnn/jvf/+bh2vaZJNNcsBdnz59Sra/+OKLOQDvyiuvjIkTJ8aWW26ZM00AAACU5a677io13OsyyyyT6x0psG7ZZZd10wAAAAAAAFh4AuwaNGgQDz/8cOy44445yC4N35SmYinorlivXr1yYxkAAEBFWrduHbvvvnsOqttoo43cLAAAAACgWvx48l9ixo9j8nKH5h0i4j/uNEAtUj8KRMeOHeOll17KmezWWWedHHRXrF69erH66qvHVVddFc8//3y0bdu2RssKAAAUtp133jlGjRoV1113neA6AAAAAKBapeC6maNH5anBuP8NtAOg9iiIDHbFGjVqFIcddlieJk+eHD/88EPMmDEjB9+1atWqposHAAAsJBZbbLGaLgIAAAAAAAC1QEEF2M2qWbNm0a1bt5ouBgAAAAAAAAAAAHVUwQwRCwAAAAAAAAAAAIWkoDLYffrpp/Hggw/G0KFDY/z48TFz5swK93/ooYcWWNkAAAAAAAAAAACoWwoiwK6oqChOPvnk+Oc//5mXAQAAAAAAAAAAoKYVRIDd1VdfHZdeemlebteuXfTq1Ss6d+4cDRo0qOmiAQAAAAAAAAAAUEcVRIDdf//73zzfdttt4+67744WLVrUdJEAAAAAAAAAAACo4+pHAfjss8/y/JJLLhFcBwAAAAAAAAAAQEEoiAC74ox1yyyzTE0XBQAAAAAAAAAAAAonwK579+55/u2339Z0UQAAAAAAAAAAAKBwAuwOPvjgPP/3v/9d00UBAAAAAAAAAACArGEUgH322ScGDBgQV155ZTRv3jxOPvnkaN26dU0XCwAAAAAAAAAAgDqsIALskptvvjmGDRsWF1xwQVxyySWx1FJLRZMmTSo8ZsiQIQusfAAAAAAAAAAAs+t4w30ly299FhFXuEcAtUlBBNhNmjQpNttss3jvvffy39OnT48vvviiposFAAAAAAAAAABAHVYQAXbnnntuvPnmm3l5zTXXjC233DI6d+4cDRo0qOmiAQAAAAAAAAAAUEcVRIDdfff9b7rUww8/PK655pqaLg4AAAAAAAAAAABE/UK4B8OHD8/zk08+uaaLAgAAAAAAAAAAAIUTYNeiRYs8X2KJJWq6KAAAAAAAAAAAAFA4AXZrr712ng8dOrSmiwIAAAAAAAAAMNdGH7R7jNq+d546/WN3dw6glimIALsjjzwyz88999yaLgoAAAAAAAAAAAAUToDdDjvsECeddFL0798/dtlll3jrrbdi2rRpNV0sAAAAAAAAAAAA6rCGUQAaNvz/xXjwwQfzlDRo0KDC46ZPn17tZQMAAJhfM2bMiKFDh8aYMWOibdu2sfLKK0fjxo2r7MaOGzcuPv3005gwYUI0a9YslltuuejQocMCK191Pz4AAAAAAIA6ncEuNcakqbz15U0AAACFLNVbLrrooujcuXN07949Nttss1hzzTVjscUWi9NOOy1+++23+Tr/c889F717985Bbeuvv3706dMnevbsGR07dowePXrE/fffX63lq+7HBwAAUFPSSEupI9GQIUPiyy+/rPLzp/rUiBEj8rmnTJlS5ecHAABqWQa7J554oqaLAAAAUKWKiopi7733jnvuuSf/vfjii8cKK6wQ33zzTXz99ddx4YUXxuDBg3N9qFGjRpU+/y233BIHHHBAvk6y4oor5muMHj06hg0bFu+9917stttuccEFF8Spp55a5eWr7scHAACwIM2cOTPXX1555ZU8pfrMr7/+mrelDk2vv/56lVzn559/jjPPPDPuvPPOvFw80tMWW2yR61Grr756lVwHAACoZQF2W221VU0XAQAAoEpdf/31JcFn55xzTpxxxhnRoEGDHJjWr1+/OOyww3IGuhQAd9ZZZ1Xq3KmR55hjjsnn6tq1azz88MM5c1yxNFzs7rvvHh988EG+7n777ZezzFVl+arz8QEAACxokydPju22267k79RRqHnz5jFp0qQqu8b3338fvXr1iq+++ir/3aFDh3yN1FEpBfelOtQjjzwSffv2rbJrAgAAtWSIWAAAgNqW+eAf//hHXt52223j7LPPzsFnSb169eKQQw6JQw89NP99ySWXVLrB5v3334/x48fn5TRE66zBdUnKJHfNNdeUDDv02muvVWn5qvvxAQAALGipTpPqN+edd1688MIL8csvv1R5goh99tknB9e1atUqd5RKGcjT36mT1FprrRVTp06NP/7xjzFmzJgqvS4AADB/BNgBAABUsRTQNnz48LycMs2V5dhjj83ziRMnxuOPP16p87do0aJkebHFFitzn06dOpW5f1WUr7ofHwAAwILWtGnTeOyxx+Jvf/tbbLzxxvnvqvT888/nKbn66qtjhx12KNm23HLL5YC7lM0uBfZdfPHFVXptAABg/giwAwAAqGKvvvpqnjds2DB69+5d5j4rr7xyybCtxfvPrZVWWimWXXbZvHzbbbeVuc9NN92U5+3bt48NNtigSstX3Y8PAACgtrnrrrvyfIkllog999xzju1dunQpWX/33XdHUVHRAi8jAABQNgF2AAAAVWzo0KF53rVr1wqzHqShXJOPP/64UudPgW233357zl6XAun69OkT1157bc540K9fv9hpp53i3HPPzcMOpQC8RRddtErLV92PDwAAoLYZNGhQnm+22WZ5ONqy9O3bN89TxvA0dCwAAFAYGtZ0AQAAAGqbsWPHVjh8a7Hi7T/++GOlr5Gy0n300Ufxj3/8I6644op47rnnSm1PmQ8uueSSnB2hqstXnY+vXr16c70vAADAwmDq1Knx+eefl2T7Ls+s21J9b5llllkg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g/vvv78kuO7AAw/My61bt47JkyfHeeedF+eff37cfvvtscEGG+RscJWxzTbb5KkiK620Ug6Q23TTTWOZZZap0uNntd9++8Xll19eqfIDAAAUshkzZsRLL70UH3/8cfz222/RtWvX2HLLLXNnpXk1adKkuOyyy353v3XXXTf69u07z9cBoGY033GPkuWJn4UAO4BaRoAdAABANTj99NPzvFevXnH99ddH/fr189/NmjXLAXaffvpp9O/fP/7+97/HIYcckjPcVZVBgwbFJ598kpcPOuigBX48AADAwuqZZ56Jgw8+OL799ttS65s0aRJnn312nHLKKZXOQp5MmDAhzjzzzN/dL3XAEmAHAACFRYAdAABAFXv33XdLAtROOOGEkuC6WaVGmRRgN2bMmNyAs+2221bZ9fv161cyfOsuu+yywI8HAABYGD377LO5bjZt2rRo37597LTTTtGiRYt4/vnn44MPPojTTjstfvnll7jgggvm6zqHH354tGvXrsxtKcs5AABQWATYAQAAVLEXX3wxz1Ng3eabb17mPmuvvXZusBk7dmzev6oC7NKwrilwL9lnn31yloXqPv7XX3+NUaNG5X07dOgQDRuqagIAAAuXKVOmxIEHHpiD69Zcc80cbFccBFdUVBRHH310XH311XHRRRfFrrvuGuuss848X+vYY4+NlVZaqQpLDwAAVKc50ygAAAAwXz7++OM879KlS7Rq1arMfdKQQsUNKsX7V4U777wzJk+enJfTsEbVffwtt9ySH+MyyywTnTt3jtatW8d2220XL7zwwjyUHgAAoGbcdddd8d133+XlG2+8sVSGuVR/u+yyy2LppZfOwXYXX3yxpwkAAOoQAXYAAABVbPTo0Xm++OKLV7hfCkibdf+qcMMNN+R5yqaw+uqrV/vx48aNi2bNmkW3bt2iefPmMWnSpBgwYEBsuummcfrpp1f6+qnhqqIJAACgOjz44IN53qNHjzzNrlGjRrHffvvl5VTn+e233zwRAJSY9PC9MeHOG/PU4sV73RmAWsa4PQAAAFUsBZklTZs2rXC/FJiWTJgwoUqu+8EHH8Rbb701z9nrKnP8euutF3vssUdsueWWeajbZMaMGXm42+OPPz7ef//9OP/882PZZZfNwywBAAAUsuK60EYbbVTuPr17987zlPV76NCheSjZefHuu+/GwIEDc12wTZs2OaBvrbXWivr15cUAWFhNeuS+mDl6VF5u0aZTRKs9arpIAFQhAXYAAABVrLhRZObMmRXuN3369Dxv2LBqqmb9+vUrCdzbc889q/X4v/3tb3Osa9CgQWy22WYxaNCgWH/99XOD05lnnhkHHHDAXGefS8MtVUQWOwAAoKpNnDgxvv/++7ycOgmVZ7nllitZ/uSTT+Y5wG6vvfaaY126bhp6dpdddpmncwIAANVHVxgAAIAq1rJly5JGmooUby/ef36k4YnuuOOOvLz77rtHq1atFujxs0qP57TTTsvLI0eOjA8//HCezwUAAFDdxo0bV7Lcrl27cvdr27ZtyfIvv/wyT9fq0qVL7LDDDnHsscfGcccdF3369Mmdlb744ovYdddd49JLL630OVNHpIomAABg/shgBwAAUMW6du2a599++22F+xVvL95/fjzwwAPx008/zfPwsPN7/OzSEEfFvvvuu1h99dXn+5wAAADV4ddffy1Zbty4cbn7NWnSpGQ5DRNb2Y5Izz//fGy66aZzbPvss89it912iw8++CBOOeWUHHQ3r9nxAACAqieDHQAAQBUrDiYbO3ZsuUF2U6ZMiY8++igvd+/efb6vWTy860orrRQbbbTRAj9+XhuoAAAAatqsgXNTp04td79UjyvWtGnTSl2jefPmZQbXJcsvv3wMGDAgmjVrFjNnzoyrr766UucuKiqqcAIAAOaPADsAAIAqtuWWW5YMw/Pggw+WuU9qPEnDsibbbLPNfF3vq6++ioEDB+blgw46aIEfX5annnqqZHnFFVesknMCAABUh9atW5csF2f2LsvPP/9csrzoootWaRnS0LF9+/bNy6+99lqVnhsAAJg/AuwAAACqWOfOnWPrrbfOyxdddFHOZDd7drezzz47L6+11lqlhlMtNnz48Bg2bFh8+eWXv3u9G264IWclaNSoUey3336VLm9lj581O11Z3nrrrbj44ovz8gYbbFAlQ+ACAABUlzR8a6dOnfJyRXWwL774olo7EqUgu98L8gMAABY8AXYAAADVIAWYpSGDvv/+++jZs2fceeed8f7778dDDz0Uf/jDH/LwsA0bNozLL7+8zOMPPvjgWHnllXM2vIrMmDEjbr755ry8/fbbR8eOHStVznk5/oorrohVV101zjrrrLjvvvvi5Zdfjg8//DBnrTv22GPzELMTJkzI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2+blAQMGVOPTAQAA1CZFRUUVbq/qYX9ScFy6ZqNGjWK//farcN/LL788/va3v8U666wTbdu2zes6deoUBx54YLz11ls5+1w610knnbRAH186Z0UTAABAdSge6WjjjTeOdu3albnPrrvumuevv/56/PTTTwV1fgAAoPrU+QC7zz//PH799dd8M1LDUnnWXXfdPP/qq69kXwAAACrUokWLUpneylO8vWXLlvN9R1MGultuuSUvp2GFOnbsOM/nSoF2Kfgu+fTTT2PYsGE1/vgAAACqS+rMM2TIkLluK5p1/0I4PwAAUL3qfIDdN998U3IzunbtWu6NKt6WKjXffvttNT8tAADAwiwNsZoMHz68wv2KtxfvPz9SNoSRI0fm5YMPPni+z9ezZ8+S5dTRqKYfHwAAQHVJ2eImTpw4121Fyddff10w5wcAAKpXw6jjJkyYULJcUVaFWbfNeszvqVevXpXuB/zO/9LV7hAUgj/VdAGAiH18v6xJq622Wp6PGjUqfvjhh1hsscXm2Gf69Onx8ccfl9p/foeHTZZccsnYcsst5/t8sw7HWr9+/Rp/fLNTh6KQqIdQqK7+1zk1XQQokzojhape3FzTRaCWthVpi6q71tVmAXVY3ft91nse1C3r1rHvOXU+g11q9CnWoEGDcm9Uw4YNyzwGAABgdltssUXJ8mOPPVbmDXr++edLMhjMuv+8SIFuAwYMyMsHHHDAHAFx8+Kll14qWV5mmWVq9PEBAAAszG1F2qIAAGDhVucz2DVv3rzkZkyePLncGzXrthYtWsxT1gfqruLsGl4P4P8Q6jqfidQVKSAtDbH66quvxkUXXRR77rlnNGvWrGT7zJkz4+9//3teXm655UoNx1ps2rRpMWPGjBws17hx4wqvd8stt+QGm/Q/duCBB853+dNQROeff35eXmWVVWL55Zev8sc3r3ynLhze0ylEXpcUKq9NCpXXJiyYtqK63Bblfabu8tzXXZ77ustzX3d57uumenUsDqbOZ7Dr2LFjqawP5fn+++9Lljt06FDtTwwAALBwu+SSS3Lmg88++ywP2ZqC0X755Zd47733Yqeddsp/F+9XVsa57bffPpo2bZoD3H7PjTfemOd9+vSJpZZa6nf3/9e//hXbbrttPm7w4MExfPjwPGTRJ598Epdddlmss846eejXVK60b3U8PgAAgELRrl27ksx11dFWVN3nBwAAqledz2C38sorl9yMoUOH5gapsqRtSevWraNz587V/LQAAAALu5S17dprr40jjjgiXnnllejVq9ccvbtSlrgUjDa/Q7l++umnefnggw+eq2NSdrzHH388T+Vp2bJlLn/fvn1r9PEBAABUt5Q1fNlll811q+L2oLLMum3VVVctmPMDAADVq86nEWjVqlWsscYa+WY888wz5d6op59+Os979+5dzU8JAABQWxx00EHx1ltvxf7775+HVU1Ba127do0//vGPMWjQoDj11FMrbIBZZJFF8lSR2267Le+TOgLNbTDb8ccfn+s/Rx99dGy44Yax5JJL5uGH0jk22WSTOPfcc3PDTxr6tboeHwAAQCEpbv95/vnnY+bMmWXu89RTT+X5EkssketAhXR+AACg+tQrqiuD4VbgggsuiL/97W85PfeHH35YKqtdcWVn8803z8t33HFH7LXXXjVUUhZWdW3saShE/g+hMPhfBKg9vKdTiLwuKVRemxQqr034/1InpC233LLctqCRI0fGiiuuGBMnToy//vWv8a9//augzl+ovM/UXZ77ustzX3d57usuz33dVK+OxcHU+Qx2yVFHHRUdO3aMGTNm5IwPKciu2Kuvvhr77LNPXl5ttdVyJgYAAAAAAABqjy222KIky9yRRx4Zjz/+eMm2r776Knbccccc/Jayf59yyilzHJ8C5NZcc8083XrrrVV+fgAAoObIYPd/Xn755dxz6Ndff81/d+vWLQfcffvtt/nvdu3a5X1WWmmlmnu2WGjVtchdKET+D6Ew+F8EqD28p1OIvC4pVF6bFCqvTSht+PDh0bNnz/juu+/y34svvng0b948vvzyyzysa8OGDePBBx+M7bbbbo5b9/XXX+e2peSSSy6JE088sUrPv7DyPlN3ee7rLs993eW5r7s893VTvToWByOD3f/ZaKON4p133okddtghGjVqlHsLpeC6pk2b5jTd77//vuA6AAAAAACAWqpLly65reiQQw6JVq1axffffx+ff/55bjzs06dPvP766/MV/Fbd5wcAAKqHDHZlmDx5cowYMSIaNGgQSyyxRCyyyCLVdPsBAAAAAAAoNNOmTcttRVOnTo3OnTvnoVsrkvb7+OOP83JqW+rQoUOVnh8AABnobxkAAFIWSURBVKg5AuwAAAAAAAAAAACgDIaIBQAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwAwAAAAAAAAAAgDIIsAMAAAAAAAAAAIAyCLADAAAAAAAAAACAMgiwY6H2yCOPxJprrhnbbLNNTRelIO/LDjvsUNNFgejXr19+PR588MHuBhSgH3/8Mf+Ppum7776r6eIAAHXQoEGD8neRnj17Vul5H3zwwRr7zaC6HhO1i/oytc2pp56a3/vOO++8mi4KAAAAVKmGVXs6WLB++umneP/992PcuHEFcevvueeeuOCCC6Jbt275h/yavi8TJ06c53N8+umn8corr+QpnWvatGmVflxvv/123HrrrTFkyJCYPHlyLLnkkrlhY++9945GjRrNc9lYuIwaNSq/hlq3bl3TRQHKkN7f0/9o8ttvv9XKe/TUU0/F/fffH5988klMnTo1llhiiejbt2/ss88+0bRp03KPS4HBb731VoXnbtOmTQwcOLDMbelz+PLLL4/nnnsu39u11lorTjzxxFh66aXLPd/333+fA+TTZ2b//v2jfn39YQCKff3117HTTjvl5aeffjo6duzo5tQSv/zyS/4+0rx58yrvSFBTvxlU12OidlFfpjZ+Vqf3vg022KCmiwJUg88++yxuuumm/Lt/+q6z2GKLRZ8+feLPf/5ztGrVquDPz7xL7UR33HFHDB06NP+21rVr19hxxx1j9913jwYNGszzeT/88MN45pln8nzEiBH5e3v6rS0Fa//xj3/Mv6VRc4qKiuKBBx7I7YLpM75hw4axwgorxF577RWbbLJJlV/vv//9b56Szp07x+OPP17l12DuTJo0KW677bb8u3r6vbply5bRo0ePOOCAA2LllVeusmuk37/T7zup43/6HXzZZZfNv9un95dFFlnE01UDRo4cmT+L0/t++k2lffv20bt37zjwwAPn+3e4GTNmxEMPPRRPPvlk/sxP7SfptbX88svHtttuG9tvv732kBry+eefx6uvvpqf93fffTd/1qfvYek9oKp88MEHccstt5TEsKR2uq233jq30zVp0iQWGkWwELvpppuK0st4qaWWKioEV111VS7PiiuuWBD3Zdlll52n4xdbbLF8/OxTZR7XaaedVlSvXr0yz7P66qsXfffdd/NUNhY+//jHP/LzvvHGG9d0UYAyfP/99yXvz5999lmtukc///xzUd++fcv8LErTcsstVzRkyJByj0/vW+UdWzy1a9euzGMnTJiQP+9m379NmzZFH3zwQbnX3GWXXfJ+Tz75ZJXcA4DaZOjQoSXvp+oTtcujjz6an9fmzZtX6Xmvv/76GvvNoLoeE7WL+jK1zR//+Mf83nfYYYfVdFGAKnbNNdcULbLIImX+NtK1a9eid999t6DPz7yZOXNm0eGHH17u72Ibbrhh0dixY+fp3Ouss87v/u627777Fk2ZMsXTV0O/q/7hD38o97k54IADiqZPn15l1/viiy+KmjVrVnL+Qmn3rYvS7+XLLLNMmc97o0aNii6//PL5vkb67btLly7lvr5qup29rnrooYeKWrVqVeZz0rZt26Knn356ns/95ZdfFnXv3r3C9/y11167aPjw4VX6mPh9KZ6krOdjiSWWqNLfPho0aFDmdVZaaaX8GbCwkBIDmEOKSE+9UPbff/+4/vrrY7vttqvUXbrssstyJr/UuyUdO2DAgBzxfOGFF+be+ylCOWWyS9HPAFAd0mfQbrvtlnvYpN5vf/nLX3KP2Ndffz2uvvrq3Dsm9cpJPeLGjh1b4bkOO+yw3GunrOmFF14o85jzzz8/f96lbHX33Xdf7om36aabxs8//xyHHHJImcc8/PDDuVdoyvSaygUAAABQ0x555JH8u0rKzp8yVKaMQ6+99lr85z//iU6dOsW3336bM5CMGTOmIM/PvDvjjDNKMoqljHLpd7ZBgwbF2WefHY0bN87PU8oynn6Hq6z0G1mSMladeeaZOaNRakdK2W3WXXfdvC1l0EoZs1jw0u+qL730Us5ad8opp+TlZ599NrcbJinD1cknn1xl10u/l6aRsNJIWtSc9H+51VZbxZdffhlt27aNK664Ime1SlnHUtbCNBrOcccdF/fee+88XyO956dMZcOHD49VVlklrrzyynjxxRfz7/Z33XVX/m08ZTpjwUoj+aT3+fHjx+cshen9N73H33zzzTnDXBo9b+edd86ZTCtr5syZOSthylaa2mrS+3p6HaTnPL22UlbMJGWwTe89LPi4kPRZvO++++bP/JSdtipdf/31+XM+/V9vscUW+blPn/f//Oc/Y9FFF41hw4bl9530GbBQqOkIP5gfMthVTwa72XscHXTQQXPdY2D06NFFLVu2zPunXquze+GFF4rq16+ft1922WXzVD4WLnrkQ2GrrRns+vfvX/K4/vnPf86xPWU+Kv68OvLIIyvMYHf22WdX+vqpl2U69pVXXimV1S719ErrZ++RM378+NxrL21Pn6UAzEkGu9pLBjvqKvVlahsZ7KD2mTp1alG3bt3ybxkbbLBB/ntWn376aVGLFi0q/H2lJs/PvEu/XTVu3Djf+7/85S9zbH/ggQdKfnu77bbbKn3+jTbaqOjWW28tMwtaypy39957l5z//fffn+fHQeXdd999Jfc+PUezO+WUU/K2lI0o1dPn17XXXpvPt/322+f/cxnsas4JJ5yQ73+TJk3mGIUl/a9uttlmJZmtfv3110qff8SIEXmEl3SO3XffvWjatGll7jdx4sR5fgzMm169euXnJWUvHDduXKltY8aMKercuXPevu2221b63K+99lrJe0qqA5el+H0lTRWNOkTVmz0u5Nhjj62yDHbjxo0raRPbbrvtimbMmFFq+xtvvFHUsGHDCl8bhaZhTQf4UfukXkQpmjlFnqZx2dOYyV26dIlVV101RyCnTC7lSb0fHnzwwRypmsZeTmN5p2wvBx98cLRq1WqeyzQ/502ZaVL09EcffZSjthdffPFYccUVcwT96quvnveZMGFCHn+8OAPO119/HWuuuWap8xx77LFl9rSZ17L98MMPcc0118TLL7+cj0vlStluinuPzI927drN87F33313vh8NGjSIiy++eI7tG2+8cY5ST4/5uuuuyz0dKFvK+JfuZ3o9pKyAzz//fO4V8umnn8a4ceNy76A//elPed/Ro0fHo48+mnsTpF4fqSdBmzZtokePHjniPP3/zc01Uu+z22+/Pb8eU4bBFLGe/m9TxsGKfPXVV/n1OHjw4JgyZUosueSSOXth+j+ZG7/88kt+30j/b8XvGymL4q677lpuFqfZy56yQ6W/U0aq1GMu9XxJWafWWmutkmPSPUqZpFJ569WrF+utt14cf/zx0blz50q/DBfU9dO5UjR/OjadO93ftP+WW26Z/9/LG5e+pl8TdcEnn3ySe/Qk6bU7adKk/DpOPbrSZ2HqbXfPPffk7alnV+qFlXpipud/xIgRscgii8QyyyyT7+Uuu+yS3zd/7xrpOejXr19+XlKvkvR+vdlmm8Xhhx8eLVu2LLes6fqpbI899liMGjUqvxZSj+CDDjooGjVqNFeP94knnsjZ1T777LOS12F6naTXYVnXnr3s6bMhlT31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AAhAgB0AAAAAAAAAAAAAAAAAAAEIsAMAAAAAAAAAAAAAAAAAIAABdgAAAAAAAAAAAAAAAAAABCDADgAAAAAAAAAAAACADDZ//nxTr14906hRoxNq26Z3vd30TZs2Ncez7Dw+TpRtnJN8/vnnJ+T1ADhenJTdCwAAAAAAAAAAAAAAwPFmz549ZsWKFaZQoULmRJLe9XbTFylSxBzPsvP4OFG2cU6yd+/eE/J6ABwvyGAHAAAAAAAAAAAAAAAAAEAAMtgBAAAAAAAAAAAAAJDBmjVrZpYvX27y5MlzQm3bE3W9AQDHLwLsAAAAAAAAAAAAAADIYKeeeqqpV6/eCbddT9T1BgAcvwiwAwAAAAAAAAAAAAAgjj179pjx48ebefPmmR9++MGcfPLJply5cqZ69eqmc+fO9l+/+fPnm969e5uCBQuazz//PHCeO3fuNKNHjzaffvqpnX+pUqXMX/7yF9OjRw+bBU7Tn3LKKXZeftOmTTOPPfaYKVOmjJk5c6b97rhx48yqVavMoUOHzJlnnmluvPFGc8UVV3jTLFq0yLz66qtm7dq19ju1atWy8z/nnHNC1/nIkSPmzTffNLNnzzabN282uXLlMhUrVjStWrUynTp1MiedlDrkIJH13r17t11vbctff/3VW+9bbrklW/aVn7bTlClTzIoVK+x8SpYsaapVq2bXt2HDhqm+v3//frt9PvnkE/P999+bn376yRQuXNhu3/bt25vmzZuna10WLFhgJk+ebFavXm327t1rTj/9dJslUMeI/g6TWds4+thbuXKlee211+z20vF8zTXXmEceeSTFcui403L8+OOPdl/UqFHD7ocWLVqk2pZatz///NMedzVr1kzxufbLoEGD7N/33nuvueGGG1J8rmO0TZs29jj9+OOPTbFixbzPvvzyS/POO+/Y43/Lli3m6NGjpmzZsvb3brrpJhsYGqRp06Zm3759ZsiQIaZx48Z2XebMmWO2bdtmDh48aI93naOi40XbXMeC2+Zax4w6rgFkowgAAAAAAAAAAAAAAAi1fPnySOnSpSPqYg97jRs3LsU07777rn2/UKFCgfNcvHhxpESJEoHzKl++fGTIkCH27yJFiqSa9uWXX7afVaxYMfLMM89EcufOHTiffv362e/36dMn8PO8efNGZs6cGbh869ati1SvXj10fWvXrh357rvvUk0X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" alt="scaled_suite_summary" style="max-width:100%;height:auto;" />
````

````raw
task,modality,agent_candidate,agent_score,global_candidate,global_score,random_mean,random_sd,oracle_candidate,oracle_score
heldout-cancer-half-features,tabular,sgd-log_loss-a0.01,0.935672514619883,sgd-log_loss-a0.01,0.935672514619883,0.9251345029239765,0.010168222980760597,linearsvc-c12.743-b1,0.935672514619883
heldout-diabetes-above-median,tabular,sgd-log_loss-a0.01,0.7819548872180451,sgd-log_loss-a0.01,0.7819548872180451,0.6971804511278197,0.08559587375127735,linearsvc-c12.743-b1,0.7894736842105263
heldout-digits-noisy,image,sgd-log_loss-a1.67e-05,0.9095238095238096,sgd-log_loss-a0.01,0.9261904761904762,0.9100988095238095,0.013950093653112679,knn-k15-distance,0.9333333333333332
heldout-digits-noisy-parity,image,knn-k1-uniform,0.9404761904761904,sgd-log_loss-a0.01,0.8761904761904762,0.8832916666666666,0.041920261496638495,knn-k15-distance,0.9476190476190476
heldout-digits-shifted,image,sgd-log_loss-a1.67e-05,0.9642857142857144,sgd-log_loss-a0.01,0.9428571428571428,0.9442619047619049,0.014534057911482849,sgd-log_loss-a1.67e-05,0.9642857142857144
heldout-iris-setosa,tabular,knn-k15-distance,1.0,sgd-log_loss-a0.01,1.0,1.0,0.0,logreg-c0.001-b0,1.0
heldout-text-computer-systems,text,sgd-log_loss-a1.67e-05,0.7880952380952381,sgd-log_loss-a0.01,0.8380952380952381,0.7041476190476191,0.13840347277209816,sgd-log_loss-a0.01,0.8380952380952381
heldout-text-religion,text,linearsvc-c12.743-b1,0.7642857142857142,sgd-log_loss-a0.01,0.8214285714285714,0.687847619047619,0.10458770602246527,sgd-log_loss-a0.01,0.8214285714285714
heldout-text-science,text,linearsvc-c12.743-b1,0.8738095238095238,sgd-log_loss-a0.01,0.9023809523809524,0.7587392857142856,0.15041360944269938,sgd-log_loss-a0.01,0.9023809523809524
heldout-wine-class-zero,tabular,sgd-log_loss-a0.01,1.0,sgd-log_loss-a0.01,1.0,0.9738055555555555,0.03798234891913228,linearsvc-c12.743-b1,1.0

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_405f476208b7", "created_at": "2026-07-17T19:47:05+00:00", "title": "Hugging Face T4 evidence run submitted"}
-->
Substantive scaled rerun: https://huggingface.co/jobs/binzhango/6a5a8675bee6ee1cf4ecdd6e. Hardware: one Nvidia T4 (16 GB), t4-small. Command: UV execution of scripts/run_scaled_suite.py with all 19 tasks and 120 actions. Timeout: 30 minutes; maximum hardware charge 0.20 USD at the current 0.40 USD/hour rate. Persistent output mount: https://huggingface.co/buckets/binzhango/ml-agent-repro-v2-jobs#scaled-suite. Initial status after submission: SCHEDULING.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e2548f909443", "created_at": "2026-07-17T20:00:34+00:00", "title": "T4 retry after mount-init failure"}
-->
The first T4 Job failed before execution because the custom read-write Bucket mount init container exhausted retries: https://huggingface.co/jobs/binzhango/6a5a8675bee6ee1cf4ecdd6e. Resolution: the exact 20 Newsgroups cache was staged at https://huggingface.co/buckets/binzhango/ml-agent-repro-v2-jobs#datasets/20newsgroups and retry https://huggingface.co/jobs/binzhango/6a5a89c6bee6ee1cf4ecddc5 reads it through a read-only mount while writing outputs to the automatic jobs-artifacts Bucket.


---
<!-- trackio-cell
{"type": "code", "id": "cell_ebe1bce27032", "created_at": "2026-07-17T20:05:45+00:00", "title": "Fail-closed verification of downloaded T4 Job results", "command": ["rtk", "env", "UV_CACHE_DIR=/tmp/uv-cache", "uv", "run", "scripts/verify_scaled_suite.py", "--results", "outputs/hf_jobs/scaled-suite/results.json"], "exit_code": 0, "duration_s": 0.591}
-->
````bash
$ rtk env UV_CACHE_DIR=/tmp/uv-cache uv run scripts/verify_scaled_suite.py --results outputs/hf_jobs/scaled-suite/results.json
````

exit 0 · 0.6s


````python title=verify_scaled_suite.py
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

````


````json title=results.json
{
  "diversity": {
    "candidate_mean_distances": {
      "knn-k1-distance": 5.920928137738175,
      "knn-k1-uniform": 5.920928137738175,
      "knn-k11-distance": 5.7604195837888446,
      "knn-k11-uniform": 5.918637095703093,
      "knn-k15-distance": 5.786383404572587,
      "knn-k15-uniform": 5.977106011084752,
      "knn-k21-distance": 5.717482750517408,
      "knn-k21-uniform": 6.261726571477978,
      "knn-k3-distance": 5.896660990837703,
      "knn-k3-uniform": 5.864208784824316,
      "knn-k31-distance": 5.784504838015107,
      "knn-k31-uniform": 6.218735579026524,
      "knn-k45-distance": 5.722057147498744,
      "knn-k45-uniform": 6.138888562944115,
      "knn-k5-distance": 5.969836064665068,
      "knn-k5-uniform": 5.894167726290663,
      "knn-k7-distance": 5.987309885181486,
      "knn-k7-uniform": 6.077970527928121,
      "knn-k9-distance": 6.064313515693472,
      "knn-k9-uniform": 6.032269188425973,
      "linearsvc-c0.001-b0": 4.184400168399826,
      "linearsvc-c0.001-b1": 4.337982766820017,
      "linearsvc-c0.0020691-b0": 3.140888508695204,
      "linearsvc-c0.0020691-b1": 3.698119948145655,
      "linearsvc-c0.0042813-b0": 3.2268643728487083,
      "linearsvc-c0.0042813-b1": 3.672438735318998,
      "linearsvc-c0.0088587-b0": 3.1141391337375466,
      "linearsvc-c0.0088587-b1": 3.287561421402814,
      "linearsvc-c0.01833-b0": 3.2767368711344536,
      "linearsvc-c0.01833-b1": 3.7221583072947673,
      "linearsvc-c0.037927-b0": 3.93322669661403,
      "linearsvc-c0.037927-b1": 3.6276250779605266,
      "linearsvc-c0.078476-b0": 3.814695614348074,
      "linearsvc-c0.078476-b1": 3.8073904660741844,
      "linearsvc-c0.16238-b0": 3.500392509826375,
      "linearsvc-c0.16238-b1": 3.570190439786032,
      "linearsvc-c0.33598-b0": 3.326273747151447,
      "linearsvc-c0.33598-b1": 3.334821320703031,
      "linearsvc-c0.69519-b0": 3.199199700422467,
      "linearsvc-c0.69519-b1": 3.2019687932410874,
      "linearsvc-c1.4384-b0": 2.7921593330098897,
      "linearsvc-c1.4384-b1": 2.79404977216995,
      "linearsvc-c1000-b0": 2.930748540300963,
      "linearsvc-c1000-b1": 3.0109394541204413,
      "linearsvc-c112.88-b0": 2.927468477018071,
      "linearsvc-c112.88-b1": 3.012403134534411,
      "linearsvc-c12.743-b0": 3.2587983047659965,
      "linearsvc-c12.743-b1": 3.264754029076523,
      "linearsvc-c2.9764-b0": 2.9716370484039314,
      "linearsvc-c2.9764-b1": 3.076852476286831,
      "linearsvc-c233.57-b0": 2.929636364258152,
      "linearsvc-c233.57-b1": 3.0115091256159805,
      "linearsvc-c26.367-b0": 3.122990456377266,
      "linearsvc-c26.367-b1": 3.122990456377266,
      "linearsvc-c483.29-b0": 2.930317066698811,
      "linearsvc-c483.29-b1": 3.01182858069665,
      "linearsvc-c54.556-b0": 3.0099546300729165,
      "linearsvc-c54.556-b1": 3.12523821451038,
      "linearsvc-c6.1585-b0": 3.162048030574767,
      "linearsvc-c6.1585-b1": 3.0771159223782956,
      "logreg-c0.001-b0": 4.705791696812804,
      "logreg-c0.001-b1": 4.04194982854558,
      "logreg-c0.0020691-b0": 4.162049004971668,
      "logreg-c0.0020691-b1": 3.2809513919718154,
      "logreg-c0.0042813-b0": 3.225816281085642,
      "logreg-c0.0042813-b1": 2.773172878154288,
      "logreg-c0.0088587-b0": 2.7566841077156425,
      "logreg-c0.0088587-b1": 2.837734332376239,
      "logreg-c0.01833-b0": 2.8071350439121527,
      "logreg-c0.01833-b1": 2.935443126267557,
      "logreg-c0.037927-b0": 2.7140149465168055,
      "logreg-c0.037927-b1": 3.080719045413493,
      "logreg-c0.078476-b0": 2.907610834399618,
      "logreg-c0.078476-b1": 3.0519183976356996,
      "logreg-c0.16238-b0": 2.7595068708835084,
      "logreg-c0.16238-b1": 3.08199827920311,
      "logreg-c0.33598-b0": 2.783915203611229,
      "logreg-c0.33598-b1": 3.0903846879339683,
      "logreg-c0.69519-b0": 2.763565800114331,
      "logreg-c0.69519-b1": 2.7342350664402995,
      "logreg-c1.4384-b0": 2.758859148481258,
      "logreg-c1.4384-b1": 2.8852617917202803,
      "logreg-c1000-b0": 2.8097979152946255,
      "logreg-c1000-b1": 2.8159542485110847,
      "logreg-c112.88-b0": 2.68008699491391,
      "logreg-c112.88-b1": 2.821449492743437,
      "logreg-c12.743-b0": 2.600281503802552,
      "logreg-c12.743-b1": 2.6025396178780986,
      "logreg-c2.9764-b0": 2.642538194444407,
      "logreg-c2.9764-b1": 2.636273952199353,
      "logreg-c233.57-b0": 2.6748612933431315,
      "logreg-c233.57-b1": 2.8205796201646236,
      "logreg-c26.367-b0": 2.6749682023762156,
      "logreg-c26.367-b1": 2.667106102285331,
      "logreg-c483.29-b0": 2.6709472634354703,
      "logreg-c483.29-b1": 2.8159542485110847,
      "logreg-c54.556-b0": 2.6790784291589906,
      "logreg-c54.556-b1": 2.6936745947307847,
      "logreg-c6.1585-b0": 2.585299022883983,
      "logreg-c6.1585-b1": 2.5761191863769377,
      "sgd-hinge-a0.000215": 3.1352329253911773,
      "sgd-hinge-a0.000774": 3.2233149978579947,
      "sgd-hinge-a0.00278": 2.8161353378358145,
      "sgd-hinge-a0.01": 3.471134260498808,
      "sgd-hinge-a1.29e-06": 5.566059446647918,
      "sgd-hinge-a1.67e-05": 3.608126435043192,
      "sgd-hinge-a1e-07": 3.3659516796884557,
      "sgd-hinge-a3.59e-07": 3.4559948544424572,
      "sgd-hinge-a4.64e-06": 3.440352226679868,
      "sgd-hinge-a5.99e-05": 3.3415140851245706,
      "sgd-log_loss-a0.000215": 2.9347221374898127,
      "sgd-log_loss-a0.000774": 2.900888729871019,
      "sgd-log_loss-a0.00278": 2.9690830759768545,
      "sgd-log_loss-a0.01": 3.6059445102129852,
      "sgd-log_loss-a1.29e-06": 4.73004442901241,
      "sgd-log_loss-a1.67e-05": 5.922580296344863,
      "sgd-log_loss-a1e-07": 3.6574861607006386,
      "sgd-log_loss-a3.59e-07": 3.460458482575095,
      "sgd-log_loss-a4.64e-06": 3.858623928461797,
      "sgd-log_loss-a5.99e-05": 3.3281483822569045
    },
    "empirical_p_ge_selected": 0.0001999600079984003,
    "random_max": 5.626313489495825,
    "random_mean": 3.698375000327932,
    "random_p95": 4.8145191444109035,
    "random_sd": 0.7370096892649434,
    "random_trials": 5000,
    "selected_ids": [
      "sgd-log_loss-a1.67e-05",
      "knn-k31-uniform",
      "sgd-log_loss-a0.01",
      "knn-k1-uniform",
      "logreg-c0.001-b0",
      "linearsvc-c0.001-b0",
      "sgd-log_loss-a1.29e-06",
      "linearsvc-c12.743-b1",
      "knn-k15-distance",
      "sgd-log_loss-a4.64e-06"
    ],
    "selected_mean_pairwise_distance": 5.851605329297916,
    "selection_method": "standardize per training task, then deterministic maximin farthest-point sampling"
  },
  "environment": {
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "platform": "Linux-6.12.94-123.180.amzn2023.x86_64-x86_64-with-glibc2.36",
    "python": "3.12.12 (main, Feb  3 2026, 05:51:02) [GCC 12.2.0]",
    "sklearn": "1.9.0",
    "wall_seconds": 230.43339289200003
  },
  "generalization": {
    "agent_mean": 0.8979532163742692,
    "agent_minus_global_mean": -0.0045238095238095185,
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
    "random_mean": 0.847499074770259,
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
    "corner_latency_budget_seconds": 0.5907439558999841,
    "error_rewards": [
      0.0
    ],
    "neutral_rewards": [
      0.5
    ],
    "successful_edit_max_reward": 1.0,
    "successful_edit_mean_reward": 0.45,
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


````output
PASS: exactly 9 training tasks
PASS: exactly 10 held-out tasks
PASS: at least 100 candidate actions
PASS: exactly 10 selected actions
PASS: selected-id manifest is complete
PASS: selected actions are unique
PASS: task manifest has 9 training rows
PASS: task manifest has 10 held-out rows
PASS: held-out tasks span image, tabular, and text
PASS: 5,000 random diversity controls
PASS: selected diversity is finite
PASS: selected set is more diverse than random mean
PASS: selected set exceeds random 95th percentile
PASS: ranker train/test counts are exact
PASS: ranker evaluation covers all modalities
PASS: agent accuracy is bounded
PASS: trained ranker beats random selected actions on held-out tasks
PASS: invalid/error trajectories receive reward 0
PASS: non-edit/corner outcomes receive reward 0.5
PASS: sigmoid reward evaluated on all held-out edits
PASS: success rewards are sigmoid bounded
PASS: error branch comes from an executed invalid action
PASS: evidence run uses real 20 Newsgroups text, not smoke corpus
ALL_VERIFICATIONS_PASS

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_dd74d215d979", "created_at": "2026-07-17T20:06:00+00:00", "title": "Completed T4 result and persistent artifacts"}
-->
Remote T4 result: COMPLETED in 243 seconds (230.433 seconds experiment wall time) at an estimated hardware charge of about 0.03 USD. Job: https://huggingface.co/jobs/binzhango/6a5a89c6bee6ee1cf4ecddc5. Immutable/persistent artifacts: https://huggingface.co/buckets/binzhango/jobs-artifacts#20260717T200004-ddae91/scaled-suite-retry. Newly measured held-out mean was 0.897953; random-action mean was 0.848451 and global-best was 0.902477. Maximin behavioral diversity was significant against 5,000 random subsets (empirical p=0.00019996). The downloaded result bundle passed every fail-closed verifier assertion.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_68561e42b969", "created_at": "2026-07-17T20:06:01+00:00", "title": "T4 held-out results"}
-->
````html
<img src="data:image/png;base64,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9TbS4tza+/pP6tzdvn1bTZo00caNG232ExkZmWAJgLSQI0cOvfHGG5JiZ9PbtGmTzfoHDhwwj/l//vMfh8yOCAAA8KQ7ceKE+bl69eo268YvP378uF22f+/ePTVr1kzLli2zWB4SEqJXXnlFW7dutdrHyZMnVbt2bc2ePdvivb8Ue/+7dOlS1a5dW1euXElVrL/++qt69Ohhzsz+KB8fHzVu3Fi3b99OVf/WREVFqVOnThozZozVcasUO1t2/fr1tWfPHrtuPyM7fvy4Tp06ZbPO7du39dFHH2ngwIEWyy9evKhmzZrpwIEDNvu5f/++fHx8UhXn5MmT1bVrV927d09lypTRrl271KxZs3SNAQAA4HE4OTklmLH5rbfeUlhYmAMjSujQoUOqVauWfv31V0VGRlqtFxISIj8/vzTr4/jx46pTp47mzJljc2yyePFi1alTJ8GsZXHeeOMNTZo0yWYMUuwMZAEBAXZvn1nY41xI0i+//KKXX37Z6rHatm2bGjdurDt37tgt9rQSGhqqRo0aWXz2EvfMKDIyUi+88IKGDRummzdvWu3r+PHjev7557Vw4cIE35cvX14lS5aUFDtGteTR75Oq5+7urqZNmyYq37Vrl9XxeZwLFy6oa9eumj9/vs16cQYNGqSvvvpKDx8+TPB9/GeagwcP1kcffaQHDx4kam8YhubNm6du3brZfA4aGRmp5557zlyhyZro6Ght3749yX/PwNOMJWIBPHE2bdqUaJroyMhI+fv7a+fOneZAq0aNGvr666+t9rN9+3b17dvXvNktUKCAOnXqpBIlSih37ty6fv26du3apW3btkmKXZZowoQJGjlypNU+X3/9dYWHh6tChQrq1KmTihcvrsDAQG3fvt18SHPp0iVNmTJFn3zyidV+SpcurcaNG6tQoUIqWLCgYmJidOPGDW3atMkcTK1evVqzZs1Sv379ErR98803zZvM2bNnq3bt2la3Ezd1cly7lPr555/NZUslqWDBgurRo4fKlSunhw8fas+ePVq5cqWioqIUERGh1157TX5+fjanwX777bcVHR2tsmXLqnPnzipVqpTc3d0tTv/8OMfJnv73v/8leEhXunRpvfDCCypRooTu3LmjLVu2aOPGjfr1119TvNxwfPY8t2lx/Sd17ubPn28+YHRxcVG7du1Uu3Zt5c2bV+Hh4bp+/bou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u/I9+eSTudZ06tTJ9Do+Pr4A6e+vsJ8PAAAAAApDYmKiXn31Vf3000/KyMjI1zVdu3alwQ4AAAB4xFlFg10WFxcXjR49Ws8++6z8/PxkMBgsHckkOTlZ0t0d9QAAAABYtz179ujdd981vb969aq++eab+143bNgwszTYrV+/3rQDQo8ePXKt6dixo0qVKqVbt25p3bp1BWpAq127tkqUKKHbt2/r6NGjCggIyFFz5MgRSXd/MNSgQYO/8RT3VtjPBwAAAADmZjQa9cwzz2jTpk0yGAzy8/PT+fPnlZSUpEaNGik9PV1//PGH7ty5I0mqW7euvLy85OzsbOHkAAAAACzNKhrsWrVqpb179yohIUFTpkzRlClT5OXlpcDAQLVv317t27dXzZo1LZYvOTlZ69evlyT5+/tbLAcAAACA/ClXrpyaN29e4OuqVq1qlvsfO3ZMklS2bNl77nTg4OCgevXqad++fab6/HJzc9O7776rd955RxMnTpSLi4ueffZZlSxZUmlpaVq7dq1GjBghSRo5cqRq1Khxz7U2b96sNm3a6PLlyypZsqR8fX0VFBSkIUOGyN3d3SLPBwAAAADmFhISok2bNqlEiRJat26dAgMD1aRJE+3fv19fffWVmjRpovj4eH3wwQeaPn26bt++rUWLFqls2bKWjg4AAADAwqyiwW779u1KTk5WeHi4QkNDFRYWpoMHD2rx4sVavHixJKl8+fKmZrv27durUqVKZrl3YmKiFi5cqJ49e8rDwyPH/OXLl/XCCy8oJiZGkjR8+HCz3BcAAABA4enZs6d69uxpsftHRkZKkipWrJhnXcWKFbVv3z5dvny5wPd4++23VblyZX3yyScaOHCgBg0aJFdXV928eVNGo1E1a9bU22+/rVGjRuW5zv82v0VERGj9+vX64IMP9OOPP6pr1645rimK5wMAAAAAc1q1apUkadCgQQoMDMy1xs3NTR999JHs7Ow0depUvfPOO/r666+LMCUAAAAAa2QVDXaS5OjoqE6dOqlTp06SpJs3b+r3339XaGioQkNDdezYMc2fP1/z58+XJFWrVk1nzpx54PsmJydr6NChGjFihOrXry8fHx+VK1dO6enpOn36tPbu3auMjAxJ0uTJk9W2bdsHvicAAACAh1tiYqIkycnJKc+6rPms+oJydXWVq6urpLvHHcXHx5vmnJ2d8zzKyNnZWUOHDlXHjh1VtWpVeXp6Kjo6WmvXrtVnn32mGzduqFevXgoNDVXr1q2zXVuYz2cwGPJdCwAAAAD5FRERIUkKDg42jWV9/sj6HijLW2+9pU8//VSLFi3SnDlzZGdnNV+nAQCslEOtusoo6y1JSrErI92wcCAAgFlZ7ScCV1dXde/eXd27d1dmZqY2bdqkSZMmaffu3ZKks2fPmuU+Li4uGjFihFavXq1Dhw7p0KFDOWqaNWum9957T126dDHLPQEAAAA83DIzMyXdv1nMxsYmW31BTJs2TW+++aZsbGw0ZMgQPffcc6pSpYqioqK0fPlyzZo1S4MHD9auXbv0xRdf5Lj+hx9+MN0/S6VKldSiRQv169dPrVu31o0bNzR69Gjt27evyJ8PAAAAAMwpNTVVkuTt7W0ac3BwkJTzR0FOTk6qVq2aTpw4ocjISFWpUqXIcgIAiie3Cf9nen3+tKSZlssCADA/q22wO3HihGn3ui1btujGjewt3v/7RdDfVbp0ac2ZM0dz5sxRRESELly4oCtXrshgMKhs2bJq0KCBHnvsMbPcCwAAAEDRS01N1cKFC7VhwwadOnVKN2/elJOTk6pUqaLAwEANGjRI7u7uZr1n1s5tKSkpedZlzd9vJ7j/dezYMb3zzjuSpKlTp2rChAmmOV9fX7Vp00aBgYHq1q2bvvzyS/Xo0cO0W3iWvD5T1alTR2+++aYmTJig/fv368KFC9m+UCrM5zMajXnOs8MdAAAAgL8jq7Hur983eXl5SZLOnz+foz6r6S4hIaEI0gEAAACwZlbTYHfu3DlTQ11YWJhiYmKyzRsMBvn5+aldu3Zq3759oRzVWrNmTdWsWdPs6wIAAACwjG3btql///66fPlyjrlDhw5p+fLlevfddzVnzhwNGDDAbPctV66cJCk6OjrPuqioKEnZd1DIj19//VUZGRkqXbq0xo0bl2tN165d1aBBAx0+fFg///xzjga7+2nfvr3pdURERLYGu8J+PgAAAAAwt6zPNJGRkaYxf39/rVixQqtXr9bQoUNN44cOHTLVZTXhAQAAAHh0WUWDXeXKlXXp0qUc47Vr1zY11AUGBsrT09MC6QAAAAAUR/v371dwcLBpF7UmTZqoWbNmcnd3V1JSko4eParff/9diYmJGjRokOzs7NS3b1+z3LtevXqS7jaYxcfHy83NLUdNZmam/vjjj2z1+XX27FlJUtWqVWVra3vPumrVqunw4cOm+oLIOipJktLT07PNFfbzAQAAAIC5BQUF6dNPP1VYWJiGDx8uSXr22Wf1r3/9SytXrtTYsWP1zDPP6OLFi3r77bdlNBrl5+fHKUcAAAAArKPBLqu5rmrVqmrfvr2pqY5dDgAAAAD8XSNHjlRKSoqqVq2qBQsWqFmzZjlqLly4oIEDB2rbtm365z//qa5duxb4uNbcBAYGSrp73OmGDRvUp0+fHDW7du0yHU2UVZ9fpUuXlvTfHeLuJWtn8Kz6gti7d6/pdaVKlbLNFfbzAQAAAIC5dejQQS4uLlq1apWSkpLk5OSk+vXr65VXXtHnn3+uzz77TJ999pmp3s7OLtt7AAAAAI8uG0sHkKTvv/9eFy9e1NmzZ/XNN9+of//+NNcBAAAA+NtOnz6t3bt3y8HBQevXr8+1uU66e0RQSEiIKlasqGvXrmndunVmub+fn5/q168vSfroo49y7AAnSR988IEk6bHHHst2HGt+NG3aVJJ048YNLVmyJNeaw4cPa/fu3dnq8+v69eumfJUqVcqxA11hPx8AAAAAmJuDg4NiYmIUGxub7UdIs2bN0vTp01W1alVJdxvr2rRpo82bN6tDhw6WigsAKGbSTh7X7aMHdfvoQTlcOG7pOAAAM7OKHewWLFign3/+WRs2bLB0FAAAAAAPgUOHDkm6u0NB9erV86x1cnLSc889p2nTpunQoUN69tlnzZJh6tSp6tatm/bv369+/fpp5syZ8vHx0fXr1/Xuu+9qzZo1kqQpU6bI3t4+x/X9+/dXaGiofH19tXPnzmxz/fr109tvv63r16/rpZde0rVr19S/f3+5uroqKSlJy5cv1/jx45WRkSEHBwe9/PLL2a6fN2+eDhw4oF69esnPz0+enp4yGAxKSUnR+vXr9cYbb5iOlf3ggw9kMBjM/nwAgEfDwjW7LR0BVugfTzW3dAQAj6hSpUrlGLOxsdH48eM1fvx4paeny8bGRjY2VrE/BQCgGImfPlmZsXdPkyjj7i25LLZwIgCAOVnFJ4RNmzZp48aNlo4BAAAA4CGRmJgoSfneGTurLiEhwWwZunbtqvfee0+StGTJEpUvX17u7u7y9PTUnDlzJEnDhg3TkCFDcr3++vXrunr1qq5du5ZjztXVVYsXL5azs7Nu3rypV155RW5ubnJ1dZWzs7MGDhyoq1evqkSJEvrxxx/l6+ub7fpr165pzpw56tChg8qWLasSJUrIw8NDLi4ueuaZZ3T69GnZ2tpq2rRpeu655wrl+QAAAADA2tjZ2dFcBwAAACAHq/iU4OHhIUlKTU21cBIAAAAAD4MyZcpIuntUbH5ERERI+u9nE3OZPHmy1qxZo8DAQNnZ2Sk+Pl4Gg0HNmzfXL7/8oi+//PJvr92+fXsdPXpUo0aNUpUqVST9t0HQx8dHgwcP1oEDB/SPf/wjx7VDhw7VN998o27duumxxx5TRkaGrl+/royMDFWpUkUvvfSSDhw4oDfeeMNizwcAAAAA5vT222+rb9++OnfuXKHUAwAAAHh4WcURsY0bN1ZISIgOHTqkFi1aWDoOAAAAgGKuefPmMhgM2r59u7Zs2aLAwMB71l64cEE//fSTJKlly5Zmz9KlSxd16dJFmZmZio+Pl6urq2xtbe973YIFC3T79m3Z2d37Y1vlypU1e/ZszZ49W7dv31ZiYqIcHR1zPfborzw8PPTSSy/ppZdekiSlp6fr5s2bcnFxKfBxrn/3+QAAAACgKK1fv1779+/X+PHjVbVqVbPXAwAAAHh4WcUOdsOGDZMkvffee8rIyLBwGgAAAADF3WOPPaYePXrIaDSqe/fu+vrrr3PsmJ2ZmamVK1fqiSeeUEJCgmrUqKH27dsXWiYbGxuVKVMm381n7u7u8vb2lqenZ77qS5QoIU9Pz/s21+XGzs5OHh4eBW6u+6uCPh8AAAAAWDOj0ShJMhgMFk4CAAAAwNKsosGuR48eGjt2rDZs2KB27dopJCREN2/etHQsAAAAAMXYrFmzVL58eSUmJmrYsGHy9PRUixYt1LlzZz3xxBPy8vJSjx49dPnyZZUqVUrff/99nrvFAQAAAAAeHVevXpUkOTo6WjgJAAAAAEuzim+P/volVnh4uMLDwyXd3QEhr18GpaenF3o2AAAAAMVTxYoVtXPnTr300kvasGGDkpOTtXv37hx1jz/+uObNm6cmTZpYICUAAAAAoDCkpaUpMzPT9D5rR7q0tLQcO5z/VUJCghYvXqyoqCjZ2dmpSpUqhR0VAAAAgJWziga7ex0L+9cPPgAAAABQUBUrVtT69et19OhRbdiwQadOnVJCQoIcHR1VpUoVBQYGKiAgwNIxAQAAAABm1qpVK+3fvz/HeOvWrfO9Rvfu3VWyZElzxgIAAABQDFlFg11ISIilIwAAAAB4iPn5+cnPz8/SMQAAAAAAxYCNjY2eeuopffHFF5aOAgAAAMAKWEWDXadOnSwdAQAAAAAAAAAAAA+JefPmKTEx0fT+pZde0qlTpzR37lzVqlUr12tsbGxUunRpVatWTc7OzkUVFQAAAICVs4oGOwAAAAAAAAAAAMBcGjRokO19zZo1lZqaqmbNmrHDOQAAAIACocEOAAAAQLG2Y8cOLVq0SJLUunVr9e7dO9tYQWRdDwAAAAB4uKxcudLSEQAAAAAUU0XeYBcfH2967ebmlmOsILKuBwAAAPDoOnLkiGbOnClJSk1NVe/evbONFUTW9QAAAAAAAAAAAIBkgQY7d3d302uj0ZhjrCCyrgcAAADw6PL19VWPHj0kSf7+/jnGCiLregAAAAAAAADIr5KtApWZEC9JupbpJp23aBwAgJlxRCwAAACAYi04OFjBwcH3HQMAAAAAAACAwuAyZKTp9ZnTkgp+uAYAwIoVeYPdwYMH8zUGAAAAAAAAAAAAAAAAAIAlFXmDXW5HLnEMEwAAAABzio+PV0xMjNzd3VWuXDmz1wMAAAAAAAAAAODRYGPpAAAAAABgbr/++qvq1Kmj9957r1DqAQAAAAAAAAAA8GigwQ4AAAAAAAAAAAAAAAAAgFwU+RGx93Py5Ent2rVLMTExSk1NzbN28uTJRRMKAAAAwEPt1q1bkiQHBwcLJwEAAAAAAABQ3KRuD1Pm//83xlLxpSS1s2wgAIBZWU2D3cWLF/XCCy8oLCws39fQYAcAAADAHPbv3y9J8vT0tHASAAAAAAAAAMVNwrdzlBkbI0lydfeWXGiwA4CHiVU02MXHxyswMFAXLlyQvb29GjZsqD179kiSAgICFBsbq4iICBmNRklS8+bNVbJkSUtGBgAAAGBFdu/erWXLlpneHzp0yDQ+ceLEe153584dnTx5UiEhIZKkli1bFmpOAAAAAAAAAAAAFC9W0WA3a9YsXbhwQRUrVtSWLVtUtWpVGQwGSdLWrVsl3d3hbsyYMVq2bJlcXFxMX4ABAAAAwMGDB/Xhhx/mGD906JCp2e5+2rVrpw4dOpg5GQAAAAAAAAAAAIozq2iwW7VqlSTpnXfeUdWqVXOtqVy5spYsWaKnnnpK69at07x58/Tyyy8XZUwAAAAAVqpChQrZmuOioqJ08uRJlS9fXrVr177ndQ4ODnrsscfUtm1b9evXTzY2NkURFwAAAAAAAAAAAMWEVTTYRURESJKCg4NzzGVkZMjW1laSZGNjo0mTJmndunX65ZdfaLADAAAAIEnq2rWrunbtanr/5ZdfasSIEeratau+/PJLCyYDAAAAAAAAAABAcWYVDXapqamSJG9vb9OYg4OD0tLSlJiYKDc3N9O4v7+/JOnYsWNFGREAAABAMdKuXTt99913qlWrlqWjAAAAAAAAAAAAoBizivOPshrrbty4YRrz8vKSJJ0/fz5bbWJioiQpISGhiNIBAAAAKG5q1aqlwYMHq2XLlpaOAgAAAAAAAAAAgGLMKhrsqlSpIkmKjIw0jWXtVLd69epstVnvsxrwAAAAAAAAAAAAAAAAAAAoDFbRYBcUFCRJCgsLM4317t1bkjR16lR99tln2rZtm2bOnKnRo0dLkoKDg4s+KAAAAIBiJSUlRTNmzFBAQIA8PDxka2srg8Fwz7/hw4dbOjIAAAAAAAAAAACsiFU02PXs2VOStGDBAtPYgAED1Lp1a926dUtjx45VQECAXnvtNSUlJcnT01PvvfeepeICAAAAKAbi4uLUvHlzjRkzRtu2bdP169eVmZlp6VgAAAAAAAAAAAAoRqyiwa5+/fpKTExUeHi4aczW1lZr167V6NGjVbZsWUlSqVKl9PTTT2vnzp2qXLmypeICAAAAKAZGjhypY8eOycvLS//5z39Mu2F369ZN69at0+zZs9W4cWNJkqenp5YtW6axY8daMjIAAAAAAAAAAACsjFU02EmSk5OTHB0ds425uLhoxowZunr1qm7fvq2UlBQtW7ZM1atXt1BKAAAAAMVBbGyslixZIklaunSpRo4cqdq1a0uSfHx8FBwcrFGjRmnPnj0aM2aM4uLi9PXXX6tmzZqWjA0AAAAAAAAAAAArYzUNdvfj4OBg6QgAAAAAiom9e/cqMzNTDRo0UEBAwD3rbGxs9PHHH8vPz08hISE6cOBAEaYEAAAAAAAAAACAtbOzdAAAAAAAMLfY2FhJUq1atUxjtra2kqTbt29nq7WxsVHPnj119OhRhYaGqlGjRkUXFAAAAAAAAECx59i9tzKTkyRJUSlO0iHL5gEAmJdVNdgZjUaFhIRo4cKF2rdvn+lLsbJly6pJkybq27evOnXqJIPBYOGkAAAAAKxZ6dKlJUklSpQwjTk5OUn6b/PdX7m7u0uSoqOjiyAdAAAAAAAAgIeJY48+ptdJp0WDHQA8ZKymwS4yMlL9+vXTtm3bcszFxcXpxIkT+vHHH9W2bVv98ssv8vHxsUBKAAAAAMVBhQoVJN39nJHF19dXknTo0CEZjcZsP9w5efKkJPFjHgAAAAAAAAAAAGRjFQ12N2/eVGBgoM6ePStJat68uYKCgrJ9KbZp0ybt3r1bv//+uwIDA7Vv3z65uLhYMjYAAAAAK1WvXj3Z2dkpIiLCNNawYUO5uroqOjpac+fO1dChQyXdbbibP3++JKl+/foWyQsAAAAAAAAAAADrZGPpAJL0wQcf6OzZs3J0dNSKFSu0a9cuvf/++xo+fLiGDx+u999/X7t27dKKFSvk6Oio06dPa+rUqZaODQAAAMBKubm5qW3btoqKitLOnTsl3T0udsSIEZKkl19+WU2aNFHr1q3VvHlzpaSkyMvLS88++6wlYwMAAAAAAAAAAMDKWEWD3dKlSyVJH374obp3737Puu7du2vatGmSpCVLlhRJNgAAAADF08SJEzVu3DglJSWZxiZPnqwePXpIkvbv368dO3YoLS1NHh4e+u233+Ts7GypuAAAAAAAAAAAALBCVnFE7OXLlyVJffr0uW9tnz599Oqrr5quAQAAAIDcBAUFKSgoKNtYiRIltHz5coWHhys8PFzJycmqVq2aevXqJVdXVwslBQAAAAAAAFCcJa9YpMzkuz/0dUpxknT/3gcAQPFhFQ12Tk5Oun79er52i3BxcZEkdpYAAAAA8LcFBAQoICDA0jEAAAAAAI+QM2fOaOPGjTp69KiioqJ048YNubu7q0GDBurTp4/q169v6YgAgL8peeViZcbGSJKc3L0lFxrsAOBhYhUNdv7+/goNDdXevXvv+yXXnj17JEkNGzYsimgAAAAAAAAAAADAA2nbtq22bt2a69zKlSs1ZcoUvfjii/riiy/k4OBQxOkAAAAA5MXG0gEkadiwYZKkcePGKTEx8Z51iYmJGjduXLZrAAAAAAAAAAAAAGt29epVSVKFChU0fvx4LV++XKGhofr6669Nm0p8++23evHFFy0ZEwAAAEAurGIHuz59+mjHjh2aOXOmGjZsqIkTJ+rJJ59UhQoVJEmRkZHauHGjpk2bprNnz+q1115Tr169LJwaAAAAgDXYtWuXlixZYpa1WrZsyWcNAAAAAIDZeXh46IsvvtCQIUNkb29vGm/Xrp2GDBmiAQMGaMGCBfr555/1+uuvq0GDBhZMCwAAAOCvrKLBzs7uvzHOnj2roUOH5lk/e/ZszZ49O9e59PR0s2YDAAAAYN0OHTqkTz75xCxrDRs2jAY7AAAAAIDZbdq0SaVKlcp1zsbGRtOmTdOCBQskSVu2bKHBDgAAALAiVtFgl5GRUaj1AAAAAB5evr6+6tGjR65zR44c0fnz5yVJzs7O8vPzk7u7u5KTk3XixAnFxsZKksqUKaOAgAD5+/sXVWwAAAAAwCPkXs11WSpUqCAbGxtlZmYqMTGxiFIBAAAAyA+raLALCQmxdAQAAAAAxVRwcLCCg4NzjG/cuFHdunWTm5ubPvvsM/Xv318ODg6meaPRqPXr12vkyJE6d+6cOnTooOHDhxdldAAAAAAAJEnHjh1TZmamJKl69eoWTgMAAADgr6yiwa5Tp06WjgAAAADgIZKZmamhQ4fqzp07CgsLU8uWLXPUGAwGderUSeHh4fLz89OECRPUu3dveXt7WyAxAAAAAOBR9sEHH0iS3N3d1aVLFwunAQAAAPBXNpYOAAAAAADmtmfPHl28eFFt27bNtbnur3x8fDRo0CClpqZq5cqVRZQQAAAAAIC7fvjhBy1cuFDS3UY7FxeXAl1vMBjy/AMAAADwYGiwAwAAAPDQOXPmjCSpSpUq+ar39fXNdh0AAAAAAEUhNDRUL7/8siSpb9++Gj58uIUTAQAAAPhfVnFELAAAAACYk9FolCRdunQpX/UXL16UdPdoWQAAAAAAikJ4eLi6d++utLQ0de3aVT/88MPfWifrM/C9sIsdAABA4UpMkU5FWTpF0apVXnIubekURYcGOwAAAAAPnbp160qStmzZogMHDqhRo0b3rL127Zrmz58vSapXr16R5AMAAAAAPNrCw8PVpUsXJScn66mnntLSpUvl4OBg6VgAAAD4G05FSS/NtHSKojV3tNSkhqVTFB2OiAUAAADw0GncuLEaNWqkjIwMderUSUuWLFFGRkaOuvDwcLVt21bXrl2Th4eHevbsWai5EhMT77uzwN+Vnp6umzdv6s6dO3/r+qSkJN2+fTvf9YmJiYqLi8vz788///xbWQAAAADgYbZlyxZ17txZSUlJ6tKlC811AAAAgJVjBzsAAAAAD6UffvjB1DzXu3dveXp6yt/fX25ubkpOTtbx48dNR8ja29vr+++/l5ubm9lzbNq0SdOnT9fWrVuVmpoqe3t7NW/eXK+99pp69er1QGtHRUXp448/1urVq3X27FlT816lSpXUsWNHvf7666pZs2aO627evKmNGzdq48aN2rZtmy5fvqzExERJko+Pj4KCgjR27Fg1aNDgnvceOHCgVqxYkWc+W1tbpaenP8ATAgAAAMDDJTQ0VN26dVNKSoq6dOmi3377TSVKlLB0LADAA3J58RVl3rolSTofX0raYtk8AADzosEOAAAAwEOpfv362rVrl4YNG6awsDDFxcVp06ZNOerq1aunr776Sq1btzZ7hqlTp+qtt94yvXd0dFRycrK2bdumbdu2afTo0ZoxY8bfWnv79u3q0qWLEhISTGNOTk5KSkrSpUuXNHfuXM2fP18LFy5Ujx49sl373XffacyYMdnGSpcurVu3bik6Olo//vijfvnlF3366ad69dVX88xRsmRJOTo65jpnZ8dHTgAAAADIsnHjRvXo0UO3bt0yHQtLcx0APBxKtm5nen3rtGiwA4CHDEfEAgAAAHho1ahRQ6GhoTpy5Ig+/vhjDR06VP/4xz/0wgsvaMqUKdq6dauOHTtWKM1169atMzXX9ezZUxcvXlRSUpJiY2P18ssvS5JmzpypH374ocBrZ2Rk6LnnnlNCQoIcHR31zTffKDExUYmJiUpJSdGCBQvk5eWl27dva9CgQdma8P6qUaNG+u6773T16lUlJycrKSlJK1euVI0aNZSenq5//vOf2rJlS55Zhg0bds8jYmNiYgr8bAAAAADwMFq/fr26d++uW7duqWvXruxcBwAAABQjbCcAAAAA4KHn5+cnPz+/Ir3nxIkTJUkNGzbUokWLTLu5eXl56auvvtLly5cVEhKit99+W88991yBdns7cuSILl68KEmaNm2aXnrpJdNcqVKl1LdvX5UqVUpPP/20EhIStHXrVnXt2tVU4+3trQULFqhv377Z1i1durS6deumpk2bqk6dOoqPj9dnn32mwMDAv/ufAQAAAAAgqU+fPkpNTZUkRUVF6YknnrhnbY8ePfTmm28WVTQAAAAA90GDHQAAAACY2fHjx3X48GFJ0oQJE3JtnnvrrbcUEhKiqKgobdmyRUFBQflePyMjw/S6QYMGudY0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Sd9++63CwsKs7VqnTp00fPhwBQQEOLQdNGiQ/vrrL+v/99xzj8M27I477rBGVoD7xMTEaPz48Vq7dq3Onz+v0qVLq2PHjho2bJi2bt2qp59+WkWKFNG6deuy3XdycrJmzpypZcuW6ciRI7LZbKpUqZK6dOmiAQMGWPsTmdmyZYumTZumXbt2KSEhQVWrVtWgQYMy3be9ePGili1bplWrVuno0aM6deqUAgICVLduXfXp00ft27fP9rwg9zheLtjHy8i6yMhI6zgyJiZGRYoU0W233aaBAweqSZMmTtsMGzZMW7Zs0WOPPaYnn3xSc+fO1eLFixUREaG4uDhNnDjRanv48GEtWLBA27ZtU2RkpHXusXnz5hoyZIhCQkIyjS8pKUk//vijfv31V0VERCglJUUVK1ZU48aNNXjwYJUuXTrb83z58mXNmjVLK1as0NGjR5WamqoqVaqoZ8+e6tu3rzw8eOgOUJD89ttvOnz4sCTpww8/zHAeqUaNGnriiSf08ccfa9q0afrkk0/k4+NTYPpHzk2dOlVJSUkqXLiwRo0alaH8vvvuU5s2bbRu3Tp98803evDBB7PV/2+//eZyWdpsNo0aNUozZsyQJK1Zs8a6Hof8N3HiRElSw4YNNXjw4Azlb775pr7++mvFxsbqu+++c3otMKuioqL08ssvKyAgQOPGjdOHH36Y476QOykpKZoyZYokaciQIapfv75Duaenpz7++GM1btxYUVFRWrp0qXr37p2taXzwwQc6d+6cQkJCNHnyZJfH8f7+/jmaB+TM7t27FRYWJkl6++23FRgY6FAeHByskSNHasSIEVq6dKmioqKueRyR3unTp63XrtaZvn37Wt//06dPuzxOR97LTV7ItcybN09nz56VJI0ZMybDsV6zZs30wAMPaPr06Zo4caLeeOONfIslz7g3vw83m7Vr11qZ587+bDabWbRoUYZ2Z86cMZ06dXLZrlSpUiY8PDxDu2uNYJfTfo25MhLbnXfememzwMePH2+MMebcuXOZ1pNkPvjggzyLLbPPuUaNGuY///lPrkawu1p2RrC79957jSTTvHlzl3W+//57K95Dhw7lSYw3I/udOt27dzeDBw/OsKw/++wzY8yVkRHsowK6+t69++67mU6jR48e5sknn3TZx0svveQyzp9++skULlzYabsmTZpYd7y4uiN/5cqVJjg42OW0O3fubM6fP5/p5/PII484bevl5WW+//57c+nSJdO1a1endYKDg3N0l9X1mP6RI0dMkyZNXH42VapUcdrW3evErWLr1q3W5/Hdd99Zd9Pa/+rXr2/Vtd/d4+qvadOm5tSpU5lOY+bMmS5/+2vVquW0vTFXtme3336703b+/v7m22+/tf7vbAS72NhYc/fdd7uMvWTJkmbNmjWZxj5lypQMn4/979577zUpKSlm0qRJxtfX12md0aNHZ3v51K9f31qXXVm4cKE1jZ9//jlDeU5HsDt//rz12Vzt/vvvN5LMvHnzrPc2btxoPDw8TMeOHbM1HQC3tlKlSmW6bWnevLmJjo522jYqKsqEhoY6bVekSBGHbcPhw4cztE9NTTUvv/yy8fT0dNqHt7e3+eKLLzK027x5s1Vn1qxZJjAw0Gn7OnXqZIi9cePGmc7vnXfemSefK3Ju06ZNpmTJkk6XT4UKFcyYMWOMJBMYGJih7bVGe9u3b5+pVauWy+Vfr149p+tq+hHs/vvf/7rcR37ooYecjpxx6NAh4+Pjk+m6N2DAAKd33TOCXf7iePnKX0E9XkbWTJo0yfj5+blcvk8++aTT3yb7ccozzzxjWrdunaHd77//bowx5qOPPsr098vX19f88MMPLuPbsGGDqVixosv2Xl5eZt++fQ5trjWC3aZNm6zRvp39tWjRwuX+CwD3eOqpp4x05RxHWlqa0zobN260vscrVqwoUP0j5+znpTMbrejTTz81koyHh0eGEY9yKykpyVruOTk3iJxJSEgw3t7eRsp8NCH7Oc7Q0NBcTa9bt25Gkvn888+NMYYR7Nxo/fr11nfu119/dVmvcuXKRpIZOnRotvpPSkoyRYsWNZJcHoPBPeznazw9PV3+lp88edJaP7755pts9b9v3z6r7eLFi53WmTVrlrU9OX78eLbnAXknL0ew69u3r5Fk6tat67LOggULrPXj6pEzCyIS7JBnkpOTTfny5Y0kU7NmTTNu3DizatUqEx4ebmbNmmXefPNNU7VqVTNz5kyHdgkJCdaFfz8/P/Piiy+a5cuXmz/++MNMnz7d3HbbbUaSKVasmDly5IhD28wS7HLT78WLF03dunWtjcmQIUPMggULzB9//GEWLVpkPvroI9OgQQPz8ccfG2OuDIu7detW89prrxlJpnLlymbr1q0Of+kTH3IT29GjR60dkDJlypjPP//cbNiwwSxfvtwMGTLE2Gw2ayhNdyTYhYSEGCnz4dr37t1r/VAuXLgwT2K8GdkPJOzL86677jI//vij2bx5s9m6dat1wnHmzJmmQoUK5sUXXzTTpk0zK1euNOvWrTPTp093SIr58ccfrzmNbt26mZ9++sls3LjRzJo1yyG5a+3atRnab9u2zTrYqlatmvn222/Nxo0bzeLFi03v3r0d+nZ2wWDXrl1WQk3ZsmXN559/bsLDw82KFSvMY489Zmw2m5Fk7r777gwnWK6OvXfv3mbhwoUmPDzcfPPNN6ZMmTJGkilatKjp2rWr8fDwME888YRZvny5Wb9+vRkzZozx9/c3kkzbtm1zvXzyevoxMTHWSedixYqZt956y6xcudJs3LjRfPPNN6Zq1apGkqlYsaI5d+6cQ1t3rhO3kvQJZF5eXqZMmTJmzJgxZv369Wbr1q1mz549Vt0SJUqY++67z4wfP94sXrzYbNy40SxcuNC88MIL1mNK77nnnmtOo0KFCubTTz81YWFhZvny5ebZZ5+1Lhg+9NBDGdqnpqZaF1y8vb3NCy+8YFauXGnWr19vRo0aZYoUKWItayljgl1aWpqVbO7h4WEee+wxs2LFCrNhwwbz+eefW+t54cKFM1x4uzr2KlWqmPHjx5sNGzaYhQsXWidPpCsXpj08PEyTJk3M999/bzZu3GjmzJljrW85edxAzZo1jSQzaNAgl3VWrlxpxfD+++9nKLdfuGrbtq3p1auXadasmWnXrp0ZOnSomT17tssh7NPS0qyEwqsTH+vVq2ckmY0bNxpjrpxYqF+/vvH19b2pHtELIP8VK1bM9O7d20yYMMEsWbLE2rY8//zz1m+QswshKSkpVrKar6+vee2118zq1avN+vXrzfvvv28CAwMdtg3OkpbsF8Ekmb59+5qffvrJbNq0yWEfUJL56aefHNqlT7Dz8vIylSpVsvb/li1bZp566ilr/+/qE7V79+418+fPt9ovX77c4XjrwIEDefr5IntOnz5tJQEFBwebTz75xISHh5uVK1ea5557znh5eVnrVXYT7M6dO2clePj7+5u33nrLrF271qxdu9a88cYbVnJK1apVzYULFxza2hPs7MmgVatWNRMnTjQbN240S5YssU70STKvv/56hmnv3r3bBAUFmWHDhplJkyaZ5cuXm/DwcDNnzhwzaNAga55GjhyZrXlC7nG8XLCPl3Ft6S8i1K1b10yZMsXalnfv3t0qe/755zO0tR+n2Jfv4MGDzeLFi82ff/5ptm7dauLi4owxxowcOdKEhoaad955x8ycOdOsX7/erFy50kyYMME6L+nt7e30sXt//fWX9ftavHhx89Zbb5nffvvNbNiwwcyaNcu8/PLLpkyZMmbnzp0O7TJLsPv777+t9apixYrm//7v/8y6detMeHi4+eijj6xHB7Vu3dppYiEA97D/5jg7b2WXlJRk7W998sknBap/5Exqaqq1HcjsxtOwsDBrmxUWFpanMWzatMnlsSXyT/pzur/88ovLep988om1P3L58uUcTcs+GEezZs2sxwaSYOc+48ePt5b91dec0uvXr5+RrgwakB1//PGH1f+GDRtMZGSkGTlypLnzzjtN8+bNTe/evc2XX37J42Hd4OGHH85SHkCFChWMJPP0009nexr2G+MqV65s1q9f71D2+++/W8eoOekbeSsvE+zsN8s+/PDDLutERUVZvw0zZszI9TTzGwl2yDP2nd3MMovT0tLMpUuXHN575513rBM6GzZsyNAmPj7eGoXm6uSBzBLsctOv/Tnfnp6eZunSpS7n+eqN/BdffJGlDVBuYrNv5IKDg01ERESGth9++KH1I3S9E+xSUlKsaX/44Ycu6yUmJlr17HeVIyP7gYR9w+PqDr6LFy+6LDPGmFdeecVIMo0bN850Gi+88EKG8sTERGuH6dFHH81Qbk+8qVatmomJiclQ/vTTT1v9O7tgcNdddxlJpnTp0ubo0aMZyu3fKUlm7ty5LmN/+eWXM7RNfwFXujKC1tWmT5+e6cXjzOT39IcNG2akK6PIXH1HuDFXEvDso6L9+9//dihz5zpxK0l/sqFUqVLm2LFjLuvaL3A4s3PnTmtklK1bt7qcRtWqVZ3eyf/6668bScbHx8ckJiY6lM2cOTPTE1GbN282hQoVsupcneA1e/Zsq+yrr77K0P7IkSPWCEr33nuvy9id/UakpaWZhg0bWnU6duxokpKSHOrExcWZEiVKuLxwnZk2bdpYJ2hc+frrr63p/+tf/8pQbj/J6+qvZs2aGZaZXc+ePY0k8+CDD1qJeBMmTDCSTIkSJawTT6NGjTKS8wQ/AMhMZtuWbdu2Wb/vV1/4njJlSqYny//880+HbcPV+yjp72h2NYrA8OHDrYvn6S+Qp98/ql69utP9x5dfftlIV25Eunq7sHv3bqt9ZttdXH/PPfdcpvuu6Ucxz26CnX2d8Pb2dnrRbvXq1dYF1zfeeMOhzJ5gJ10Z/fnMmTMuY/fy8spwTHL58uVMLxb98MMPRpIJCAgw8fHxWZ4n5B7HywX7eBmZS0lJsRKHGzVq5PQion1barPZMtxslP445aOPPnI5ncz2FVJTU61EPmcXO+zHapUqVXK6/hlz5Wbrq38jM0uwa9GihbUP4Oyi7a5du6x9kMxG1gNwfdlHKnrssccyrWe/8T67F8bzu3/kTPqRiiZNmuSy3rFjx6x6eX1R3H5urWTJkiTcXEfpbwL4+++/XdabM2eOVe/gwYPZns6pU6dMiRIljJeXl8OIRSTYuY99EJkiRYpkWs8+Crezp7dkJv3TEubNm2eKFSvm9Jx7SEiI2bRpU25mBdnUvn17I125cS0zrVq1MtKVkdKzKzk52bz99tvW9aQSJUqY2rVrW09tKlOmjPnggw+sZFu4T14m2NlvGnzzzTdd1klLS7NuTvzvf/+b62nmNxLskGfsd6r4+/tn+S7DtLQ06xEyr776qst6v/zyi5GujLCQ/sSNqwS73PSbkpJibdSfeuqpLM2HXVYS7HITW2JionXX0JgxY5y2S01NNdWrV7dO4uaFrCbYpX9UrrNEjPTsd9mOGjUqT2K8GdkPJLy9vc3p06dz3M/Ro0etk7JXPzrGPo3AwMAMF4Ts7DvVTZo0cXg/MjLSumN+1qxZTtteunTJ+j5dfcEgMjLSWl/sj1t2xn5S9+oRWLISu/270KBBA6flSUlJ1uhfCxYscBmDM/k5/djYWOuksrPHq9nZLxhWrFgxW7Hn1zpxq0mfQObqNzmr7KNnjB071uU0pk6d6rTtgQMHrDrbtm1zKOvcubORZDp16uRy2k888YTV/uoEO3v7zJb1l19+aa1P6RPs08c+bdo0p21Hjx5t1dm8ebPTOvZHWnXr1s1lDM6MHDnSisvZAXlqaqrDo3MfeOCBDHXuuusu06dPH/PVV1+ZX375xfzxxx9m7ty55qGHHrK+u0FBQU4TCbZs2WIdFBQrVsw6ESz9L7l8//79xtfX19SvX99KItmyZYt56KGHTPPmzc3dd99txo0b53KkPADITMeOHZ3uS9gTPjL7XU2/bbg6qcJ+4bxJkyYuk1ZiY2OtEWpWrlxpvZ8+ocPVxZc9e/ZYdXbt2uVQRoJdwZSWlmaNXnd1glt69sSK7CbY2U++unrcoDH/21+4+uRf+gS777//3mnb+Ph464TuBx984HIarthvBli3bl2W5wm5x/HyFQX1eBmZW758ubV8r/7tsLt48aL1KPWrzx/aE+wqVaqUqwtQS5cudXpeNX0y/ZIlS7LVp6sEu/T7AKtWrXLZ3n4xp0uXLtmaLoD8Y9/XcZZsnl7t2rWNJPPII48UqP6RM/v377d+t52N9GsXGxtr1ZswYUKeTX/cuHFWv85uBED+SX+Thaske2OMWbZsmctz0lnRv39/I2UcSZwEO/exf/blypXLtJ59EBlfX99s9f/RRx8Z6cpAPUWLFjX+/v7m7bffNmvXrjXr1q0z//nPf6wnMgQHB5uoqKjczA6ywf6ki969e2da79577zWSTPv27XM0nd9//9107drV+u1I/9ezZ0+zZs2aHPWLvJVXCXbJycnW8s3sxjBjjHVu5LXXXsvVNK8HLwF5JDQ0VL6+vrp06ZIGDRqkd999V7Vq1cq0za5duxQdHS1J6t69u8t6LVu2lCQlJiZq165datSoUb71u23bNsXGxkqSHnzwwUynkxO5jS0hIUGS1KNHD6ftPDw81L17d40dOzaPI7+2lJQU67Wnp2emdb28vJSSkuLQBs41aNBAJUuWzLROWlqali5dql9++UV79+7V+fPnlZycbJVJkjFGx48fV9GiRTO0b9y4sfz8/Jz2XblyZUnSuXPnHN7fuHGjjDHy8PBQt27dnLYtXLiw7rrrLs2ZMydD2caNG63XvXv3djlvffr00bZt27Rhwwan5ZnFXqFCBR04cECtWrVyWu7t7a3SpUvrxIkTGeYvq/Jj+uHh4UpKSpKUtd+Jo0eP6vTp0ypVqpRV5o514lbWqVOna9bZt2+fZs+erb/++kunTp1SQkKCjDGSpCNHjkiSIiMjXbZv06aN0/crVqwom80mY0yGZWL/3rjaZkhSr169NH78eKdl9vb33Xefy/Z9+vTRU089JWOMNm7c6LRu69atnbatUKGCJMnHx0eNGzfOtE5217fHH39c//d//6eEhAQNHDhQs2bNsqZx/vx5vfzyy9q6datV39n2aMGCBSpcuLDDe82aNVOfPn304IMPqlu3bjp37pxeeuklLVy40KFe48aNtXjxYj355JM6dOiQYmNjFRgYqDfffFPPPPOMJGn48OFKSkrSxIkT5e3trcWLF6t3794OsaxYsUIrVqzQ/PnzZbPZsvUZALj57dmzR7Nnz9bWrVt1+vRph21LRESEpIzblj/++EOSXO6/SVf2P1xtG1atWmW1d/W7FBgYqNDQUG3evFmbN29Whw4dMtRxtV2z72dI7GvcKA4ePKgzZ85IuvZ65WqfPrO+T58+LenaxwxTpkxRVFSUjh07Zu0/2NlsNpf71X5+furUqZPmzJmj8PBwp3XCw8O1YMEC7dq1S2fPnlViYqJVFhcXJynz/TjkH46XC/bxMpyzL6+SJUu6PFby9/fXvffeq9mzZ7tcvh06dJCHh0em00pISNBPP/2k1atX6+DBg7pw4YJSU1MlSfHx8ZIy/n6tXr1akhQUFKQuXbpkeb4yY99/CAwMVNu2bV3Wa9mypT777DNt3rw5T6YLIPfs5yiycr4/ff2C0j9yJqvXeuzL5eo2ufHzzz/r2WeflSQNHTpUDz/8cJ70i6y5Hst+wYIF+vHHH1W9enW99dZb2Q8S+SK/f4/t17zS0tJ04cIF/f777+rYsaNV3rp1a7Vt21bt27fXmTNnNHr0aH3++efZmgZy5npsiz/++GO9+uqrMsaof//+6tOnjypUqKCjR49q1qxZWrBggRYvXqyxY8da109wY8tu3sjVbQoqEuyQZ4oVK6YPPvhAzz//vGbPnq3Zs2ercuXKuuOOO9SmTRv17NlTISEhDm3sF30kacSIEdaXx35RKP2/Hh4eSktLs06uZyY3/aY/qVS7du3sfARZkpvYjh8/LunKxYGqVau6nEb16tXzPO6s8Pf3t17bT9I5k5aWZl2QCAgIyPe4bnRlypTJtPzEiRPq3r27/vzzz2v2denSJafvFylSxGUbHx8fSRk3avb1sUyZMg7L/mqu1kd7+6JFizokhl2tRo0akqSzZ8/q8uXLVjxZib1QoUJZrpPTjXZ+TN/+O2Gz2dSnTx9Jzn8j7CfFJTkk2LlrnbiVXet7+sYbb2j06NEOy8wZV8tDcr1MvL295eHhodTUVIdlkpCQYCWMZ7ZdcFWWvr39e+hMmTJlFBAQoIsXL1rf66zGbl//AwICXCZp5PQ7Wr58eU2ePFkPPvigDhw4oCZNmqh8+fIKCAjQoUOHlJSUpHbt2unEiRPat2+fgoKCMvRxdXJdenfddZeefPJJffrpp1q6dKnOnz+vwMBAhzr33HOPDh48qIiICF2+fFlVq1aVt7e3JGny5MlauXKlnn76aTVv3lyJiYl69NFHlZKSoocffliPPPKIDhw4YCXvzZgxI18S/wHcuF5//XV99NFHVnKIK+m3LRcuXNDFixclKdPjiWrVqjl9PyEhwTo2mTp1qhYvXizJ+X7K4cOHJcnl8ZurbUP6fT32NW4M6bf/OVmvstp3Zvsj6cuOHz+eIcGuZMmSThOn7Oz7Q1fvyyQmJmrQoEGaN2/eNWPNbD8O+Yfj5YJ9vAzn7Ms3s9+19OWujrOutf5v3bpVvXr10tGjRzOtl5qa6rD+2M+N1qpVK89u8rGf50hOTlazZs0kOd9/sH8PY2JilJaWds0EQgD5z9/fX+fPn8/0fL/0v+sB2T3fn9/9I2eyeq0nfVleLJsVK1aob9++SklJUb9+/fTNN9/kuk9kT34v+9jYWD355JOSpK+//lq+vr45iBL5wb7s8/P33q5z584OyXV27dq1U/fu3bV48WItXLiQBLvrJL+X/datW/Xaa6/JGKOPP/5YL730klXWokULDRgwQGPGjNHLL7+sF154QZ06dVJoaGg25wIFja+vrzw9PZWamnpT7eeRYIc89dxzz6l+/fr6/PPP9fvvvysiIkIRERGaNWuWnnnmGQ0dOlSfffaZdfesfTQ2Sdq+fXuWppH+TnFXctOvPYNe+t+JxLyUm9jsd1h7eHhkmulrv3h/vfn5+VlJFidPnnRZL33Zte40x7WzugcMGKA///xT/v7+Gjp0qFq3bq2QkBD5+fnJw8ND8fHx1h3p9pOWecG+Pl5rfXNVntX26b+HSUlJGS4Y3IzsvxPGmCxdCJIcfxvdtU7cyjL7nk6bNk3//e9/JUl33nmn+vbtqxo1aqho0aLW+v/qq69qxYoV+fIdlTL/nl3rO3qt9tL/vqfpt6EFwYABA1S5cmW99dZbWrVqlXWhqFixYnruuef09ttvq3Tp0pKkSpUqZbv/zp0769NPP1VKSor279+vJk2aOK2XfjQmSYqOjtZLL72k8uXLW+vGb7/9pujoaDVq1EhTpkyRJLVv317GGP3rX/8iwQ6Ag++++06jR4+WdCXht0+fPqpevbrDtuWll17S77//7rBtSZ8gkZNtQ/pjmcOHD1tJdJnJyvEbbmy5Xa8yk9X9kauPGbI7bXt5+ulJ0iuvvKJ58+bJw8NDDzzwgO6++25VqlRJAQEB1v5f9+7dFRkZyX61m3C8fOsdL98Msrt8XR1nZbb+x8fHq0ePHoqMjFTp0qU1bNgwNWnSRKVLl5avr69sNpt27NhhjQqUfv23Ty8vz4va9yHi4+OzdJ4jLS1NSUlJXHQHCoBSpUrp+PHjmZ7vl/53zj+75/vzu3/kTHBwsPXEjMyWzYkTJ6zXuV02K1asUM+ePZWYmKi+ffvqhx9+uOa+HvJe+ps7Tp486fKGkJwu+5dfflknTpzQI4884jTBCu5jX/bnzp1TUlKSy31B+7LPye+9XdOmTV3Wa9KkiRYvXqyjR49mGgfyjn3ZXGtbnNNlP2nSJKWlpalUqVJ64YUXnNZ5/vnnNWbMGJ06dUqTJ0/Wxx9/nK1poGAqWbKkTp48mem6FRcXZyXY3Qj7eSTYIc/deeeduvPOO5WSkqLt27crLCxMCxcu1MqVKzVx4kQZYzRx4kRJUokSJax24eHhLh9bkV6VKlWuWSc3/QYHB1vvHT9+PEd32edXbPbRdVJTUxUTE+PQV3rX2gDmp9DQUG3atEm7d+92WSd9Wd26da9HWDet3bt3a926dZKkJUuWqH379hnqHDt2LF+mbV8fo6OjM72r2NX6WLx4cUlX7lhydqf91e29vb0zvbP+ZmL/bhcpUkRr167NUhv7I7nduU7Aua+//lqSNGjQIH3//ffXbbpFihSxHsd96tQpl/VcfUez2j4xMVHnz5+X9L/vdUHSvHlzLV++XPHx8YqMjJSXl5cqVKggb29vbdu2zRrJqUWLFtnuO/0Id5cvX85yu2effVZnz57V5MmTrd+1PXv2SLqSKJPevffe61AOANL/ti0PP/ywlZR7NWeJIkWLFs3SyOCufveLFStm3Xk4duxYp/sZV7sRTowgd9KPAnv69GkVK1bMab3M9idcSb9vcerUKZUvX95pvfT7M872R2JiYpSamuryAp09tvRtk5KSNHnyZEnSJ598oueee85p22vdhQv34XgZBZV9+V7rd9G+fHNynPXzzz8rMjJSfn5+2rRpkypWrJihTvoL4+nZz426GjkvJ+znOW6//XZ99913WWpDwihQMISGhmrbtm2Znu8/cuSItU+U3fP9+d0/csbPz09VqlTRoUOHrsu1nmXLlum+++5TYmKi+vTpo5kzZzo8ghTXT/pRo3bv3u3ycfb2ZV+mTJls7avYHwO/fv16NWzYMEN5VFSUpCv7IfbyZ599VkOGDMnyNJAz9mWflpamvXv3qn79+k7r2Zd9dr/z9erVs15nduySvowEu+sjNDRUCxYs0L59+1yeO7l8+bJ1o212l/3+/fslSXXq1HF5bOzp6amaNWvq1KlT2rdvXzbnAAVVaGioTp48mem+xD///GO9vhH289g7Qb7x8vJS48aN1bhxYz3zzDMaOXKk3n//fc2YMUNff/21PDw81KhRI+sCT0JCQo4ubjuTm37Tt12xYoWeeOKJLLfNymMLchPbbbfdZr1ev369evbs6bTe+vXrs9xnXmvTpo02bdqktWvXujwJvHz5cklSYGCgyx00ZM3evXslXfksXV3gDA8Pz5dp29fH+Ph4bd26VY0bN3Zaz9X6aG+fmpqqsLAwl3crrVmzxqH+rcA+ClZcXJwKFy6smjVrZrmtO9cJOGdfJr169XJanpSUpC1btuT5dG02m+rVq6dt27Zp/fr1GjRokNN6rr6jNptN9evX19atW7VmzRoNHz7cZXv7o28bNGiQN8HnA2ffpalTp0qSypYtqzZt2mS7zx07dlivQ0JCstRm+fLlmjlzpvr06aMePXpY79tHibj6sbT2/xe00QEBuNe1ti2XL192OjqMl5eX6tSpo7///lvh4eHq37+/0/au9hU8PDx0++23a8uWLTp9+rTTk+H5icfEFUy1a9e2kvLDw8Nd7rvmZB+0du3aKlSokJKSkrRmzRqXxxz2YwY/Pz+n009MTNSWLVvUvHlzp+3t+0PpjzmOHz9uJeK7+q7t3btXZ8+ezfL84PrieBkFlX157du3TydPnnT5qFf78s3JcZZ9/W/UqJHT5DrJ9fpvPydx4MABHTp0KNPHf2eVvc+DBw+qdu3ajEwH3EDatGmjH374Qbt371ZUVJTT8x/28/02m80aGbag9I+ca9OmjQ4dOmSNjO7sseH2ZVOxYsUcPR1CupIU3qdPH12+fFl9+vTRrFmzSK5zo5IlS6pWrVrau3evfv31Vz322GNO661YsUKScnROVbqyn5GZ5ORk6ylgOblZC9nXokUL66bKX3/91en127Nnz1rXMrK77OvWravg4GCdOXMm0wQqezJW0aJFb4jHRd4M7MsyLi5Of/zxh1q2bJmhzpo1a6ynVGR32dtH7r7WdzkmJkYSN9rcTNq0aaOVK1dqw4YNunjxotPvtH1fws/Pz+VTogoSzk7jumnWrJmkK48EsD8KISgoSF26dJEkvf322xkex5JTuek3MDBQnTt3liSNHj1a586dy3Jb+zPKL1y4kC+xlS5d2rogMGbMGKejUvz555/67bffstxnXhswYICkK3dZT5o0KUO5fcQeSerduzd3HuSSfSfjwoULTi/sJCUlWY8Py2u33367dZLW1VC9S5Ysccg8T69hw4bWQffo0aOdrs979uzR/PnzJcllQunNqG7dutYF65EjR2brUUXuXCfgnH2ZuHqM3pdffplvF2bt35sZM2Y4HX0gPj5eX3zxxTXbz5s3z+lBrzHGWp+qVKlyQ13Y27p1q7766itJV4Yfz+6Ju/Pnz+uTTz6RJFWvXj3DY2CdiY+P1+OPP67AwMAMn3u5cuUkSX///bfD+7t27XIoBwDp2tuWzz77zBpd9Grdu3eXdOUR5mfOnMlQnpCQoHHjxrmctv1x1ePHj7/uo+Laj7ekzI+5cH35+fmpU6dOkq6se/bE+/QiIiI0d+7cbPft6+uru+++W5L0xRdf6NKlSxnqXLhwQV9++aWkK49vd/XIRVfHLL/88ou1/U1/zJH+hK6r79q7776bhbmAu3C8jILq3nvvlY+Pj9LS0vThhx86rbNo0SKnv01ZZV//jxw5orS0tAzlp06d0oQJE1zGZx+N9LXXXsv2tJ3p0qWLgoKCdOHCBc5JADeY++67T4UKFZIxxjoPkl5ycrI+//xzSVLr1q1djjjsrv6Rc/ZrPUeOHNGcOXMylEdGRmr27NkOdbNr4cKF6t27ty5fvqx+/fqRXFdA2JfnggULnCbCLV261NqPvf/++7PV9+zZs7V161aXf/Zply1b1npv6NChuZwjZEVwcLDuvPNOSdJXX32lhISEDHU+/fRTpaSkyMvLS3379s1W/56entby/fHHHxUdHZ2hTkxMjGbOnClJVizIfx07drRGsR4zZozTOvZtdKVKlbI9YJL9eue+fftcDjbx119/WaOcXe8bepF/+vfvL5vNpoSEBOt6XHpxcXH65ptvJEndunVzOPdbUJFghzyzbNkyvfzyy1q3bl2GUVb27Nmj9957T9KVuzTTn6gePXq0/P39tX79et111136448/Mpw4PHbsmCZMmKCXXnopy/Hkpt///ve/8vX11dGjR9W2bVutXbvWoW10dLQ+++wz/fTTTw7tatSoIenKIxYyG0UuN7H9+9//lnTlLueHHnrI4STxunXr1LNnT6cnzq6Xpk2bWhftXnnlFc2dO9eat6ioKN133306c+aMfHx89MYbb7gtzptF06ZN5e3tLWOMHn74YYdHfR04cEBdu3bVtm3b8mXaHh4eevXVVyVdOSh64YUXrBEejDFasmSJBg8enGl7+zrw66+/aujQoQ7r84YNG9SlSxclJSWpVKlSevLJJ/NlPgqqTz/9VF5eXpo7d6569+6tnTt3OpQbY3TgwAGNHTtWo0aNst535zoB5+x31o4ePdoagUC6ksDw0Ucf6ZVXXsm3aT/xxBMqVqyY4uLi1LlzZytZS7qynenRo4fLC8aS9OSTTyo4OFiXL19W586dtXHjRqvs7NmzeuSRR/T7779Lkt56660CN7LQ6NGjNXfuXIfHtyYmJmrq1Knq1KmTkpKS1Lx5cz3//PMZ2n733XcaPXq09u/f77CNNsZo9erV1l28kvTOO+9kKZ633npLERER+vDDD1W2bFmHsg4dOsjDw0Pz58/XqlWrJF25GDxy5EhJshIXAED637blv//9r/X4Q+lKIu8HH3xgHTM4M2LECAUEBOjcuXPq3LmzNcKNdOUCSa9evazfN2eGDx+uevXqKTY2Vq1bt9bcuXMzHP+dP39eP/30kwYPHmztH+aFsmXLWnc6OrvAA/exHxds27ZNAwYMcNgH3b59u7p27er05HxWvPHGG/L09FRERIS6du3qsO9y8OBBdenSRVFRUfL29ra2m8789NNPeu655xzWyaVLl1qj/LZs2dJhlLCyZcuqSpUqkq48kij9d+XMmTN67LHHrBP/KJg4XkZBVapUKT3++OOSrhz7v/fee9ZoEMYYLViwwFo/GjVqlKMEO/u+QmRkpJ577jmHx1n/8ccf6tixo9NEe+nKY7nsCcRz5sxR//79Mxw37t69Wy+//HKm+wxX9/nBBx9IupKc/Mwzz+jo0aMOdVJTU7Vt2za9++67mjJlSpb6BZD/SpcurREjRki6cjPFl19+qZSUFEnSuXPn9NBDD1kJwfZrQFfr0KGDGjZs6DRpNy/6R/7o3LmzNYrR448/bo0wI13ZD+/Zs6fi4+NVtGhRvfzyyxnaHzt2TA0bNlTDhg01Y8aMDOULFixQv379lJSUpP79++uHH34gua6AeO6551S8eHElJyerZ8+eDo/2W7NmjfW41kaNGum+++7L0H7p0qXWsr/6ukatWrWsMmd/9iSfQoUKWe+VKlUqH+cW6b377ruy2Ww6ePCg7r//fmt/MS0tTd999521P/fYY485HSV59OjR1nJzZuTIkQoICND58+fVt29fRUREWGVHjx5Vv379FBsb63A8hfxXqFAhvfnmm5Kk+fPna+TIkdbxSXx8vF544QVr1Mr//Oc/Tkc0HTRokBo2bKiHHnooQ9mQIUPk7e2ttLQ09evXz2F7YozRwoUL1aNHDxlj5Ovr67QPFEypqanWd97ZzYWhoaFWIvZbb72l77//3rredurUKfXr109RUVHy9PTM8nU2tzNAHpk8ebKRZCQZm81mSpUqZUJDQ03ZsmWt9wsXLmzWrVuXoe3y5ctNsWLFrHqBgYEmNDTUVK5c2fj4+FjvN27c2Ok0K1Wq5DSmnPZrjDE//fST8fPzs+oUL17c1KlTx5QoUcJ67+OPP3Zok5KSYqpUqWJ9BpUqVTINGjQwDRo0MN99912exfbMM89Y5d7e3qZWrVqmXLly1nS7dOliJJlq1apda7E59e9//9uKu0GDBqZ48eJGkvHx8XF4v3v37k7bnz592tSoUcOKsVSpUqZGjRrGy8vLSDIeHh5mxowZOYrtVjJixAgjyfTs2TPTeq+//rrD+lC9enVTqVIl67377rvPer158+ZsTyOz71laWprp27ev1b+vr6+pU6eOKVWqlJFkChUqZO666y4jybRr185p/8OHD8+wPoeEhFjvFSlSxKxZsyZHn88999xjJJlXX33VZR37ZzVx4kSXdZy5HtOfMWOG8fX1tT6LEiVKmLp165oKFSoYb29v6/2rY3DnOnEr2bp1q/U5RkdHu6y3c+dOU7hwYatu6dKlTe3ata1lW7ZsWdO8eXMjyQwfPjxH0/D09DSSzK+//pqhbOHChdbvryRTuXJlU6NGDePh4WEkmW7dulll+/fvz9B+5cqVJiAgwKoTEhJiateu7bAOjhgxIkefz5w5c6x125X33nvPSDLNmzd3WceVrl27WtudChUqmJo1azp8p5o3b27Onj3rtO2rr77q8D2qUKGCCQ0NNYGBgQ77O2+//XaWYvnzzz+Np6enadWqlUlLS3Na51//+pfDcrLvh5QsWdKcPHky2/MP4Oa1bds2h2OVq7ctISEhpmnTpi5/o2fNmmVtBySZKlWqOGwbunfvbpUdO3YsQ/sjR46YevXqWXV8fHxMzZo1Tc2aNU2RIkWs9yWZc+fOWe02b97s9P2r2eusWrUqQ9mQIUMcjjPq169vGjRokGEbiutv5MiR1rLx8vIyNWvWNBUrVrTes+9zONvuL1682Egy/v7+Tvv+8ssvjc1ms7a/1apVM9WqVbPe8/DwcLo/PXHiROs70rhxY6fHLPb9sQMHDmRoP3v2bIftfsWKFU2NGjWsfa9mzZpZx+JXT/9a84Tc4Xi5YB8v49oSEhLMnXfeaS3LwoULmzp16pjg4GDrvYoVKzr9bWrXrp2RZEaOHJnpNNJvz319fU2tWrVMmTJlnK7/CQkJGdo/++yzDtv0ChUqmFq1ajkc3+7cudOhzYABA5we29q988471m+3JFOuXDlTt25dU7ZsWeu3NSvzBuD6SkhIMK1bt7a+o0FBQaZWrVoO1zHee+89l+3t11QGDBiQL/0j/xw5csRh36Ns2bKmevXq1rGjt7e3Wbp0qdO2+/fvt9qNHTs2Q3n6c3T16tVzuPZ09d+oUaPyeU5xtd9//936DtpsNlO1alVTvnx5h/MQzvZTjDFm+vTpVr0NGzZka7r2/dhb/dqDO40ZM8ZafoUKFTI1a9Z0uDberFkzc/HiRadt7ctPcp2C8vPPP1vXFuzH99WrV3c4vv/iiy/ya/bgQmpqqunfv7+1/AICAkzt2rWNv7+/9d6wYcNctrefc3F1HWfixIkO5yIDAgJMzZo1HY4tvLy8yB9wg1GjRjlsc+3HpN7e3g7v33nnnRnaJicnW8vv2Wefddr/uXPnHM4jBwcHm5o1azr8DnzzzTf5PJd5hwQ75JmTJ0+aMWPGmHbt2jkc+NhPKPfv39/8888/LtsfO3bMPP300w4JefYvVeXKlc2IESMynDTMSpJHTvq12717txk4cKApWrSoQ9syZcqY559/3kRERGRos3PnTtOmTRuHk0WSzAcffJCnsY0fP97hYoV0JRlv2bJl1ueS0wS7QYMGOfTr6i+z/mNjY83TTz9tgoKCHOarbdu2TpMskVFWLxikpaWZjz/+2OECkSRTvXp18/nnn5vY2Nh8u2BgzJWN5/vvv29Kly5tTcfDw8O0adPGbNy40UqOcXXBwBhjvv/+e9OgQYMMvxsDBgww+/bty/Hnc6Mn2BljzJ49e8yQIUMcDmDsn3GtWrXMiy++aLZs2eLQxt3rxK0iq8lvxhizadMm06JFC4flUaRIEfPQQw+ZY8eOubwIkRcJdsYYs379enPHHXc4TL98+fLm448/NpGRkdZ7zhLsjLmyHvbv39/hYMdms5nbb7/dzJw5M8efT34n2IWFhZkHHnjA4SBQkrntttvMF198YVJSUly23b9/v3nrrbfM7bff7nChR5IpWrSo6d27t1m/fn2W4khJSTGNGjUyhQoVynRfKCkpyTzzzDMO+1FNmzbNcNEKAIwxZuPGjVaCdvrfp0ceecRERUWZPn36GMl5gp0xV06WN2nSxKF95cqVzWeffWZ27dplvXf+/Hmn7S9dumQ+/vhjExoamuE4oXjx4qZ///5m+vTpDr+1eZFgFxsbawYPHuxwMUaS05M8uP4mT55sqlat6rBsGjRoYObPn29mzpxppCsJnVfLSjLaqlWrTIcOHRy2y15eXqZTp04ut8n2BLtKlSqZ2NhY8+STTzoky3t7e5s+ffqYI0eOuJzunDlzHG4gs58TePXVV01cXJypVq2a0/15EuzyF8fLBft4GVmTlJRkPv744wy/myVKlDAjRowwp0+fdtouqwl2CQkJ5vnnn3e4Ycpms5lGjRqZuXPnmg0bNljvO0uwM8aYJUuWmLZt2zrctGWz2Uy9evXMJ598YuLj4x3qXyvBzpgrx2k9e/bMcJzm5eVlGjdubN555x2XF+wBuE9iYqJ55513MlzLaNiwoZk3b16mba+VYJfb/pG/Tp8+bR599FGHm6k8PT3N3Xffbf766y+X7a6VYHf1+bbM/rihyj127txpunTp4rAf4O/vbx566CFz/Phxl+1IsLvxLVmyxLpx0/5XqlQp8/rrr7vcbzQmawl2xhizY8cO061bN4eb+L28vEyHDh2cngvC9ZGammo+//zzDMcnNWvWvGYC1LUS7IwxJjw83HTr1s3hpmHpys1GvXr1ynCtE9dH+hv7MvsrXbp0hrZZSbAzxpi4uDjz4osvOlzrttlspmXLlua3337Lx7nLezZjrnouJZAHjDE6ffq0zpw5o4CAAFWoUCFbj407efKkzp49qyJFiqhMmTLy9vZ2Wu/s2bM6evSoChUqpNDQ0Dzr92opKSmKiopSQkKCQkJCVKRIkWu2uXjxoiIjI3X58mUZY1SuXLlMhzHOSWzGGEVGRurixYsqU6aMgoKCJP3vc/Hx8VGdOnWyNI/pHT161OHRI65kpf+UlBQdP35c8fHxKleunIoWLZrteG5VUVFRio6OVmBgoPVooswYYxQREaHExEQFBwerZMmSkq4M3bxjxw5JV4bf9vPzy9Y0svo9S01N1bFjx3T58mWVLVvWWtanTp3SiRMnFBAQoOrVq2c6D+fOndPJkyfl6+urkJAQFSpUyGXdrMR+6NAhXbhwQaVLl87wOEa7f/75R0lJSapYsaKKFy+eaXzunL4xRsePH9f58+dVrFgxlS5dWp6enpnG6O514maXkJBgPSqsfv3611wekhQTE6OTJ08qICBAISEh1mMPjhw5onPnzik4OFjly5fP9jS2b98uY4yqV69uPTrP1fRPnTqlYsWKqVy5cpKu/E7bHx0bGhqa6fcuKSnJ2ral3+44k5XYz58/r8OHD8vLy0v16tVz2o/9N8Tf3996FHt2GWN04sQJxcXFqUyZMgoMDMxW+9TUVJ08eVKxsbEKDAxUSEiI02HQXbF/Fln5HZSuDLt+5MgRBQYGWssJAFyxb1uKFCmicuXKWduWiIgIxcbGqmTJkgoJCXHZ/syZMzp9+rSCgoKs/ZXZs2fr/vvvV3BwsKKjo68ZQ2xsrE6dOiVPT0+VLVtW/v7+Tutldbtmf2RjZtu15ORkHT16VJcuXVJaWpqKFCmiatWqXTNWXB9RUVG6cOGCSpUqpRIlSkiSXn/9dY0ePVqdOnXSr7/+6lD/woULOnTokDw9PVW/fv1M+7548aKOHz8um82mkJAQFS5c2GVdZ/utKSkpioyMVGJiosqXL5/pvtPV8xQbG2t9V+z7Art379bly5cz7M9nZ56QfRwvF+zjZWTfqVOnFBMTo4CAAJUvXz7T86gHDhywzgWWKVPmmn0nJycrIiJCaWlpDsdD8fHx2rdvnySpQYMGmR7jxMfHW7+95cuXl4+Pj9N6ro5tnbGfM7x48aL1vczOcRYA90hLS9OJEyesbUhWtg+7du1SSkqKgoKCVKlSpTzvH9dHcnKyoqKilJSUpHLlyl1zPzopKUn//POPJCkkJMTa/7Kzn8/MiqxsV5B/4uPjrUf4lS9fPtP9UOnKfuuRI0ckSTVr1sz0mO1q9v3YW/3aQ0ERGxtrXc8oV67cNa/125efJJePiU0vISFBUVFRSktLU0hIiMvzSbj+7McnwcHBWXpM8759+xQfH5+l6zj28zIXL1608kiycn0N+SMyMtJ6HHRmvL29Vbdu3Qzv28/jXusctHTl3Mjx48d16dIllS1bNtvX6goCEuwAAAAAAICkKxe02rZtq7CwMPXp00dz5851d0i4CcTGxio0NFQnTpzQe++9pzfeeMPdIQEAAAAAAABAlmV9SDEAAAAAAHDDW7hwoSZOnKiYmBiH9w8ePKi+ffsqLCxMkvTUU0+5IzzcoL766ivNmzdP8fHxDu9v2rRJd911l06cOKHChQtr6NChbooQAAAAAAAAAHLGy90BAAAAAACA6+fw4cN6/vnn9a9//UvFixdXmTJlFBsbq+PHj1t1Xn31VbVv3959QeKGs2nTJk2dOlU2m816NKz9kSKS5OnpqW+++YbHnwMAAAAAAAC44ZBgBwAAAADALaRXr17as2ePFixYoFOnTuns2bOSriRANWnSRC+99JL69u3r5ihxo3nqqacUHx+vZcuW6dSpUzp16pQkydfXVx06dNAbb7yhli1bujlKAAAAAAAAAMg+mzHGuDsIAAAAAABw/V28eFHHjx+XzWZTuXLl5O/v7+6QcBM4d+6cTp48KV9fX5UrV04+Pj7uDgkAAAAAAAAAcowEOwAAAAAAAAAAAAAAAAAAnPBwdwAAAAAAAAAAAAAAAAAAABREJNgBAAAAAAAAAAAAAAAAAOAECXYAAAAAAAAAAAAAAAAAADhBgh0AAAAAAAAAAAAAAAAAAE6QYAcAAAAAAAAAAAAAAAAAgBMk2AEAAAAAAAAAAAAAAAAA4AQJdgAAAAAAAAAAAAAAAAAAOEGCHQAAAAAAAAAAAAAAAAAATpBgBwAAAAAAAAAAAAAAAACAEyTYAQAAAAAAAAAAAACQx9atW6eGDRuqZcuW7g7lusrtfNvbt2nTJo8jK1jcuX7cKp9xQRIeHn5L/h4ANwsvdwcAAAAAAAAAAAAAAMDN5vz589q+fbv8/f3dHcp1ldv5trcPDAzM48gKFneuH7fKZ1yQXLhw4Zb8PQBuFoxgBwAAAAAAAAAAAAAAAACAE4xgBwAAAAAAAAAAAABAHmvbtq22bt0qT09Pd4dyXd2q8w0AuHmRYAcAAAAAAAAAAAAAQB4rWrSoGjZs6O4wrrtbdb4BADcvEuwAAAAAAAAAAAAAALiG8+fPa+rUqVqzZo2ioqLk7e2t8uXLq1atWnrggQdUq1Yth/rr1q3T008/rcKFCys8PNxpnzExMRo/frzWrl2r8+fPq3Tp0urYsaOGDRumrVu36umnn1aRIkW0bt06h3bz58/Xu+++q3Llymnp0qXaunWrpkyZop07dyoxMVGVK1fWQw89pHvvvddqs3HjRk2bNk179uxRYmKi6tatq6efflq33Xaby3lOTk7WzJkztWzZMh05ckQ2m02VKlVSly5dNGDAAHl5ZUw5yMp8nzt3TuPHj9eaNWsUGxtrzfejjz7qMpbsyO6ySm/jxo366aeftH37dp0/f16lSpVSzZo1NWDAADVr1ixD/YsXL2rZsmVatWqVjh49qlOnTikgIEB169ZVnz591L59+1zNS1hYmObMmaNdu3bpwoULCg4OVtu2bTVs2DAFBwe7bJdfn/HV696OHTs0ffp0bd++XTExMerRo4fefvtthzimTZumNWvW6MSJE/L29lbt2rX1wAMPqEOHDg59X7x4UW3btlVaWppmzpypOnXqOJT/9NNPeu+99yRJL774ogYPHuxQfuTIEfXs2VM2m00rV65UUFCQVbZt2zYtWrRIe/bs0bFjx5SamqqQkBC1bdtWDz/8sIoWLep0ftu0aaO4uDiNGTNGrVq10rRp0/Trr7/q+PHjio+P17p161SkSBFJV9a78ePHa9WqVdZn3qFDhzxbrwG4kQEAAAAAAAAAAAAAAC5t3brVlClTxkhy+TdlyhSHNosXLzaSjL+/v9M+N23aZEqWLOm0rwoVKpgxY8YYSSYwMDBD24kTJxpJplKlSuajjz4yHh4eTvt57bXXjDHGPP/8807LCxUqZJYuXeo0vn379platWq5nN969eqZw4cPZ2h3rfnO7LMsX758pvOdFTlZVsYYc/78eXPfffdl2u69995zaHPo0CHj4+OTaZsBAwaY5OTkbH9OcXFxpnfv3i77DQwMNMuWLcv2Z5Dbzzj9ujd69Gjj6enp0P+QIUOsurNnzzaBgYEu52HQoEHm8uXLDv3Xr1/fSDKfffZZhmkPHjzYanvfffdlKJ80aZKRZGrUqOHw/sCBAzNdRqVLlzYbNmxwOr/2+D/88ENTpUqVDG1jYmKMMcZs377dlC1b1mn/5cqVM5988kmmyxtAwcYIdgAAAAAAAAAAAAAAZGLw4ME6efKkKlasqOeff1633XabChcurMjISO3Zs0c//PCD4uListxfdHS0unTpojNnzig4OFivv/66WrRoocTERC1atEhffvmlXnvttWv2ExkZqVdeeUV33HGHnn32WVWpUkURERH673//q507d+rDDz9URESEZs2apZ49e2rIkCEqXbq0/vnnH7311luKiorSkCFDFBERIV9fX6vf2NhYderUSUePHpW/v79efPFFderUSZK0YsUKffLJJ9q1a5fuvPNObdu2zRrB61rOnj2rzp076+TJkwoKCtLrr7+uVq1aKTExUfPnz9f48eOzNN+ZycmySk5OVufOnRUeHi6bzaYBAwaob9++qlChgqKjo7V//37NmjVLly5dcmh3+fJlFS5cWIMHD1aLFi1Uvnx5FSlSRFFRUVqwYIFmz56t2bNnq3r16ho1alSW5yE1NVXdu3fX6tWr5e3trX/961/q2rWrgoODdfjwYY0dO1YbN25Ur169tGXLFtWtW/e6fsbSlXXvtddeU/369fX8888rNDRUPj4+KlGihCRpwYIFuv/++2WMUaNGjTRixAjVrVtXCQkJWrp0qT777DPNmDFDRYsW1VdffWX126FDB+3cuVMrV67UM8884zDNVatWSZK8vLy0Zs0apaWlycPDwypfuXKlJKljx44O7eLi4tSqVSv16tVLVapUUUhIiM6fP68dO3Zo3LhxOnLkiHr16qU9e/aoWLFiTud35MiR8vT01IsvvqjOnTtb8xkYGKjY2Fh17txZJ06cULFixfTaa6+pTZs2SkxM1MKFCzVu3Di9+uqrufvAAbiXuzP8AAAAAAAAAAAAAAAoqA4dOmSNRPX333+7rHfx4kWH/2c2Qtlzzz1nJJkiRYqYffv2ZSj//vvvHUYqu5p9FDFJpnPnzhlGSDt79qwpUqSIVee5557L0MeOHTuMzWYzksz8+fMdyl5++WUjyXh7e5uwsLAMbVevXm2NXPbGG29keb5ffPFFq8zZZ2kfgczVfF9LTpfVxx9/bLWbMWNGlttdvnw5wwhs6f3www9GkgkICDDx8fEOZZl9Tl9++aWRZGw2m9MRBpOSkkzr1q2NJNOlSxeHsvz+jNOve82bN88wX8Zc+ZyCg4ONJHPvvfc6HcFvzpw5TpfVggULjCRTrFgxk5qaar2/b98+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" alt="scaled_suite_summary" style="max-width:100%;height:auto;" />
````

````raw
task,modality,agent_candidate,agent_score,global_candidate,global_score,random_mean,random_sd,oracle_candidate,oracle_score
heldout-cancer-half-features,tabular,sgd-log_loss-a0.01,0.935672514619883,sgd-log_loss-a0.01,0.935672514619883,0.9251345029239765,0.01016822298076059,sgd-log_loss-a4.64e-06,0.935672514619883
heldout-diabetes-above-median,tabular,sgd-log_loss-a0.01,0.7819548872180451,sgd-log_loss-a0.01,0.7819548872180451,0.6971804511278197,0.08559587375127735,linearsvc-c12.743-b1,0.7894736842105263
heldout-digits-noisy,image,sgd-log_loss-a1.67e-05,0.9095238095238095,sgd-log_loss-a0.01,0.9261904761904762,0.9100988095238095,0.0139500936531127,knn-k15-distance,0.9333333333333333
heldout-digits-noisy-parity,image,knn-k1-uniform,0.9404761904761905,sgd-log_loss-a0.01,0.8761904761904762,0.8832916666666666,0.04192026149663852,knn-k15-distance,0.9476190476190476
heldout-digits-shifted,image,sgd-log_loss-a1.67e-05,0.9642857142857143,sgd-log_loss-a0.01,0.9428571428571428,0.9442619047619049,0.014534057911482812,sgd-log_loss-a1.67e-05,0.9642857142857143
heldout-iris-setosa,tabular,knn-k31-uniform,1.0,sgd-log_loss-a0.01,1.0,1.0,0.0,logreg-c0.001-b0,1.0
heldout-text-computer-systems,text,sgd-log_loss-a4.64e-06,0.7976190476190477,sgd-log_loss-a0.01,0.8380952380952381,0.7061607142857143,0.14044739677291995,sgd-log_loss-a0.01,0.8380952380952381
heldout-text-religion,text,linearsvc-c12.743-b1,0.7738095238095238,sgd-log_loss-a0.01,0.8214285714285714,0.6842583333333333,0.10558010433034093,sgd-log_loss-a0.01,0.8214285714285714
heldout-text-science,text,linearsvc-c12.743-b1,0.8761904761904762,sgd-log_loss-a0.01,0.9023809523809524,0.7507988095238095,0.160372270629095,sgd-log_loss-a0.01,0.9023809523809524
heldout-wine-class-zero,tabular,sgd-log_loss-a0.01,1.0,sgd-log_loss-a0.01,1.0,0.9738055555555555,0.03798234891913227,knn-k15-distance,1.0

````
