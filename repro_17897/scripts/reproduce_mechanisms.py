#!/usr/bin/env python3
"""Independent proxy reproduction of ML-Agent's diversity and reward mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_FILE = globals().get("__file__")
ROOT = Path(SCRIPT_FILE).resolve().parents[1] if SCRIPT_FILE else Path.cwd()
OUT = ROOT / "outputs"


def candidate_ideas() -> list[dict[str, str]]:
    axes = {
        "data": [
            "standardize numeric features", "apply robust scaling", "add MixUp augmentation",
            "add CutMix augmentation", "balance classes with weighted sampling", "use focal sampling",
            "impute missing values by median", "add missing-value indicators", "remove extreme outliers",
            "winsorize heavy-tailed columns", "expand text with synonym augmentation", "use random erasing",
            "crop images more aggressively", "increase image resolution", "add color jitter",
            "use stratified folds", "use group-aware folds", "engineer interaction features",
            "apply log transforms", "apply target encoding", "use frequency encoding",
            "tokenize with character n-grams", "tokenize with word bigrams", "apply label smoothing",
            "oversample minority labels", "undersample majority labels", "normalize each image channel",
            "denoise inputs with a median filter", "augment with Gaussian noise", "add polynomial features",
            "drop highly correlated features", "select features by mutual information", "cache preprocessed tensors",
            "use test-time augmentation", "blend multiple augmentations", "shuffle training examples each epoch",
            "bucket examples by sequence length", "truncate long sequences adaptively", "use mixout regularization",
            "construct pseudo-labels from confident predictions",
        ],
        "model": [
            "replace the linear model with gradient boosting", "use a residual convolutional network",
            "switch to a vision transformer", "add squeeze-and-excitation blocks", "use a deeper MLP",
            "widen hidden layers", "add batch normalization", "replace batch norm with layer norm",
            "add dropout to the classifier", "use stochastic depth", "initialize from pretrained weights",
            "freeze the backbone initially", "unfreeze the final backbone stage", "add an attention pooling head",
            "use bidirectional recurrent layers", "replace recurrence with temporal convolutions",
            "use a linear SVM", "use an elastic-net classifier", "use CatBoost for categorical features",
            "use LightGBM with leaf-wise growth", "ensemble diverse model families", "stack out-of-fold predictions",
            "average checkpoints from late epochs", "use a larger embedding dimension", "tie input and output embeddings",
            "add residual connections", "add a feature-cross network", "use a mixture-of-experts head",
            "add multi-scale image features", "use depthwise separable convolutions", "use cosine classifier weights",
            "add a calibration layer", "distill from a larger teacher", "use a sparse linear baseline",
            "add a U-Net decoder", "replace ReLU with GELU", "replace GELU with SiLU",
            "use an ordinal regression head", "use pairwise ranking loss", "add a graph message-passing layer",
        ],
        "learning": [
            "switch from Adam to AdamW", "use stochastic gradient descent with momentum",
            "lower the initial learning rate", "increase the initial learning rate", "use cosine learning-rate decay",
            "use one-cycle learning-rate scheduling", "add linear warmup", "clip gradient norms",
            "accumulate gradients over more batches", "increase the batch size", "decrease the batch size",
            "train for more epochs", "enable early stopping", "optimize weight decay", "add L1 regularization",
            "add L2 regularization", "use exponential moving averages", "apply sharpness-aware minimization",
            "use focal loss", "use Huber loss", "use class-weighted cross entropy", "optimize label smoothing strength",
            "tune the decision threshold", "calibrate with temperature scaling", "use mixed-precision training",
            "seed multiple training runs", "perform Bayesian hyperparameter search", "use successive halving",
            "monitor validation metric each epoch", "save the best validation checkpoint", "anneal dropout over training",
            "use curriculum learning", "use hard-example mining", "restart training from the best checkpoint",
            "average predictions across folds", "optimize directly for the evaluation metric", "use adversarial training",
            "use knowledge distillation loss", "use gradient centralization", "decay augmentation strength over time",
        ],
    }
    return [{"axis": axis, "text": text} for axis, values in axes.items() for text in values]


def hash_embeddings(texts: Iterable[str], dim: int = 384) -> np.ndarray:
    if not isinstance(texts, list):
        texts = list(texts)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = text.lower().replace("-", " ").split()
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 else -1.0
            matrix[row, index] += sign
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed(texts: list[str], backend: str, model_id: str) -> np.ndarray:
    if backend == "hash":
        return hash_embeddings(texts)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=True))


def cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)


def select_v1_mean_distance(distances: np.ndarray, k: int) -> list[int]:
    means = distances.sum(axis=1) / (len(distances) - 1)
    return np.argsort(means)[-k:][::-1].tolist()


def select_v2_fps(distances: np.ndarray, k: int) -> list[int]:
    mean_distance = distances.sum(axis=1) / (len(distances) - 1)
    selected = [int(np.argmax(mean_distance))]
    while len(selected) < k:
        min_to_selected = distances[:, selected].min(axis=1)
        min_to_selected[selected] = -1.0
        selected.append(int(np.argmax(min_to_selected)))
    return selected


def diversity_metrics(indices: list[int], distances: np.ndarray, ideas: list[dict[str, str]]) -> dict[str, object]:
    sub = distances[np.ix_(indices, indices)]
    upper = sub[np.triu_indices(len(indices), k=1)]
    counts = Counter(ideas[i]["axis"] for i in indices)
    return {
        "mean_pairwise_cosine_distance": float(upper.mean()),
        "minimum_pairwise_cosine_distance": float(upper.min()),
        "axis_coverage": len(counts),
        "axis_counts": dict(sorted(counts.items())),
    }


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def reward_v1(*, valid: bool, editing: bool, outcome: str, before: float, after: float, baseline: float, best: float) -> float:
    if not valid or outcome == "error":
        return 0.0
    if not editing or outcome == "corner":
        return 0.5
    if outcome != "success":
        raise ValueError(f"unknown outcome: {outcome}")
    alpha = 100.0 / (best - baseline)
    return sigmoid(alpha * (after - before))


def reward_v2(*, valid: bool, editing: bool, outcome: str, before: float, after: float, baseline: float, best: float) -> float:
    if not valid or outcome == "error":
        return -1.0
    if not editing or outcome == "corner":
        return 0.0
    if outcome != "success":
        raise ValueError(f"unknown outcome: {outcome}")
    return (after - before) / (best - baseline)


def reward_checks() -> dict[str, object]:
    common = {"before": 0.60, "baseline": 0.50, "best": 1.00}
    cases = {
        "invalid": {"valid": False, "editing": False, "outcome": "error", "after": 0.60},
        "runtime_error": {"valid": True, "editing": True, "outcome": "error", "after": 0.60},
        "corner_case": {"valid": True, "editing": True, "outcome": "corner", "after": 0.60},
        "valid_non_edit": {"valid": True, "editing": False, "outcome": "success", "after": 0.60},
        "successful_improvement": {"valid": True, "editing": True, "outcome": "success", "after": 0.61},
        "successful_no_change": {"valid": True, "editing": True, "outcome": "success", "after": 0.60},
        "successful_regression": {"valid": True, "editing": True, "outcome": "success", "after": 0.59},
    }
    result = {}
    for name, case in cases.items():
        result[name] = {
            "v1": reward_v1(**case, **common),
            "v2": reward_v2(**case, **common),
        }
    assert result["invalid"] == {"v1": 0.0, "v2": -1.0}
    assert result["corner_case"] == {"v1": 0.5, "v2": 0.0}
    assert result["successful_improvement"]["v1"] > 0.5
    assert result["successful_no_change"]["v1"] == 0.5
    assert result["successful_regression"]["v1"] < 0.5
    return result


def make_plot(random_means: list[float], selected: dict[str, dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(random_means, bins=35, color="#cbd5e1", edgecolor="white", label="Random 10-of-120")
    colors = {"v1_mean_distance": "#2563eb", "v2_fps": "#dc2626"}
    for name, color in colors.items():
        value = selected[name]["metrics"]["mean_pairwise_cosine_distance"]
        ax.axvline(value, color=color, linewidth=2.5, label=f"{name}: {value:.3f}")
    ax.set_xlabel("Mean pairwise cosine distance")
    ax.set_ylabel("Random selections")
    ax.set_title("Exploration-enriched selection versus random subsets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def maybe_trackio(result: dict[str, object], args: argparse.Namespace) -> None:
    space_id = os.getenv("TRACKIO_SPACE_ID")
    if not space_id:
        return
    import trackio

    trackio.init(
        project="ml-agent-repro-17897",
        name=args.run_name,
        space_id=space_id,
        config={"backend": args.backend, "model_id": args.model_id, "random_trials": args.random_trials},
    )
    for name, payload in result["selection"].items():
        trackio.log({
            f"{name}/mean_pairwise_distance": payload["metrics"]["mean_pairwise_cosine_distance"],
            f"{name}/min_pairwise_distance": payload["metrics"]["minimum_pairwise_cosine_distance"],
            f"{name}/axis_coverage": payload["metrics"]["axis_coverage"],
        })
    trackio.log({
        "random/mean_pairwise_distance": result["random_baseline"]["mean"],
        "random/p95_pairwise_distance": result["random_baseline"]["p95"],
    })
    trackio.finish()


def maybe_upload(output_files: list[Path]) -> str | None:
    repo_id = os.getenv("HF_RESULTS_REPO")
    if not repo_id:
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    for path in output_files:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=f"gpu_run/{path.name}",
            commit_message=f"Add ML-Agent reproduction artifact {path.name}",
        )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hash", "sentence-transformer"], default="hash")
    parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17897)
    parser.add_argument("--run-name", default="local-smoke")
    parser.add_argument("--output-prefix", default="local")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ideas = candidate_ideas()
    assert len(ideas) == 120
    embeddings = embed([item["text"] for item in ideas], args.backend, args.model_id)
    distances = cosine_distances(embeddings)

    selections = {
        "v1_mean_distance": select_v1_mean_distance(distances, 10),
        "v2_fps": select_v2_fps(distances, 10),
    }
    random_metrics = []
    for _ in range(args.random_trials):
        indices = rng.choice(len(ideas), size=10, replace=False).tolist()
        random_metrics.append(diversity_metrics(indices, distances, ideas)["mean_pairwise_cosine_distance"])

    selection_payload = {}
    for name, indices in selections.items():
        selection_payload[name] = {
            "indices": indices,
            "ideas": [ideas[i] for i in indices],
            "metrics": diversity_metrics(indices, distances, ideas),
        }

    result = {
        "scope": "Independent mechanism reproduction on a disclosed 120-idea synthetic corpus; not the unreleased training pool.",
        "backend": args.backend,
        "model_id": args.model_id if args.backend == "sentence-transformer" else None,
        "seed": args.seed,
        "candidate_count": len(ideas),
        "selection_size": 10,
        "random_trials": args.random_trials,
        "selection": selection_payload,
        "random_baseline": {
            "mean": float(np.mean(random_metrics)),
            "std": float(np.std(random_metrics)),
            "p95": float(np.quantile(random_metrics, 0.95)),
            "max": float(np.max(random_metrics)),
        },
        "reward_cases": reward_checks(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{args.output_prefix}_mechanism_results.json"
    plot_path = OUT / f"{args.output_prefix}_diversity.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    make_plot(random_metrics, selection_payload, plot_path)
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))
    maybe_trackio(result, args)
    hub_url = maybe_upload([json_path, plot_path])
    if hub_url:
        result["hub_results"] = hub_url
        json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
